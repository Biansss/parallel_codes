#pragma once

#include <vector>
#include <queue>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <random>
#include <mpi.h>
#include "pq_simple.h"
#include "IVF.h"

// IVFPQ MPI版本 - 结合倒排文件索引和乘积量化的混合索引结构
class IVFPQ_MPI {
private:
    int rank, size; // MPI进程号和总进程数

public:
    // 基础参数
    size_t dim;                // 向量维度
    size_t nlist;              // 聚类中心数量(倒排表数量)
    size_t code_size;          // PQ编码的大小(字节)
    bool by_residual;          // 是否对残差进行量化
    size_t n_vectors;          // 索引中的向量数量
    const float* raw_vectors;  // 原始向量数据指针(用于重排序)

    // 量化器
    IVF* ivf;                 // 倒排文件索引
    ProductQuantizer_Simple pq;      // 产品量化器

    // 存储结构
    struct InvertedList {
        std::vector<int> ids;           // 向量ID列表
        std::vector<uint8_t> codes;     // PQ编码列表
    };
    std::vector<InvertedList> invlists; // 倒排表

    // 预计算表相关
    bool use_precomputed_table;        // 是否使用预计算表
    std::vector<float> precomputed_table; // 预计算距离表
    
    // 向量重排序相关
    bool vectors_reordered;              // 标记向量是否已重排
    std::vector<float> reordered_vectors; // 重排后的向量存储
    std::vector<int> original_to_new_ids; // 原始ID到新ID的映射
    std::vector<int> new_to_original_ids; // 新ID到原始ID的映射

    // 辅助函数 - 计算残差
    void compute_residual(const float* x, float* residual, int list_no) const {
        const float* centroid = ivf->centroids + (size_t)list_no * dim;
        for (size_t i = 0; i < dim; i++) {
            residual[i] = x[i] - centroid[i];
        }
    }

    // 预计算表计算
    void precompute_tables() {
        if (!use_precomputed_table) return;

        size_t M = pq.get_M();
        size_t Ks = pq.get_Ks();
        
        // 为聚类中心和PQ子空间中心之间的距离表分配空间
        precomputed_table.resize(nlist * M * Ks);

        // 计算每个子空间聚类中心的L2范数
        std::vector<float> r_norms(M * Ks, 0);
        size_t dsub = dim / M;
        for (size_t m = 0; m < M; m++) {
            for (size_t j = 0; j < Ks; j++) {
                float norm = 0;
                const float* centroids = pq.centroids.data() + (m * Ks + j) * dsub;
                for (size_t d = 0; d < dsub; d++) {
                    norm += centroids[d] * centroids[d];
                }
                r_norms[m * Ks + j] = norm;
            }
        }

        // 预计算每个聚类中心与每个子空间的点积表
        std::vector<float> centroid(dim);
        for (size_t i = 0; i < nlist; i++) {
            // 重建聚类中心向量
            const float* cluster_centroid = ivf->centroids + i * dim;
            
            // 计算聚类中心与PQ子空间中心的点积表
            float* tab = &precomputed_table[i * M * Ks];
            pq.compute_distance_tables(cluster_centroid, tab);

            // 加上残差范数项 (残差范数项对应第二项)
            for (size_t j = 0; j < M * Ks; j++) {
                tab[j] += 2.0 * r_norms[j];  // 2.0 因子来自于距离计算公式
            }
        }
    }
    
    // 重新排序向量以提高缓存局部性
    void reorder_vectors() {
        // 确保有原始向量数据
        if (!raw_vectors || n_vectors == 0 || vectors_reordered) return;
        
        // 分配存储空间
        reordered_vectors.resize(n_vectors * dim);
        original_to_new_ids.resize(n_vectors, -1);
        new_to_original_ids.resize(n_vectors, -1);
        
        // 计算每个簇的起始偏移量
        std::vector<size_t> list_offsets(nlist, 0);
        size_t offset = 0;
        for (size_t i = 0; i < nlist; i++) {
            list_offsets[i] = offset;
            offset += invlists[i].ids.size();
        }
        
        // 复制向量并构建ID映射
        std::vector<size_t> list_counters = list_offsets;
        for (size_t list_id = 0; list_id < nlist; list_id++) {
            auto& list = invlists[list_id];
            for (size_t j = 0; j < list.ids.size(); j++) {
                int original_id = list.ids[j];
                int new_id = static_cast<int>(list_counters[list_id]++);
                
                // 复制向量数据
                memcpy(reordered_vectors.data() + new_id * dim,
                       raw_vectors + original_id * dim,
                       dim * sizeof(float));
                
                // 更新ID映射
                original_to_new_ids[original_id] = new_id;
                new_to_original_ids[new_id] = original_id;
                
                // 更新倒排表中的ID引用
                list.ids[j] = new_id;
            }
        }
        
        // 更新向量指针
        raw_vectors = reordered_vectors.data();
        vectors_reordered = true;
        
        if (rank == 0) {
            std::cout << "向量内存重排优化完成，将提高缓存命中率" << std::endl;
        }
    }

    // MPI广播支持函数
void broadcast_from_rank0() {
    // 广播基本参数
    MPI_Bcast(&dim, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nlist, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&code_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&by_residual, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_vectors, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&use_precomputed_table, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&vectors_reordered, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // 广播PQ量化器
    pq.broadcast_from_rank0();

    // 广播IVF聚类中心
    if (rank != 0) {
        if (ivf) delete ivf;
        ivf = new IVF(dim, nlist);
    }
    
    // 确保聚类中心已分配
    if (rank != 0 && !ivf->centroids) {
        ivf->centroids = new float[nlist * dim];
    }
    
    MPI_Bcast(ivf->centroids, nlist * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 广播倒排表结构
    if (rank != 0) {
        invlists.clear();
        invlists.resize(nlist);
    }

    for (size_t i = 0; i < nlist; i++) {
        // 广播ID列表大小
        size_t ids_size = 0;
        if (rank == 0) {
            ids_size = invlists[i].ids.size();
        }
        MPI_Bcast(&ids_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        
        // 调整接收进程的容器大小
        if (rank != 0) {
            invlists[i].ids.resize(ids_size);
        }
        
        // 广播ID数据
        if (ids_size > 0) {
            MPI_Bcast(invlists[i].ids.data(), ids_size, MPI_INT, 0, MPI_COMM_WORLD);
        }

        // 广播编码数据大小
        size_t codes_size = 0;
        if (rank == 0) {
            codes_size = invlists[i].codes.size();
        }
        MPI_Bcast(&codes_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        
        // 调整接收进程的容器大小
        if (rank != 0) {
            invlists[i].codes.resize(codes_size);
        }
        
        // 广播编码数据
        if (codes_size > 0) {
            MPI_Bcast(invlists[i].codes.data(), codes_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
        }
    }

    // 广播预计算表
    if (use_precomputed_table) {
        size_t table_size = 0;
        if (rank == 0) {
            table_size = precomputed_table.size();
        }
        MPI_Bcast(&table_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            precomputed_table.resize(table_size);
        }
        
        if (table_size > 0) {
            MPI_Bcast(precomputed_table.data(), table_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
    }

    // 广播重排序数据
    if (vectors_reordered) {
        size_t reordered_size = n_vectors * dim;
        
        if (rank != 0) {
            reordered_vectors.resize(reordered_size);
            original_to_new_ids.resize(n_vectors);
            new_to_original_ids.resize(n_vectors);
        }
        
        if (reordered_size > 0) {
            MPI_Bcast(reordered_vectors.data(), reordered_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
        
        if (n_vectors > 0) {
            MPI_Bcast(original_to_new_ids.data(), n_vectors, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(new_to_original_ids.data(), n_vectors, MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        // 设置向量指针
        if (rank != 0) {
            raw_vectors = reordered_vectors.data();
        }
    }
    
    // 同步所有进程，确保广播完成
    MPI_Barrier(MPI_COMM_WORLD);
}

public:
    // 构造函数
    IVFPQ_MPI(size_t d, size_t nlist = 100, size_t m = 8, size_t ks = 256) 
        : dim(d), nlist(nlist), by_residual(true), n_vectors(0), raw_vectors(nullptr), 
          pq(d, m, ks), use_precomputed_table(false), vectors_reordered(false) {
        
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // 确保维度能被子空间数量整除
        assert(d % m == 0);
        
        // 创建IVF索引
        ivf = new IVF(d, nlist);
        
        // 初始化倒排表
        invlists.resize(nlist);
        
        // PQ编码的字节数
        code_size = pq.get_code_size();
    }

    // 析构函数
    ~IVFPQ_MPI() {
        if (ivf) delete ivf;
    }

    // 训练索引
    void train(const float* vectors, size_t n) {
        if (rank == 0) {
            // 1. 训练IVF聚类中心
            ivf->train(vectors, n);

            // 2. 计算所有向量的残差
            std::vector<float> residuals(n * dim);
            std::vector<int> list_nos(n);

            // 分配每个向量到最近的聚类中心
            for (size_t i = 0; i < n; i++) {
                // 寻找最近的聚类中心
                list_nos[i] = ivf->findClosestCentroid(vectors + i * dim);
                
                // 计算残差
                if (by_residual) {
                    compute_residual(vectors + i * dim, residuals.data() + i * dim, list_nos[i]);
                } else {
                    // 如果不使用残差，直接使用原始向量
                    memcpy(residuals.data() + i * dim, vectors + i * dim, dim * sizeof(float));
                }
            }
            
            // 3. 使用残差训练PQ量化器
            pq.train(residuals.data(), n);

            // 4. 如果使用预计算表，则预计算
            if (use_precomputed_table) {
                precompute_tables();
            }
        }

        // 广播训练好的模型到所有进程
        broadcast_from_rank0();
    }

    // 添加向量到索引
    void add(const float* vectors, size_t n, const int* ids = nullptr) {
        if (rank == 0) {
            // 设置原始向量指针（用于重排序）
            raw_vectors = vectors;
            n_vectors += n;
            
            // 为了批处理，先存储所有向量的聚类分配和残差
            std::vector<float> residuals(n * dim);
            std::vector<int> list_nos(n);
            
            // 分配向量到聚类中心
            for (size_t i = 0; i < n; i++) {
                const float* x = vectors + i * dim;
                
                // 找到最近的聚类中心
                list_nos[i] = ivf->findClosestCentroid(x);
                
                // 计算残差
                if (by_residual) {
                    compute_residual(x, residuals.data() + i * dim, list_nos[i]);
                } else {
                    memcpy(residuals.data() + i * dim, x, dim * sizeof(float));
                }
            }
            
            // 批量处理向量编码
            std::vector<uint8_t> codes(n * code_size);
            for (size_t i = 0; i < n; i++) {
                pq.encode(residuals.data() + i * dim, codes.data() + i * code_size);
            }
            
            // 添加到倒排表中
            for (size_t i = 0; i < n; i++) {
                int list_no = list_nos[i];
                int id = ids ? ids[i] : n_vectors - n + i;
                
                // 添加ID到倒排表
                invlists[list_no].ids.push_back(id);
                
                // 添加编码到倒排表
                size_t offset = invlists[list_no].codes.size();
                invlists[list_no].codes.resize(offset + code_size);
                memcpy(invlists[list_no].codes.data() + offset, 
                       codes.data() + i * code_size, 
                       code_size);
            }
            
            // 添加向量后重置重排序状态，因为向量数据已更新
            vectors_reordered = false;
        }

        // 广播更新后的数据到所有进程
        broadcast_from_rank0();
    }

    // 启用/禁用预计算表
    void set_precomputed_table(bool use) {
        if (rank == 0) {
            use_precomputed_table = use;
            if (use && invlists.size() > 0) {
                precompute_tables();
            }
        }
        
        // 广播更新
        broadcast_from_rank0();
    }
    
    // 开启向量重排序优化
    void enable_vector_reordering() {
        if (rank == 0) {
            if (!vectors_reordered && raw_vectors && n_vectors > 0) {
                reorder_vectors();
            }
        }
        
        // 广播重排序后的数据
        broadcast_from_rank0();
    }

    // MPI并行搜索函数
    std::priority_queue<std::pair<float, int>> search_mpi(
            const float* query, 
            size_t k, 
            size_t nprobe = 10,
            bool rerank = true,
            size_t rerank_count = 500) const {
        
        // 确保nprobe不超过聚类数量
        nprobe = std::min(nprobe, nlist);
        
        // 设置重排序候选数量，如果未指定则使用k的3倍
        if (rerank_count == 0) {
            rerank_count = rerank ? k * 3 : k;
        }
        // 确保重排序候选数量不小于k
        rerank_count = std::max(rerank_count, k);
        
        // 1. 所有进程计算聚类中心距离（保证一致性）
        std::vector<std::pair<float, int>> cluster_distances;
        cluster_distances.reserve(nlist);
        
        for (size_t i = 0; i < nlist; i++) {
            float dist = ivf->computeDistanceOptimized(query, ivf->centroids + i * dim);
            cluster_distances.push_back({dist, static_cast<int>(i)});
        }
        
        // 部分排序，获取前nprobe个最近的簇
        std::partial_sort(
            cluster_distances.begin(), 
            cluster_distances.begin() + nprobe,
            cluster_distances.end()
        );
        
        // 2. MPI任务分配：将nprobe个簇分配给不同进程
        std::vector<int> my_clusters;
        for (size_t i = rank; i < nprobe; i += size) {
            my_clusters.push_back(cluster_distances[i].second);
        }
        
        // 3. 每个进程在分配给它的簇中搜索
        std::priority_queue<std::pair<float, int>> local_results;
        std::vector<float> residual(dim);
        
        // 为每个子空间预计算距离表
        std::vector<float> distance_tables(pq.get_M() * pq.get_Ks());
        
        for (int cluster_id : my_clusters) {
            float coarse_dist = 0;
            
            // 找到对应的粗排距离
            for (size_t p = 0; p < nprobe; p++) {
                if (cluster_distances[p].second == cluster_id) {
                    coarse_dist = cluster_distances[p].first;
                    break;
                }
            }
            
            // 计算查询向量与该聚类中心的残差
            if (by_residual) {
                compute_residual(query, residual.data(), cluster_id);
                
                // 计算残差的距离表
                pq.compute_distance_tables(residual.data(), distance_tables.data());
            } else {
                // 如果不使用残差，直接计算原始向量的距离表
                pq.compute_distance_tables(query, distance_tables.data());
            }
            
            // 如果使用预计算表，则应用预计算表优化
            if (use_precomputed_table) {
                const float* precomputed = &precomputed_table[cluster_id * pq.get_M() * pq.get_Ks()];
                // 结合预计算表和距离表
                for (size_t j = 0; j < pq.get_M() * pq.get_Ks(); j++) {
                    distance_tables[j] = precomputed[j] - 2 * distance_tables[j];
                }
                // 加上第一项：查询点到聚类中心的距离
                for (size_t j = 0; j < pq.get_M() * pq.get_Ks(); j++) {
                    distance_tables[j] += coarse_dist;
                }
            }
            
            // 搜索当前倒排表
            const InvertedList& list = invlists[cluster_id];
            for (size_t j = 0; j < list.ids.size(); j++) {
                int id = list.ids[j];
                const uint8_t* code = list.codes.data() + j * code_size;
                
                // 使用查找表快速计算距离
                float dist = 0;
                for (size_t m = 0; m < pq.get_M(); m++) {
                    uint8_t centroid_idx = code[m];
                    dist += distance_tables[m * pq.get_Ks() + centroid_idx];
                }
                
                // 收集前rerank_count个最近邻
                if (local_results.size() < rerank_count) {
                    local_results.push({dist, id});
                } else if (dist < local_results.top().first) {
                    local_results.pop();
                    local_results.push({dist, id});
                }
            }
        }
        
        // 4. 收集各进程的局部结果
        std::vector<std::pair<float, int>> local_vec;
        while (!local_results.empty()) {
            local_vec.push_back(local_results.top());
            local_results.pop();
        }
        
        // 收集所有进程的结果大小
        int local_size = local_vec.size();
        std::vector<int> all_sizes(size);
        MPI_Allgather(&local_size, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        // 计算偏移量
        std::vector<int> offsets(size, 0);
        int total_results = 0;
        for (int i = 0; i < size; i++) {
            offsets[i] = total_results;
            total_results += all_sizes[i];
        }
        
        // 准备发送数据（距离和ID分开收集）
        std::vector<float> local_dists;
        std::vector<int> local_ids;
        for (const auto& pair : local_vec) {
            local_dists.push_back(pair.first);
            local_ids.push_back(pair.second);
        }
        
        // 收集所有结果
        std::vector<float> all_dists(total_results);
        std::vector<int> all_ids(total_results);
        
        MPI_Allgatherv(local_dists.data(), local_size, MPI_FLOAT,
                       all_dists.data(), all_sizes.data(), offsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgatherv(local_ids.data(), local_size, MPI_INT,
                       all_ids.data(), all_sizes.data(), offsets.data(), MPI_INT, MPI_COMM_WORLD);
        
        // 5. 重排序步骤（所有进程参与）
        std::priority_queue<std::pair<float, int>> results;
        
        if (rerank && raw_vectors) {
            // 收集候选并排序（距离最小的在前）
            std::vector<std::pair<float, int>> candidates;
            for (int i = 0; i < total_results; i++) {
                candidates.push_back({all_dists[i], all_ids[i]});
            }
            std::sort(candidates.begin(), candidates.end());
            
            // 限制重排序候选数量
            size_t max_rerank = std::min(candidates.size(), rerank_count);
            
            // 使用原始向量计算确切距离进行重排序
            for (size_t i = 0; i < max_rerank; i++) {
                int id = candidates[i].second;
                
                // 计算精确距离
                float exact_dist = 0;
                for (size_t d = 0; d < dim; d++) {
                    float diff = query[d] - raw_vectors[id * dim + d];
                    exact_dist += diff * diff;
                }
                
                // 保留k个最近的向量
                if (results.size() < k) {
                    results.push({exact_dist, id});
                } else if (exact_dist < results.top().first) {
                    results.pop();
                    results.push({exact_dist, id});
                }
            }
            
            // 如果向量已重排，将内部ID映射回原始ID
            if (vectors_reordered) {
                std::priority_queue<std::pair<float, int>> mapped_results;
                while (!results.empty()) {
                    auto [dist, id] = results.top();
                    results.pop();
                    // 将内部ID映射回原始ID
                    int original_id = new_to_original_ids[id];
                    mapped_results.push({dist, original_id});
                }
                return mapped_results;
            }
        } else {
            // 如果不进行重排序，直接选择top-k
            for (int i = 0; i < total_results; i++) {
                if (results.size() < k) {
                    results.push({all_dists[i], all_ids[i]});
                } else if (all_dists[i] < results.top().first) {
                    results.pop();
                    results.push({all_dists[i], all_ids[i]});
                }
            }
            
            // 如果向量已重排，且不做重排序，仍需要将ID映射回原始ID
            if (vectors_reordered) {
                std::priority_queue<std::pair<float, int>> mapped_results;
                std::vector<std::pair<float, int>> temp;
                while (!results.empty()) {
                    temp.push_back(results.top());
                    results.pop();
                }
                for (auto it = temp.rbegin(); it != temp.rend(); ++it) {
                    mapped_results.push({it->first, new_to_original_ids[it->second]});
                }
                return mapped_results;
            }
        }
        
        return results;
    }
    
    // 保存索引（仅主进程）
    bool save(const std::string& filename) const {
        if (rank != 0) return true;
        
        // 保存IVF部分
        std::string ivf_filename = filename + ".ivf";
        ivf->saveIndex(ivf_filename);
        
        // 保存PQ量化器
        std::string pq_filename = filename + ".pq";
        if (!pq.save(pq_filename)) return false;
        
        // 保存编码数据和元数据
        std::string data_filename = filename + ".data";
        std::ofstream out(data_filename, std::ios::binary);
        if (!out.is_open()) return false;
        
        // 写入基本参数
        out.write((char*)&dim, sizeof(dim));
        out.write((char*)&nlist, sizeof(nlist));
        out.write((char*)&code_size, sizeof(code_size));
        out.write((char*)&by_residual, sizeof(by_residual));
        out.write((char*)&n_vectors, sizeof(n_vectors));
        out.write((char*)&use_precomputed_table, sizeof(use_precomputed_table));
        out.write((char*)&vectors_reordered, sizeof(vectors_reordered));
        
        // 写入倒排表数据
        for (size_t i = 0; i < nlist; i++) {
            // 写入ID列表
            size_t ids_size = invlists[i].ids.size();
            out.write((char*)&ids_size, sizeof(ids_size));
            if (ids_size > 0) {
                out.write((char*)invlists[i].ids.data(), ids_size * sizeof(int));
            }
            
            // 写入编码数据
            size_t codes_size = invlists[i].codes.size();
            out.write((char*)&codes_size, sizeof(codes_size));
            if (codes_size > 0) {
                out.write((char*)invlists[i].codes.data(), codes_size);
            }
        }
        
        // 如果使用预计算表，也保存它
        if (use_precomputed_table) {
            size_t table_size = precomputed_table.size();
            out.write((char*)&table_size, sizeof(table_size));
            out.write((char*)precomputed_table.data(), table_size * sizeof(float));
        }
        
        // 如果向量已经重排序，保存ID映射和重排后的向量数据
        if (vectors_reordered) {
            // 保存ID映射
            out.write((char*)original_to_new_ids.data(), n_vectors * sizeof(int));
            out.write((char*)new_to_original_ids.data(), n_vectors * sizeof(int));
            
            // 保存重排后的向量数据
            out.write((char*)reordered_vectors.data(), n_vectors * dim * sizeof(float));
        }
        
        return true;
    }
    
    // 加载索引
    bool load(const std::string& filename, const float* vectors = nullptr) {
        if (rank == 0) {
            // 加载IVF部分
            std::string ivf_filename = filename + ".ivf";
            if (ivf) delete ivf;
            ivf = new IVF(dim);  // 临时创建，实际参数会从文件加载
            ivf->loadIndex(ivf_filename);
            
            // 加载PQ量化器
            std::string pq_filename = filename + ".pq";
            if (!pq.load(pq_filename)) return false;
            
            // 加载数据和元数据
            std::string data_filename = filename + ".data";
            std::ifstream in(data_filename, std::ios::binary);
            if (!in.is_open()) return false;
            
            // 读取基本参数
            in.read((char*)&dim, sizeof(dim));
            in.read((char*)&nlist, sizeof(nlist));
            in.read((char*)&code_size, sizeof(code_size));
            in.read((char*)&by_residual, sizeof(by_residual));
            in.read((char*)&n_vectors, sizeof(n_vectors));
            in.read((char*)&use_precomputed_table, sizeof(use_precomputed_table));
            in.read((char*)&vectors_reordered, sizeof(vectors_reordered));
            
            // 读取倒排表数据
            invlists.resize(nlist);
            for (size_t i = 0; i < nlist; i++) {
                // 读取ID列表
                size_t ids_size;
                in.read((char*)&ids_size, sizeof(ids_size));
                invlists[i].ids.resize(ids_size);
                if (ids_size > 0) {
                    in.read((char*)invlists[i].ids.data(), ids_size * sizeof(int));
                }
                
                // 读取编码数据
                size_t codes_size;
                in.read((char*)&codes_size, sizeof(codes_size));
                invlists[i].codes.resize(codes_size);
                if (codes_size > 0) {
                    in.read((char*)invlists[i].codes.data(), codes_size);
                }
            }
            
            // 如果使用预计算表，也加载它
            if (use_precomputed_table) {
                size_t table_size;
                in.read((char*)&table_size, sizeof(table_size));
                precomputed_table.resize(table_size);
                in.read((char*)precomputed_table.data(), table_size * sizeof(float));
            }
            
            // 如果向量已经重排序，加载ID映射和重排后的向量数据
            if (vectors_reordered) {
                // 加载ID映射
                original_to_new_ids.resize(n_vectors);
                new_to_original_ids.resize(n_vectors);
                in.read((char*)original_to_new_ids.data(), n_vectors * sizeof(int));
                in.read((char*)new_to_original_ids.data(), n_vectors * sizeof(int));
                
                // 加载重排后的向量数据
                reordered_vectors.resize(n_vectors * dim);
                in.read((char*)reordered_vectors.data(), n_vectors * dim * sizeof(float));
                
                // 设置向量指针
                raw_vectors = reordered_vectors.data();
            } else {
                // 使用传入的原始向量
                raw_vectors = vectors;
            }
            
            in.close();
        }
        
        // 广播加载的数据到所有进程
        broadcast_from_rank0();
        
        return true;
    }
};

// 提供便捷的MPI搜索接口
std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> ivfpq_search_mpi(
    float* base, float* query, size_t base_number, size_t vecdim, 
    size_t k, size_t nprobe = 16, bool rerank = true,
    bool save_index = true, const std::string& index_file = "files/ivfpq_mpi.index",
    bool enable_reordering = true) {
    
    static IVFPQ_MPI* index = nullptr;
    static bool initialized = false;
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // 首次调用时初始化索引
    if (!initialized) {
        bool index_exists = false;
        
        if (rank == 0) {
            // 创建files目录
            #ifdef _WIN32
            if (system("if not exist files mkdir files") != 0) {
                std::cerr << "警告：无法创建files目录" << std::endl;
            }
            #else
            if (system("mkdir -p files") != 0) {
                std::cerr << "警告：无法创建files目录" << std::endl;
            }
            #endif
            
            std::ifstream f(index_file + ".data");
            index_exists = f.good();
            f.close();
        }
        
        // 广播索引文件是否存在的信息
        MPI_Bcast(&index_exists, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        
        // 选择适合的参数
        size_t nlist = std::min(size_t(std::sqrt(base_number) * 4), base_number / 10);
        if (nlist < 10) nlist = 10;
        if (nlist > 1000) nlist = 1000;
        
        size_t m = 8; // 子空间数
        
        index = new IVFPQ_MPI(vecdim, nlist, m);
        
        if (index_exists && save_index) {
            if (rank == 0) {
                std::cout << "Loading existing IVFPQ MPI index...\n";
            }
            index->load(index_file, base);
        } else {
            if (rank == 0) {
                std::cout << "Building new IVFPQ MPI index...\n";
            }
            index->train(base, base_number);
            index->add(base, base_number);
            
            // 启用预计算表
            index->set_precomputed_table(true);
            
            // 启用向量重排序优化
            if (enable_reordering) {
                index->enable_vector_reordering();
            }
            
            // 保存索引
            if (save_index) {
                index->save(index_file);
            }
        }
        initialized = true;
    }
    
    // 动态设置重排序候选数量为nprobe和k的函数
    size_t rerank_count = rerank ? std::min(base_number, nprobe * k * 8) : k;
    
    // 执行MPI并行搜索
    auto mpi_results = index->search_mpi(query, k, nprobe, rerank, rerank_count);
    
    // 转换为与现有代码兼容的格式（最小堆）
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> output;
    while (!mpi_results.empty()) {
        output.push({mpi_results.top().first, mpi_results.top().second});
        mpi_results.pop();
    }
    
    return output;
}