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
#include <chrono>
#include "pq.h"
#include "IVF.h"

// IVFPQ类 - 结合倒排文件索引和乘积量化的混合索引结构
class IVFPQ {
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
    ProductQuantizer pq;      // 产品量化器

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
        
        std::cout << "向量内存重排优化完成，将提高缓存命中率" << std::endl;
    }

public:
    // 构造函数
    IVFPQ(size_t d, size_t nlist = 100, size_t m = 8, size_t ks = 256) 
        : dim(d), nlist(nlist), by_residual(true), n_vectors(0), raw_vectors(nullptr), 
          pq(d, m, ks), use_precomputed_table(false), vectors_reordered(false) {
        
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
    ~IVFPQ() {
        if (ivf) delete ivf;
    }

    // 训练索引
    void train(const float* vectors, size_t n) {
        // 开始计时
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 1. 训练IVF聚类中心
        ivf->train(vectors, n);

        // 记录IVF训练完成时间
        auto ivf_time = std::chrono::high_resolution_clock::now();
        auto ivf_duration = std::chrono::duration_cast<std::chrono::milliseconds>(ivf_time - start_time).count();
        std::cout << "IVF 聚类训练耗时: " << ivf_duration << " ms" << std::endl;

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
        
        // 记录残差计算完成时间
        auto residual_time = std::chrono::high_resolution_clock::now();
        auto residual_duration = std::chrono::duration_cast<std::chrono::milliseconds>(residual_time - ivf_time).count();
        std::cout << "残差计算耗时: " << residual_duration << " ms" << std::endl;
        
        // 3. 使用残差训练PQ量化器
        pq.train(residuals.data(), n);

        // 记录PQ训练完成时间
        auto pq_time = std::chrono::high_resolution_clock::now();
        auto pq_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pq_time - residual_time).count();
        std::cout << "PQ 训练耗时: " << pq_duration << " ms" << std::endl;

        // 4. 如果使用预计算表，则预计算
        if (use_precomputed_table) {
            precompute_tables();
        }
        
        // 总训练时间
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "IVFPQ 总训练耗时: " << total_duration << " ms" << std::endl;
    }

    // 添加向量到索引
    void add(const float* vectors, size_t n, const int* ids = nullptr) {
        // 开始计时
        auto start_time = std::chrono::high_resolution_clock::now();
        
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
        
        // 记录聚类分配和残差计算时间
        auto clustering_time = std::chrono::high_resolution_clock::now();
        auto clustering_duration = std::chrono::duration_cast<std::chrono::milliseconds>(clustering_time - start_time).count();
        std::cout << "向量聚类分配和残差计算耗时: " << clustering_duration << " ms" << std::endl;
        
        // 批量处理向量编码
        std::vector<uint8_t> codes(n * code_size);
        for (size_t i = 0; i < n; i++) {
            pq.encode(residuals.data() + i * dim, codes.data() + i * code_size);
        }
        
        // 记录PQ编码时间
        auto encoding_time = std::chrono::high_resolution_clock::now();
        auto encoding_duration = std::chrono::duration_cast<std::chrono::milliseconds>(encoding_time - clustering_time).count();
        std::cout << "PQ 编码耗时: " << encoding_duration << " ms" << std::endl;
        
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
        
        // 记录倒排表更新时间
        auto update_time = std::chrono::high_resolution_clock::now();
        auto update_duration = std::chrono::duration_cast<std::chrono::milliseconds>(update_time - encoding_time).count();
        std::cout << "倒排表更新耗时: " << update_duration << " ms" << std::endl;
        
        // 添加向量后重置重排序状态，因为向量数据已更新
        vectors_reordered = false;
        
        // 总添加时间
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "总向量编码和索引添加耗时: " << total_duration << " ms (向量数量: " << n << ")" << std::endl;
    }

    // 启用/禁用预计算表
    void set_precomputed_table(bool use) {
        use_precomputed_table = use;
        if (use && invlists.size() > 0) {
            precompute_tables();
        }
    }
    
    // 开启向量重排序优化
    void enable_vector_reordering() {
        if (!vectors_reordered && raw_vectors && n_vectors > 0) {
            reorder_vectors();
        }
    }

    // 近似最近邻搜索，增加了重排序候选数量参数
    std::priority_queue<std::pair<float, int>> search(
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
        
        // 1. 找到nprobe个最近的聚类中心
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
        
        // 2. 在选定的簇中搜索，收集rerank_count个候选
        std::priority_queue<std::pair<float, int>> results;
        std::vector<float> residual(dim);
        
        // 为每个子空间预计算距离表
        std::vector<float> distance_tables(pq.get_M() * pq.get_Ks());
        
        for (size_t p = 0; p < nprobe; p++) {
            int cluster_id = cluster_distances[p].second;
            float coarse_dist = cluster_distances[p].first;
            
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
            
            // 搜索当前倒排表，收集rerank_count个候选
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
                if (results.size() < rerank_count) {
                    results.push({dist, id});
                } else if (dist < results.top().first) {
                    results.pop();
                    results.push({dist, id});
                }
            }
        }
        
        // 3. 重排序步骤
        if (rerank && raw_vectors) {
            // 收集候选
            std::vector<std::pair<float, int>> candidates;
            while (!results.empty()) {
                candidates.push_back(results.top());
                results.pop();
            }
            
            // 反转候选列表，使距离最小的在前面（优先重排精度更高的候选）
            std::reverse(candidates.begin(), candidates.end());
            
            // 使用原始向量计算确切距离进行重排序
            for (const auto& cand : candidates) {
                int id = cand.second;
                
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
                    auto top_result = results.top();
                    float dist = top_result.first;
                    int id = top_result.second;
                    results.pop();
                    // 将内部ID映射回原始ID
                    int original_id = new_to_original_ids[id];
                    mapped_results.push({dist, original_id});
                }
                return mapped_results;
            }
        } else if (results.size() > k) {
            // 如果不进行重排序但结果超过k个，需要截断
            std::vector<std::pair<float, int>> candidates;
            while (results.size() > k) {
                results.pop();  // 丢弃距离较大的结果
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
    
    // 保存索引
    bool save(const std::string& filename) const {
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
        
        return true;
    }
};

// 提供便捷的搜索接口
std::priority_queue<std::pair<float, int>> ivfpq_search(
    float* base, float* query, size_t base_number, size_t vecdim, 
    size_t k, size_t nprobe = 1, bool rerank = true,
    bool save_index = true, const std::string& index_file = "files/ivfpq.index",
    bool enable_reordering = true) {
    
    static IVFPQ* index = nullptr;
    static float* last_base = nullptr;
    static size_t last_base_number = 0;
    static size_t last_vecdim = 0;
    
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
    
    // 如果索引未初始化或数据变化，需要重建索引
    bool index_loaded = false;
    if (index == nullptr || base != last_base || 
        base_number != last_base_number || vecdim != last_vecdim) {
        
        // 尝试加载索引
        if (save_index) {
            IVFPQ* temp_index = new IVFPQ(vecdim);
            if (temp_index->load(index_file, base)) {
                // 释放旧索引
                if (index) delete index;
                index = temp_index;
                
                // 更新状态
                last_base = base;
                last_base_number = base_number;
                last_vecdim = vecdim;
                index_loaded = true;
            } else {
                delete temp_index;
            }
        }
        
        // 如果加载失败，创建新索引
        if (!index_loaded) {
            // 释放旧索引
            if (index) delete index;
            
            // 选择适合的参数
            size_t nlist = std::min(size_t(std::sqrt(base_number) * 4), base_number / 10);
            if (nlist < 10) nlist = 10;
            if (nlist > 1000) nlist = 1000;
            
            size_t m = 8; // 子空间数
            
            // 创建并训练索引
            index = new IVFPQ(vecdim, nlist, m);
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
            
            // 更新状态
            last_base = base;
            last_base_number = base_number;
            last_vecdim = vecdim;
        }
    }
    // 动态设置重排序候选数量为nprobe和k的函数
    size_t rerank_count = rerank ? std::min(base_number, nprobe * k * 8) : k;
    
    // 执行搜索，传入动态计算的重排序候选数量
    return index->search(query, k, nprobe, rerank, rerank_count);
}