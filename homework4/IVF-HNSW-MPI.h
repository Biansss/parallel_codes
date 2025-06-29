#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <memory>
#include <cassert>
#include <fstream>
#include <omp.h>
#include <unordered_map>
#include <queue>
#include <limits>
#include <cstring>
#include <thread>
#include <mutex>
#include <mpi.h>
#include "IVF.h"
#include "hnswlib/hnswlib.h"

// IVF+HNSW MPI并行版本 - 结合倒排文件索引和HNSW的混合索引结构
class IVFHNSW_MPI {
private:
    int rank, size; // MPI进程号和总进程数

public:
    // 基础参数
    size_t dim;                    // 向量维度
    size_t nlist;                  // IVF聚类数量
    size_t max_elements_per_cluster; // 每个聚类的最大元素数
    size_t ef_construction;        // HNSW构建参数
    size_t M;                      // HNSW连接数
    bool trained;                  // 是否已训练
    
    // IVF相关
    float* centroids = nullptr;    // 聚类中心
    std::vector<std::vector<int>> invlists; // 倒排列表
    
    // HNSW索引 - 每个聚类一个
    std::vector<hnswlib::HierarchicalNSW<float>*> hnsw_indices;
    std::vector<hnswlib::L2Space*> hnsw_spaces;
    
    // 数据管理
    const float* raw_data = nullptr;  // 原始数据指针
    size_t n_vectors = 0;             // 总向量数
    std::vector<size_t> cluster_sizes; // 每个聚类的大小
    
    // 线程安全相关
    std::vector<std::mutex*> cluster_mutexes;

    // 优化的距离计算
    float computeDistanceOptimized(const float* a, const float* b) const {
        float sum = 0.0f;
        size_t d = 0;
        
        // 4x循环展开
        for (; d + 3 < dim; d += 4) {
            float diff0 = a[d] - b[d];
            float diff1 = a[d+1] - b[d+1];
            float diff2 = a[d+2] - b[d+2];
            float diff3 = a[d+3] - b[d+3];
            
            sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
        }
        
        // 处理剩余元素
        for (; d < dim; d++) {
            float diff = a[d] - b[d];
            sum += diff * diff;
        }
        
        return sum;
    }

    // 找到最近的聚类中心
    int findClosestCentroid(const float* vec) const {
        int closest = 0;
        float min_dist = computeDistanceOptimized(vec, centroids);
        
        for (size_t i = 1; i < nlist; i++) {
            float dist = computeDistanceOptimized(vec, centroids + i * dim);
            if (dist < min_dist) {
                min_dist = dist;
                closest = static_cast<int>(i);
            }
        }
        
        return closest;
    }

    // k-means聚类训练
    void trainCentroids(const float* data, size_t n) {
        // 分配聚类中心内存
        centroids = new float[nlist * dim];
        
        // 随机选择初始聚类中心
        std::vector<int> centroid_ids;
        selectInitialCentroids(data, n, centroid_ids);
        
        // 复制初始中心点
        for (size_t i = 0; i < nlist; i++) {
            memcpy(centroids + i * dim, data + (size_t)centroid_ids[i] * dim, dim * sizeof(float));
        }
        
        // k-means迭代
        std::vector<int> assignments(n, -1);
        std::vector<int> cluster_counts(nlist, 0);
        
        const int max_iterations = 20;
        for (int iter = 0; iter < max_iterations; iter++) {
            bool converged = true;
            
            // 分配点到最近聚类中心
            #pragma omp parallel for
            for (size_t i = 0; i < n; i++) {
                int closest = findClosestCentroid(data + i * dim);
                if (assignments[i] != closest) {
                    assignments[i] = closest;
                    converged = false;
                }
            }
            
            if (converged && iter > 0) break;
            
            // 重新计算聚类中心
            std::fill(cluster_counts.begin(), cluster_counts.end(), 0);
            std::vector<float> new_centroids(nlist * dim, 0.0f);
            
            for (size_t i = 0; i < n; i++) {
                int c = assignments[i];
                cluster_counts[c]++;
                
                const float* vec = data + i * dim;
                float* centroid = new_centroids.data() + c * dim;
                
                for (size_t j = 0; j < dim; j++) {
                    centroid[j] += vec[j];
                }
            }
            
            // 计算平均值作为新的中心点
            for (size_t i = 0; i < nlist; i++) {
                if (cluster_counts[i] > 0) {
                    float inv_count = 1.0f / cluster_counts[i];
                    float* centroid = new_centroids.data() + i * dim;
                    for (size_t j = 0; j < dim; j++) {
                        centroid[j] *= inv_count;
                    }
                    memcpy(centroids + i * dim, centroid, dim * sizeof(float));
                }
            }
        }
    }

    // 初始聚类中心选择 (k-means++)
    void selectInitialCentroids(const float* data, size_t n, std::vector<int>& centroid_ids) {
        centroid_ids.clear();
        centroid_ids.reserve(nlist);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, n - 1);
        
        // 随机选择第一个中心
        centroid_ids.push_back(distrib(gen));
        
        std::vector<float> min_distances(n, std::numeric_limits<float>::max());
        
        // 依据距离概率选择剩余中心
        for (size_t c = 1; c < nlist; c++) {
            int last_id = centroid_ids.back();
            const float* last_center = data + (size_t)last_id * dim;
            
            float sum_distances = 0.0f;
            for (size_t i = 0; i < n; i++) {
                float dist = computeDistanceOptimized(data + i * dim, last_center);
                min_distances[i] = std::min(min_distances[i], dist);
                sum_distances += min_distances[i];
            }
            
            std::uniform_real_distribution<> u(0, sum_distances);
            float threshold = u(gen);
            
            float cumsum = 0.0f;
            size_t next_id = 0;
            for (size_t i = 0; i < n; i++) {
                cumsum += min_distances[i];
                if (cumsum >= threshold) {
                    next_id = i;
                    break;
                }
            }
            
            centroid_ids.push_back(static_cast<int>(next_id));
        }
    }

    // MPI广播支持函数
    void broadcast_from_rank0() {
        // 广播基本参数
        MPI_Bcast(&dim, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nlist, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&max_elements_per_cluster, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&ef_construction, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&M, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&n_vectors, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&trained, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

        // 广播聚类中心
        if (rank != 0) {
            if (centroids) delete[] centroids;
            centroids = new float[nlist * dim];
        }
        MPI_Bcast(centroids, nlist * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);

        // 广播倒排表结构
        if (rank != 0) {
            invlists.clear();
            invlists.resize(nlist);
            cluster_sizes.resize(nlist, 0);
        }

        for (size_t i = 0; i < nlist; i++) {
            // 广播ID列表大小
            size_t ids_size = 0;
            if (rank == 0) {
                ids_size = invlists[i].size();
            }
            MPI_Bcast(&ids_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
            
            // 调整接收进程的容器大小
            if (rank != 0) {
                invlists[i].resize(ids_size);
                cluster_sizes[i] = ids_size;
            }
            
            // 广播ID数据
            if (ids_size > 0) {
                MPI_Bcast(invlists[i].data(), ids_size, MPI_INT, 0, MPI_COMM_WORLD);
            }
        }

        // 初始化其他进程的HNSW空间和mutex
        if (rank != 0) {
            hnsw_indices.resize(nlist, nullptr);
            hnsw_spaces.resize(nlist, nullptr);
            cluster_mutexes.resize(nlist, nullptr);
            
            for (size_t i = 0; i < nlist; i++) {
                hnsw_spaces[i] = new hnswlib::L2Space(dim);
                cluster_mutexes[i] = new std::mutex();
            }
        }
        
        // 同步所有进程，确保广播完成
        MPI_Barrier(MPI_COMM_WORLD);
    }

public:
    // 构造函数
    IVFHNSW_MPI(size_t d, size_t nlist = 256, size_t max_elements = 10000, 
                size_t ef_construction = 200, size_t M = 16) 
        : dim(d), nlist(nlist), max_elements_per_cluster(max_elements),
          ef_construction(ef_construction), M(M), trained(false) {
        
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // 初始化存储结构
        invlists.resize(nlist);
        hnsw_indices.resize(nlist, nullptr);
        hnsw_spaces.resize(nlist, nullptr);
        cluster_sizes.resize(nlist, 0);
        cluster_mutexes.resize(nlist, nullptr);
        
        // 为每个聚类创建HNSW空间和mutex
        for (size_t i = 0; i < nlist; i++) {
            hnsw_spaces[i] = new hnswlib::L2Space(d);
            cluster_mutexes[i] = new std::mutex();
        }
    }

    // 析构函数
    ~IVFHNSW_MPI() {
        if (centroids) delete[] centroids;
        
        // 清理HNSW索引
        for (size_t i = 0; i < nlist; i++) {
            if (hnsw_indices[i]) delete hnsw_indices[i];
            if (hnsw_spaces[i]) delete hnsw_spaces[i];
            if (cluster_mutexes[i]) delete cluster_mutexes[i];
        }
    }

    // 训练索引
    void train(const float* data, size_t n) {
        if (rank == 0) {
            if (trained) {
                std::cerr << "Warning: Index already trained\n";
                return;
            }
            
            raw_data = data;
            n_vectors = n;
            
            std::cout << "Training IVF centroids...\n";
            trainCentroids(data, n);
            
            trained = true;
            std::cout << "IVF training completed\n";
        }

        // 广播训练好的模型到所有进程
        broadcast_from_rank0();
    }

    // 添加数据点
    void add(const float* data, size_t n, const int* ids = nullptr) {
        if (rank == 0) {
            if (!trained) {
                std::cerr << "Error: Index must be trained before adding data\n";
                return;
            }

            // 设置原始数据指针
            if (!raw_data) {
                raw_data = data;
                n_vectors = n;
            }

            std::cout << "Adding " << n << " vectors to IVF+HNSW index...\n";
            
            // 分配向量到聚类并添加到HNSW索引
            for (size_t i = 0; i < n; i++) {
                const float* vec = data + i * dim;
                int cluster_id = findClosestCentroid(vec);
                int point_id = ids ? ids[i] : static_cast<int>(i);
                
                // 添加到倒排列表
                invlists[cluster_id].push_back(point_id);
                cluster_sizes[cluster_id]++;
                
                // 如果HNSW索引尚未创建，创建它
                if (!hnsw_indices[cluster_id]) {
                    hnsw_indices[cluster_id] = new hnswlib::HierarchicalNSW<float>(
                        hnsw_spaces[cluster_id], max_elements_per_cluster, M, ef_construction
                    );
                }
                
                // 添加到对应的HNSW索引
                try {
                    hnsw_indices[cluster_id]->addPoint(vec, point_id);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to add point to HNSW cluster " 
                             << cluster_id << ": " << e.what() << std::endl;
                }
            }
            
            std::cout << "Successfully added vectors to index\n";
        }

        // 广播更新后的数据到所有进程
        broadcast_from_rank0();
    }

    // 设置原始数据指针（用于MPI搜索中的重排序）
    void setRawData(const float* data, size_t n) {
        raw_data = data;
        // 只在rank 0更新n_vectors，然后广播
        if (rank == 0) {
            n_vectors = n;
        }
        MPI_Bcast(&n_vectors, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    }

    // MPI并行搜索函数 - 修复的版本
    std::priority_queue<std::pair<float, int>> search_mpi(
            const float* query, 
            size_t k, 
            size_t nprobe = 10,
            size_t ef = 50,
            bool rerank = true,
            size_t rerank_factor = 3) const {
        
        if (!trained) {
            if (rank == 0) {
                std::cerr << "Error: Index must be trained before searching\n";
            }
            return std::priority_queue<std::pair<float, int>>();
        }
        
        nprobe = std::min(nprobe, nlist);
        
        // 1. 所有进程计算聚类中心距离（保证一致性）
        std::vector<std::pair<float, int>> cluster_distances;
        cluster_distances.reserve(nlist);
        
        for (size_t i = 0; i < nlist; i++) {
            float dist = computeDistanceOptimized(query, centroids + i * dim);
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
        
        // 3. 每个进程在分配给它的簇中使用HNSW搜索
        std::priority_queue<std::pair<float, int>> local_results;
        size_t search_k = rerank ? k * rerank_factor : k;
        
        for (int cluster_id : my_clusters) {
            // 跳过空聚类
            if (cluster_sizes[cluster_id] == 0) {
                continue;
            }
            
            // 对于没有HNSW索引的聚类，使用暴力搜索
            if (!hnsw_indices[cluster_id]) {
                if (raw_data) {
                    // 暴力搜索该聚类
                    for (int point_id : invlists[cluster_id]) {
                        float dist = computeDistanceOptimized(query, raw_data + (size_t)point_id * dim);
                        
                        if (local_results.size() < search_k) {
                            local_results.push({dist, point_id});
                        } else if (dist < local_results.top().first) {
                            local_results.pop();
                            local_results.push({dist, point_id});
                        }
                    }
                }
                continue;
            }
            
            // 设置HNSW搜索参数
            hnsw_indices[cluster_id]->setEf(ef);
            
            try {
                // 在当前聚类的HNSW索引中搜索
                auto cluster_results = hnsw_indices[cluster_id]->searchKnn(query, search_k);
                
                // 合并结果到局部结果中
                while (!cluster_results.empty()) {
                    auto result = cluster_results.top();
                    cluster_results.pop();
                    
                    if (local_results.size() < search_k) {
                        local_results.push(result);
                    } else if (result.first < local_results.top().first) {
                        local_results.pop();
                        local_results.push(result);
                    }
                }
            } catch (const std::exception& e) {
                if (rank == 0) {
                    std::cerr << "Warning: HNSW search failed for cluster " 
                             << cluster_id << ": " << e.what() << std::endl;
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
        
        // 5. 结果合并和排序
        std::priority_queue<std::pair<float, int>> final_results;
        
        if (rerank && raw_data) {
            // 收集所有候选并按距离排序
            std::vector<std::pair<float, int>> candidates;
            for (int i = 0; i < total_results; i++) {
                candidates.push_back({all_dists[i], all_ids[i]});
            }
            
            // 按距离排序，选择前候选
            std::sort(candidates.begin(), candidates.end());
            size_t max_candidates = std::min(candidates.size(), search_k);
            
            // 使用原始向量重新计算精确距离
            for (size_t i = 0; i < max_candidates; i++) {
                int id = candidates[i].second;
                float exact_dist = computeDistanceOptimized(query, raw_data + (size_t)id * dim);
                
                if (final_results.size() < k) {
                    final_results.push({exact_dist, id});
                } else if (exact_dist < final_results.top().first) {
                    final_results.pop();
                    final_results.push({exact_dist, id});
                }
            }
        } else {
            // 直接选择top-k结果
            std::vector<std::pair<float, int>> all_candidates;
            for (int i = 0; i < total_results; i++) {
                all_candidates.push_back({all_dists[i], all_ids[i]});
            }
            
            std::sort(all_candidates.begin(), all_candidates.end());
            size_t take_count = std::min(all_candidates.size(), k);
            
            for (size_t i = 0; i < take_count; i++) {
                final_results.push(all_candidates[i]);
            }
        }
        
        return final_results;
    }

    // 保存索引（仅主进程）
    bool save(const std::string& filename) const {
        if (rank != 0) return true;
        
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open()) {
            std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
            return false;
        }

        // 保存基本参数
        out.write((char*)&dim, sizeof(dim));
        out.write((char*)&nlist, sizeof(nlist));
        out.write((char*)&max_elements_per_cluster, sizeof(max_elements_per_cluster));
        out.write((char*)&ef_construction, sizeof(ef_construction));
        out.write((char*)&M, sizeof(M));
        out.write((char*)&n_vectors, sizeof(n_vectors));
        
        // 保存聚类中心
        out.write((char*)centroids, nlist * dim * sizeof(float));
        
        // 保存倒排列表
        for (size_t i = 0; i < nlist; i++) {
            size_t list_size = invlists[i].size();
            out.write((char*)&list_size, sizeof(list_size));
            if (list_size > 0) {
                out.write((char*)invlists[i].data(), list_size * sizeof(int));
            }
        }
        
        out.close();
        
        // 保存HNSW索引文件
        for (size_t i = 0; i < nlist; i++) {
            if (hnsw_indices[i] && cluster_sizes[i] > 0) {
                std::string hnsw_filename = filename + "_hnsw_" + std::to_string(i);
                try {
                    hnsw_indices[i]->saveIndex(hnsw_filename);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to save HNSW index for cluster " 
                             << i << ": " << e.what() << std::endl;
                }
            }
        }
        
        std::cout << "Index saved to " << filename << std::endl;
        return true;
    }

    // 加载索引
    bool load(const std::string& filename) {
        if (rank == 0) {
            std::ifstream in(filename, std::ios::binary);
            if (!in.is_open()) {
                std::cerr << "Error: Cannot open file for reading: " << filename << std::endl;
                return false;
            }

            // 读取基本参数
            in.read((char*)&dim, sizeof(dim));
            in.read((char*)&nlist, sizeof(nlist));
            in.read((char*)&max_elements_per_cluster, sizeof(max_elements_per_cluster));
            in.read((char*)&ef_construction, sizeof(ef_construction));
            in.read((char*)&M, sizeof(M));
            in.read((char*)&n_vectors, sizeof(n_vectors));
            
            // 读取聚类中心
            if (centroids) delete[] centroids;
            centroids = new float[nlist * dim];
            in.read((char*)centroids, nlist * dim * sizeof(float));
            
            // 重新初始化数据结构
            invlists.resize(nlist);
            cluster_sizes.resize(nlist, 0);
            
            // 读取倒排列表
            for (size_t i = 0; i < nlist; i++) {
                size_t list_size;
                in.read((char*)&list_size, sizeof(list_size));
                
                invlists[i].resize(list_size);
                cluster_sizes[i] = list_size;
                
                if (list_size > 0) {
                    in.read((char*)invlists[i].data(), list_size * sizeof(int));
                }
            }
            
            in.close();
            
            // 加载HNSW索引文件
            std::cout << "Loading HNSW indices...\n";
            size_t loaded_count = 0;
            for (size_t i = 0; i < nlist; i++) {
                if (cluster_sizes[i] > 0) {
                    std::string hnsw_filename = filename + "_hnsw_" + std::to_string(i);
                    std::ifstream hnsw_file(hnsw_filename);
                    
                    if (hnsw_file.good()) {
                        hnsw_file.close();
                        try {
                            // 创建HNSW索引并加载
                            hnsw_indices[i] = new hnswlib::HierarchicalNSW<float>(
                                hnsw_spaces[i], hnsw_filename
                            );
                            loaded_count++;
                        } catch (const std::exception& e) {
                            std::cerr << "Warning: Failed to load HNSW index for cluster " 
                                     << i << ": " << e.what() << std::endl;
                            hnsw_indices[i] = nullptr;
                        }
                    } else {
                        hnsw_indices[i] = nullptr;
                    }
                }
            }
            
            trained = true;
            std::cout << "Index loaded from " << filename << std::endl;
            std::cout << "Loaded " << loaded_count << " HNSW indices" << std::endl;
        }
        
        // 广播加载的数据到所有进程
        broadcast_from_rank0();
        
        return true;
    }

    // 获取统计信息
    void printStats() const {
        if (rank == 0) {
            std::cout << "=== IVF+HNSW MPI Index Statistics ===" << std::endl;
            std::cout << "Dimension: " << dim << std::endl;
            std::cout << "Number of clusters: " << nlist << std::endl;
            std::cout << "Total vectors: " << n_vectors << std::endl;
            std::cout << "MPI processes: " << size << std::endl;
            
            size_t non_empty_clusters = 0;
            size_t min_cluster_size = std::numeric_limits<size_t>::max();
            size_t max_cluster_size = 0;
            size_t hnsw_indices_count = 0;
            
            for (size_t i = 0; i < nlist; i++) {
                if (cluster_sizes[i] > 0) {
                    non_empty_clusters++;
                    min_cluster_size = std::min(min_cluster_size, cluster_sizes[i]);
                    max_cluster_size = std::max(max_cluster_size, cluster_sizes[i]);
                }
                if (hnsw_indices[i] != nullptr) {
                    hnsw_indices_count++;
                }
            }
            
            if (non_empty_clusters > 0) {
                std::cout << "Non-empty clusters: " << non_empty_clusters << std::endl;
                std::cout << "Min cluster size: " << min_cluster_size << std::endl;
                std::cout << "Max cluster size: " << max_cluster_size << std::endl;
                std::cout << "Avg cluster size: " << (n_vectors / non_empty_clusters) << std::endl;
            }
            std::cout << "HNSW indices ready: " << hnsw_indices_count << "/" << non_empty_clusters << std::endl;
            std::cout << "========================================" << std::endl;
        }
    }
};

// 便捷的MPI搜索接口函数 - 修复版本
std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> 
ivfhnsw_search_mpi(float* base, float* query, size_t base_number, size_t vecdim, size_t k, 
                   size_t nprobe = 20, size_t ef = 50, bool rerank = true, int num_threads = 1) {
    
    static IVFHNSW_MPI* index = nullptr;
    static bool initialized = false;
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // 首次调用时初始化索引
    if (!initialized) {
        bool index_exists = false;
        
        if (rank == 0) {
            // 创建files目录 - 修复路径问题
            #ifdef _WIN32
                (void)system("if not exist files mkdir files 2>nul");
            #else
                (void)system("mkdir -p files 2>/dev/null");
            #endif
            
            std::string path_index = "files/hnsw1/ivfhnsw_mpi.index";
            std::ifstream f(path_index);
            index_exists = f.good();
            f.close();
        }
        
        // 广播索引文件是否存在的信息
        MPI_Bcast(&index_exists, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        
        // 选择合适的参数
        size_t nlist = std::min(size_t(std::sqrt(base_number) * 2), base_number / 50);
        if (nlist < 10) nlist = 10;
        if (nlist > 512) nlist = 512;
        
        index = new IVFHNSW_MPI(vecdim, nlist, 20000, 200, 16);
        
        if (index_exists) {
            if (rank == 0) {
                std::cout << "Loading existing IVF+HNSW MPI index...\n";
            }
            index->load("files/hnsw1/ivfhnsw_mpi.index");
        } else {
            if (rank == 0) {
                std::cout << "Building new IVF+HNSW MPI index...\n";
            }
            index->train(base, base_number);
            index->add(base, base_number);
            index->save("files/hnsw1/ivfhnsw_mpi.index");
            if (rank == 0) {
                index->printStats();
            }
        }
        
        // 确保所有进程都设置了原始数据指针
        index->setRawData(base, base_number);
        initialized = true;
    }
    
    // 执行MPI并行搜索
    std::priority_queue<std::pair<float, int>> mpi_results = 
        index->search_mpi(query, k, nprobe, ef, rerank, 3);
    
    // 转换为兼容格式
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> output;
    while (!mpi_results.empty()) {
        output.push({mpi_results.top().first, mpi_results.top().second});
        mpi_results.pop();
    }
    
    return output;
}