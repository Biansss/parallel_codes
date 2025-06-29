#pragma once
// filepath: \home\s2313486\ann\IVF-HNSW-MPI-1.h
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
#include "hnswlib/hnswlib/hnswlib.h"
// 数据划分策略枚举
enum class PartitionStrategy {
    RANDOM,           // 随机划分
    ROUND_ROBIN,      // 轮询划分
    KMEANS_BASED,     // 基于K-means的启发式划分
    BALANCED_HASH     // 平衡哈希划分
};

// 分布式HNSW索引类
class DistributedHNSW_MPI {
private:
    int rank, size;                              // MPI进程号和总进程数
    size_t P;                                    // 分区数量
    PartitionStrategy strategy;                   // 划分策略
    
    // HNSW相关
    hnswlib::L2Space* space = nullptr;
    hnswlib::HierarchicalNSW<float>* local_hnsw = nullptr;
    
    // 数据管理
    std::vector<float> local_data;               // 本地数据副本
    std::vector<int> local_ids;                  // 本地ID映射
    std::vector<int> global_to_local_map;        // 全局ID到本地ID的映射
    size_t local_size = 0;                       // 本地数据大小
    size_t global_size = 0;                      // 全局数据大小
    size_t dim = 0;                              // 向量维度
    
    // 分区信息
    std::vector<size_t> partition_sizes;         // 每个分区的大小
    std::vector<size_t> partition_offsets;       // 每个分区的偏移量
    
    // HNSW参数
    size_t M = 16;
    size_t ef_construction = 200;
    size_t max_elements = 100000;
    
    // K-means相关（用于启发式划分）
    std::vector<float> partition_centroids;      // 分区中心点
    
public:
    // 构造函数
    DistributedHNSW_MPI(size_t dimension, size_t num_partitions = 0, 
                        PartitionStrategy part_strategy = PartitionStrategy::RANDOM,
                        size_t hnsw_M = 16, size_t hnsw_ef_construction = 200, 
                        size_t hnsw_max_elements = 100000) 
        : dim(dimension), strategy(part_strategy), M(hnsw_M), 
          ef_construction(hnsw_ef_construction), max_elements(hnsw_max_elements) {
        
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // 如果未指定分区数，使用进程数
        P = (num_partitions == 0) ? size : std::min(num_partitions, static_cast<size_t>(size));
        P = std::max(P, static_cast<size_t>(2));  // 至少2个分区
        P = std::min(P, static_cast<size_t>(8));  // 最多8个分区
        
        // 初始化L2空间
        space = new hnswlib::L2Space(dimension);
        
        if (rank == 0) {
            std::cout << "Initialized Distributed HNSW with " << P << " partitions using ";
            switch(strategy) {
                case PartitionStrategy::RANDOM: std::cout << "RANDOM"; break;
                case PartitionStrategy::ROUND_ROBIN: std::cout << "ROUND_ROBIN"; break;
                case PartitionStrategy::KMEANS_BASED: std::cout << "KMEANS_BASED"; break;
                case PartitionStrategy::BALANCED_HASH: std::cout << "BALANCED_HASH"; break;
            }
            std::cout << " partitioning strategy\n";
        }
    }
    
    // 析构函数
    ~DistributedHNSW_MPI() {
        if (local_hnsw) delete local_hnsw;
        if (space) delete space;
    }
    
    // 随机划分策略
    void partitionRandom(const float* data, size_t n, const int* ids = nullptr) {
        std::vector<std::vector<int>> partition_assignments(P);
        
        if (rank == 0) {
            // 主进程执行划分
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<> dis(0, P - 1);
            
            for (size_t i = 0; i < n; i++) {
                int partition_id = dis(gen);
                int global_id = ids ? ids[i] : static_cast<int>(i);
                partition_assignments[partition_id].push_back(global_id);
            }
        }
        
        // 广播分区分配结果
        broadcastPartitionAssignments(data, n, partition_assignments, ids);
    }
    
    // 轮询划分策略
    void partitionRoundRobin(const float* data, size_t n, const int* ids = nullptr) {
        std::vector<std::vector<int>> partition_assignments(P);
        
        if (rank == 0) {
            // 轮询分配
            for (size_t i = 0; i < n; i++) {
                int partition_id = i % P;
                int global_id = ids ? ids[i] : static_cast<int>(i);
                partition_assignments[partition_id].push_back(global_id);
            }
        }
        
        broadcastPartitionAssignments(data, n, partition_assignments, ids);
    }
    
    // 基于K-means的启发式划分策略
    void partitionKMeansBased(const float* data, size_t n, const int* ids = nullptr) {
        std::vector<std::vector<int>> partition_assignments(P);
        
        if (rank == 0) {
            // 执行K-means聚类
            performKMeansClustering(data, n);
            
            // 根据聚类结果分配数据点
            for (size_t i = 0; i < n; i++) {
                int closest_partition = findClosestPartition(data + i * dim);
                int global_id = ids ? ids[i] : static_cast<int>(i);
                partition_assignments[closest_partition].push_back(global_id);
            }
            
            std::cout << "K-means based partitioning completed\n";
        }
        
        broadcastPartitionAssignments(data, n, partition_assignments, ids);
    }
    
    // 平衡哈希划分策略
    void partitionBalancedHash(const float* data, size_t n, const int* ids = nullptr) {
        std::vector<std::vector<int>> partition_assignments(P);
        
        if (rank == 0) {
            // 使用哈希函数进行平衡划分
            for (size_t i = 0; i < n; i++) {
                // 计算向量的简单哈希值
                uint64_t hash = computeVectorHash(data + i * dim);
                int partition_id = hash % P;
                int global_id = ids ? ids[i] : static_cast<int>(i);
                partition_assignments[partition_id].push_back(global_id);
            }
        }
        
        broadcastPartitionAssignments(data, n, partition_assignments, ids);
    }
    
    // K-means聚类实现
    void performKMeansClustering(const float* data, size_t n) {
        partition_centroids.resize(P * dim);
        
        // 随机选择初始中心点
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, n - 1);
        
        for (size_t p = 0; p < P; p++) {
            int random_idx = dis(gen);
            memcpy(partition_centroids.data() + p * dim, 
                   data + random_idx * dim, dim * sizeof(float));
        }
        
        // K-means迭代
        std::vector<int> assignments(n);
        const int max_iterations = 20;
        
        for (int iter = 0; iter < max_iterations; iter++) {
            bool converged = true;
            
            // 分配步骤
            for (size_t i = 0; i < n; i++) {
                int new_assignment = findClosestPartition(data + i * dim);
                if (assignments[i] != new_assignment) {
                    assignments[i] = new_assignment;
                    converged = false;
                }
            }
            
            if (converged && iter > 0) break;
            
            // 更新中心点
            std::vector<std::vector<float>> new_centroids(P, std::vector<float>(dim, 0.0f));
            std::vector<int> counts(P, 0);
            
            for (size_t i = 0; i < n; i++) {
                int partition = assignments[i];
                counts[partition]++;
                
                for (size_t d = 0; d < dim; d++) {
                    new_centroids[partition][d] += data[i * dim + d];
                }
            }
            
            // 计算新的中心点
            for (size_t p = 0; p < P; p++) {
                if (counts[p] > 0) {
                    for (size_t d = 0; d < dim; d++) {
                        partition_centroids[p * dim + d] = new_centroids[p][d] / counts[p];
                    }
                }
            }
        }
    }
    
    // 找到最近的分区
    int findClosestPartition(const float* vec) const {
        int closest = 0;
        float min_dist = computeL2Distance(vec, partition_centroids.data());
        
        for (size_t p = 1; p < P; p++) {
            float dist = computeL2Distance(vec, partition_centroids.data() + p * dim);
            if (dist < min_dist) {
                min_dist = dist;
                closest = static_cast<int>(p);
            }
        }
        
        return closest;
    }
    
    // 计算L2距离
    float computeL2Distance(const float* a, const float* b) const {
        float sum = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
    
    // 计算向量哈希值
    uint64_t computeVectorHash(const float* vec) const {
        uint64_t hash = 0;
        for (size_t i = 0; i < dim; i++) {
            // 简单的哈希函数：将float转换为整数并累加
            uint32_t int_val = *reinterpret_cast<const uint32_t*>(&vec[i]);
            hash = hash * 31 + int_val;
        }
        return hash;
    }
    
    // 广播分区分配结果
    void broadcastPartitionAssignments(const float* data, size_t n, 
                                       const std::vector<std::vector<int>>& partition_assignments,
                                       const int* ids = nullptr) {
        // 广播每个分区的大小
        partition_sizes.resize(P);
        if (rank == 0) {
            for (size_t p = 0; p < P; p++) {
                partition_sizes[p] = partition_assignments[p].size();
            }
        }
        MPI_Bcast(partition_sizes.data(), P, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        
        // 计算偏移量
        partition_offsets.resize(P);
        partition_offsets[0] = 0;
        for (size_t p = 1; p < P; p++) {
            partition_offsets[p] = partition_offsets[p-1] + partition_sizes[p-1];
        }
        
        // 获取当前进程负责的分区
        int my_partition = rank % P;
        local_size = partition_sizes[my_partition];
        
        if (local_size > 0) {
            // 准备接收数据
            local_data.resize(local_size * dim);
            local_ids.resize(local_size);
            
            if (rank == 0) {
                // 主进程发送数据到各个进程
                for (size_t p = 0; p < P; p++) {
                    int target_rank = p % size;  // 分区到进程的映射
                    
                    if (target_rank == 0) {
                        // 主进程自己的数据
                        for (size_t i = 0; i < partition_assignments[p].size(); i++) {
                            int global_id = partition_assignments[p][i];
                            memcpy(local_data.data() + i * dim, 
                                   data + global_id * dim, dim * sizeof(float));
                            local_ids[i] = global_id;
                        }
                    } else {
                        // 发送给其他进程
                        std::vector<float> send_data(partition_assignments[p].size() * dim);
                        std::vector<int> send_ids(partition_assignments[p].size());
                        
                        for (size_t i = 0; i < partition_assignments[p].size(); i++) {
                            int global_id = partition_assignments[p][i];
                            memcpy(send_data.data() + i * dim, 
                                   data + global_id * dim, dim * sizeof(float));
                            send_ids[i] = global_id;
                        }
                        
                        if (!send_data.empty()) {
                            MPI_Send(send_data.data(), send_data.size(), MPI_FLOAT, target_rank, 0, MPI_COMM_WORLD);
                            MPI_Send(send_ids.data(), send_ids.size(), MPI_INT, target_rank, 1, MPI_COMM_WORLD);
                        }
                    }
                }
            } else {
                // 从进程接收数据
                if (local_size > 0) {
                    MPI_Recv(local_data.data(), local_data.size(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(local_ids.data(), local_ids.size(), MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
        
        // 同步所有进程
        MPI_Barrier(MPI_COMM_WORLD);
        
        // 广播K-means中心点（如果使用）
        if (strategy == PartitionStrategy::KMEANS_BASED) {
            if (rank != 0) {
                partition_centroids.resize(P * dim);
            }
            MPI_Bcast(partition_centroids.data(), P * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
    }
    
    // 构建索引
    void buildIndex(const float* data, size_t n, const int* ids = nullptr) {
        global_size = n;
        
        if (rank == 0) {
            std::cout << "Building distributed HNSW index with " << n << " vectors...\n";
        }
        
        // 根据策略进行数据划分
        switch (strategy) {
            case PartitionStrategy::RANDOM:
                partitionRandom(data, n, ids);
                break;
            case PartitionStrategy::ROUND_ROBIN:
                partitionRoundRobin(data, n, ids);
                break;
            case PartitionStrategy::KMEANS_BASED:
                partitionKMeansBased(data, n, ids);
                break;
            case PartitionStrategy::BALANCED_HASH:
                partitionBalancedHash(data, n, ids);
                break;
        }
        
        // 在本地数据上构建HNSW索引
        if (local_size > 0) {
            size_t actual_max_elements = std::max(local_size * 2, max_elements);
            local_hnsw = new hnswlib::HierarchicalNSW<float>(space, actual_max_elements, M, ef_construction);
            
            // 添加本地数据点
            for (size_t i = 0; i < local_size; i++) {
                try {
                    local_hnsw->addPoint(local_data.data() + i * dim, local_ids[i]);
                } catch (const std::exception& e) {
                    std::cerr << "Rank " << rank << ": Error adding point " << i 
                             << " to local HNSW: " << e.what() << std::endl;
                }
            }
            
            if (rank == 0) {
                std::cout << "Rank " << rank << ": Built local HNSW with " << local_size << " points\n";
            }
        }
        
        // 同步所有进程
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) {
            std::cout << "Distributed HNSW index construction completed\n";
            printIndexStats();
        }
    }
    
    // MPI并行搜索
    std::priority_queue<std::pair<float, int>> searchMPI(const float* query, size_t k, size_t ef = 50) {
        if (!local_hnsw || local_size == 0) {
            return std::priority_queue<std::pair<float, int>>();
        }
        
        // 每个进程在其本地HNSW索引中搜索
        local_hnsw->setEf(ef);
        
        std::priority_queue<std::pair<float, int>> local_results;
        
        try {
            // 在本地索引中搜索更多候选（为了保证质量）
            size_t local_k = std::min(local_size, std::max(k * 2, static_cast<size_t>(50)));
            auto hnsw_results = local_hnsw->searchKnn(query, local_k);
            
            // 转换结果格式
            while (!hnsw_results.empty()) {
                local_results.push(hnsw_results.top());
                hnsw_results.pop();
            }
        } catch (const std::exception& e) {
            std::cerr << "Rank " << rank << ": HNSW search error: " << e.what() << std::endl;
        }
        
        // 收集所有进程的结果
        std::vector<std::pair<float, int>> local_vec;
        while (!local_results.empty()) {
            local_vec.push_back(local_results.top());
            local_results.pop();
        }
        
        // 收集结果大小
        int local_count = static_cast<int>(local_vec.size());
        std::vector<int> all_counts(size);
        MPI_Allgather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        // 计算偏移量
        std::vector<int> offsets(size, 0);
        int total_count = 0;
        for (int i = 0; i < size; i++) {
            offsets[i] = total_count;
            total_count += all_counts[i];
        }
        
        // 准备发送数据
        std::vector<float> local_distances;
        std::vector<int> local_result_ids;
        for (const auto& result : local_vec) {
            local_distances.push_back(result.first);
            local_result_ids.push_back(result.second);
        }
        
        // 收集所有结果
        std::vector<float> all_distances(total_count);
        std::vector<int> all_ids(total_count);
        
        MPI_Allgatherv(local_distances.data(), local_count, MPI_FLOAT,
                       all_distances.data(), all_counts.data(), offsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
        MPI_Allgatherv(local_result_ids.data(), local_count, MPI_INT,
                       all_ids.data(), all_counts.data(), offsets.data(), MPI_INT, MPI_COMM_WORLD);
        
        // 合并结果并选择top-k
        std::vector<std::pair<float, int>> all_candidates;
        for (int i = 0; i < total_count; i++) {
            all_candidates.push_back({all_distances[i], all_ids[i]});
        }
        
        // 排序并选择最好的k个结果
        std::sort(all_candidates.begin(), all_candidates.end());
        
        std::priority_queue<std::pair<float, int>> final_results;
        size_t take_count = std::min(all_candidates.size(), k);
        
        for (size_t i = 0; i < take_count; i++) {
            final_results.push(all_candidates[i]);
        }
        
        return final_results;
    }
    
    // 保存索引
    bool saveIndex(const std::string& base_filename) {
        if (rank == 0) {
            // 保存全局索引信息
            std::string global_info_file = base_filename + ".global";
            std::ofstream out(global_info_file, std::ios::binary);
            if (!out.is_open()) {
                std::cerr << "Cannot open global info file for writing\n";
                return false;
            }
            
            out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
            out.write(reinterpret_cast<const char*>(&P), sizeof(P));
            out.write(reinterpret_cast<const char*>(&global_size), sizeof(global_size));
            out.write(reinterpret_cast<const char*>(&strategy), sizeof(strategy));
            
            // 保存分区大小信息
            out.write(reinterpret_cast<const char*>(partition_sizes.data()), P * sizeof(size_t));
            
            // 如果是K-means策略，保存中心点
            if (strategy == PartitionStrategy::KMEANS_BASED) {
                out.write(reinterpret_cast<const char*>(partition_centroids.data()), P * dim * sizeof(float));
            }
            
            out.close();
        }
        
        // 每个进程保存自己的本地HNSW索引
        if (local_hnsw && local_size > 0) {
            std::string local_file = base_filename + "_rank_" + std::to_string(rank) + ".hnsw";
            try {
                local_hnsw->saveIndex(local_file);
                
                // 保存本地ID映射
                std::string id_file = base_filename + "_rank_" + std::to_string(rank) + ".ids";
                std::ofstream id_out(id_file, std::ios::binary);
                id_out.write(reinterpret_cast<const char*>(&local_size), sizeof(local_size));
                id_out.write(reinterpret_cast<const char*>(local_ids.data()), local_size * sizeof(int));
                id_out.close();
            } catch (const std::exception& e) {
                std::cerr << "Rank " << rank << ": Error saving local index: " << e.what() << std::endl;
                return false;
            }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) {
            std::cout << "Distributed index saved to " << base_filename << std::endl;
        }
        
        return true;
    }
    
    // 加载索引
    bool loadIndex(const std::string& base_filename) {
        if (rank == 0) {
            // 加载全局索引信息
            std::string global_info_file = base_filename + ".global";
            std::ifstream in(global_info_file, std::ios::binary);
            if (!in.is_open()) {
                std::cerr << "Cannot open global info file for reading\n";
                return false;
            }
            
            in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            in.read(reinterpret_cast<char*>(&P), sizeof(P));
            in.read(reinterpret_cast<char*>(&global_size), sizeof(global_size));
            in.read(reinterpret_cast<char*>(&strategy), sizeof(strategy));
            
            partition_sizes.resize(P);
            in.read(reinterpret_cast<char*>(partition_sizes.data()), P * sizeof(size_t));
            
            if (strategy == PartitionStrategy::KMEANS_BASED) {
                partition_centroids.resize(P * dim);
                in.read(reinterpret_cast<char*>(partition_centroids.data()), P * dim * sizeof(float));
            }
            
            in.close();
        }
        
        // 广播全局信息
        MPI_Bcast(&dim, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&P, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&strategy, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            partition_sizes.resize(P);
        }
        MPI_Bcast(partition_sizes.data(), P, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        
        if (strategy == PartitionStrategy::KMEANS_BASED) {
            if (rank != 0) {
                partition_centroids.resize(P * dim);
            }
            MPI_Bcast(partition_centroids.data(), P * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
        
        // 确定本进程的分区
        int my_partition = rank % P;
        local_size = partition_sizes[my_partition];
        
        // 加载本地HNSW索引
        if (local_size > 0) {
            std::string local_file = base_filename + "_rank_" + std::to_string(rank) + ".hnsw";
            std::string id_file = base_filename + "_rank_" + std::to_string(rank) + ".ids";
            
            // 检查文件是否存在
            std::ifstream test_file(local_file);
            if (!test_file.good()) {
                std::cerr << "Rank " << rank << ": Local index file not found: " << local_file << std::endl;
                return false;
            }
            test_file.close();
            
            try {
                // 加载HNSW索引
                local_hnsw = new hnswlib::HierarchicalNSW<float>(space, local_file);
                
                // 加载ID映射
                std::ifstream id_in(id_file, std::ios::binary);
                if (id_in.is_open()) {
                    size_t loaded_size;
                    id_in.read(reinterpret_cast<char*>(&loaded_size), sizeof(loaded_size));
                    local_ids.resize(loaded_size);
                    id_in.read(reinterpret_cast<char*>(local_ids.data()), loaded_size * sizeof(int));
                    id_in.close();
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Rank " << rank << ": Error loading local index: " << e.what() << std::endl;
                return false;
            }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        if (rank == 0) {
            std::cout << "Distributed index loaded from " << base_filename << std::endl;
            printIndexStats();
        }
        
        return true;
    }
    
    // 打印索引统计信息
    void printIndexStats() const {
        if (rank == 0) {
            std::cout << "=== Distributed HNSW Index Statistics ===" << std::endl;
            std::cout << "Dimension: " << dim << std::endl;
            std::cout << "Number of partitions: " << P << std::endl;
            std::cout << "Total vectors: " << global_size << std::endl;
            std::cout << "MPI processes: " << size << std::endl;
            
            size_t min_partition_size = *std::min_element(partition_sizes.begin(), partition_sizes.end());
            size_t max_partition_size = *std::max_element(partition_sizes.begin(), partition_sizes.end());
            size_t avg_partition_size = global_size / P;
            
            std::cout << "Min partition size: " << min_partition_size << std::endl;
            std::cout << "Max partition size: " << max_partition_size << std::endl;
            std::cout << "Avg partition size: " << avg_partition_size << std::endl;
        }
    }
};

// 便捷的搜索接口函数
std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> 
hnsw_mpi_distributed_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k, 
                           size_t num_partitions = 0, PartitionStrategy strategy = PartitionStrategy::KMEANS_BASED,
                           size_t ef = 50) {
    
    static DistributedHNSW_MPI* index = nullptr;
    static bool initialized = false;
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (!initialized) {
        if (rank == 0) {
            std::cout << "Initializing Distributed HNSW MPI index...\n";
        }
        
        // 创建索引
        index = new DistributedHNSW_MPI(vecdim, num_partitions, strategy, 16, 200, base_number / 4);
        
        // 检查是否有保存的索引
        std::string index_file = "files/hnsw2/distributed_hnsw_mpi.index";
        bool index_exists = false;
        
        if (rank == 0) {
            std::ifstream test(index_file + ".global");
            index_exists = test.good();
            test.close();
        }
        
        MPI_Bcast(&index_exists, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        
        if (index_exists) {
            if (rank == 0) {
                std::cout << "Loading existing distributed HNSW index...\n";
            }
            if (!index->loadIndex(index_file)) {
                if (rank == 0) {
                    std::cerr << "Failed to load index, rebuilding...\n";
                }
                index_exists = false;
            }
        }
        
        if (!index_exists) {
            if (rank == 0) {
                std::cout << "Building new distributed HNSW index...\n";
                
                // 创建目录
                system("mkdir -p files 2>/dev/null");
            }
            
            index->buildIndex(base, base_number);
            index->saveIndex(index_file);
        }
        
        initialized = true;
    }
    
    // 执行搜索
    auto mpi_results = index->searchMPI(query, k, ef);
    
    // 转换为兼容格式
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> output;
    while (!mpi_results.empty()) {
        output.push({mpi_results.top().first, mpi_results.top().second});
        mpi_results.pop();
    }
    
    return output;
}