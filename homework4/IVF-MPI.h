#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <memory>
#include <cassert>
#include <fstream>
#include <unordered_map>
#include <queue>
#include <limits>
#include <cstring>
#include <mpi.h>

// MPI版本的IVF实现 - 倒排文件索引
class IVF_MPI {
private:
    int rank, size; // MPI进程号和总进程数

public:
    // 构造函数
    IVF_MPI(size_t d, size_t n_clusters = 1024) 
        : dim(d), n_clusters(n_clusters), trained(false) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    // 析构函数
    ~IVF_MPI() {
        if (centroids) delete[] centroids;
        if (rearranged_data) delete[] rearranged_data;
    }

    // 训练索引 - 使用k-means创建聚类中心
    void train(const float* data, size_t n) {
        if (trained) {
            if (rank == 0) {
                std::cerr << "Warning: IVF index already trained, skipping training\n";
            }
            return;
        }

        if (rank == 0) {
            // 主进程执行训练逻辑
            centroids = new float[n_clusters * dim];
            invlists.resize(n_clusters);
            
            // 随机选择初始聚类中心
            std::vector<int> centroid_ids;
            selectInitialCentroids(data, n, centroid_ids);
            
            // 将选中的向量复制为中心点
            for (size_t i = 0; i < n_clusters; i++) {
                memcpy(centroids + i * dim, data + (size_t)centroid_ids[i] * dim, dim * sizeof(float));
            }
            
            // k-means迭代
            std::vector<int> assignments(n, -1);
            std::vector<int> cluster_sizes(n_clusters, 0);
            
            const int max_iterations = 20;
            for (int iter = 0; iter < max_iterations; iter++) {
                bool converged = true;
                
                // 分配每个点到最近的聚类中心
                for (size_t i = 0; i < n; i++) {
                    int closest = findClosestCentroid(data + i * dim);
                    if (assignments[i] != closest) {
                        assignments[i] = closest;
                        converged = false;
                    }
                }
                
                if (converged && iter > 0) break;
                
                // 重置聚类中心和计数
                std::fill(cluster_sizes.begin(), cluster_sizes.end(), 0);
                std::vector<float> new_centroids(n_clusters * dim, 0);
                
                // 累加所有属于同一簇的向量
                for (size_t i = 0; i < n; i++) {
                    int c = assignments[i];
                    cluster_sizes[c]++;
                    
                    const float* vec = data + i * dim;
                    float* centroid = new_centroids.data() + c * dim;
                    
                    for (size_t j = 0; j < dim; j++) {
                        centroid[j] += vec[j];
                    }
                }
                
                // 计算新的中心点
                for (size_t i = 0; i < n_clusters; i++) {
                    if (cluster_sizes[i] > 0) {
                        float inv_size = 1.0f / cluster_sizes[i];
                        float* centroid = new_centroids.data() + i * dim;
                        for (size_t j = 0; j < dim; j++) {
                            centroid[j] *= inv_size;
                        }
                        // 更新中心点
                        memcpy(centroids + i * dim, centroid, dim * sizeof(float));
                    }
                }
                
                std::cout << "K-means iteration " << iter << " complete\n";
            }
        } else {
            // 从进程需要分配内存
            centroids = new float[n_clusters * dim];
        }
        
        // 广播聚类中心到所有进程
        MPI_Bcast(centroids, n_clusters * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        trained = true;
    }

    // 批量添加向量时进行内存重排
    void addPoints(const float* data, size_t n) {
        if (!trained) {
            if (rank == 0) {
                std::cerr << "Error: Index must be trained before adding points\n";
            }
            return;
        }

        if (rank == 0) {
            // 主进程计算每个点所属的簇
            std::vector<int> point_clusters(n);
        
            for (size_t i = 0; i < n; i++) {
                point_clusters[i] = findClosestCentroid(data + i * dim);
            }
            
            // 更新倒排列表
            invlists.resize(n_clusters);
            for (size_t i = 0; i < n; i++) {
                invlists[point_clusters[i]].push_back(i);
            }

            // 计算每个簇的大小和起始位置
            std::vector<size_t> cluster_sizes(n_clusters, 0);
            std::vector<size_t> cluster_offsets(n_clusters, 0);
            
            for (size_t i = 0; i < n; i++) {
                cluster_sizes[point_clusters[i]]++;
            }
            
            size_t offset = 0;
            for (size_t i = 0; i < n_clusters; i++) {
                cluster_offsets[i] = offset;
                offset += cluster_sizes[i];
            }

            // 分配重排内存并复制数据
            rearranged_data = new float[n * dim];
            id_map.resize(n);
            rev_id_map.resize(n);
            
            std::vector<size_t> cluster_filled(n_clusters, 0);
            
            for (size_t i = 0; i < n; i++) {
                int cluster = point_clusters[i];
                size_t new_pos = cluster_offsets[cluster] + cluster_filled[cluster];
                
                memcpy(rearranged_data + new_pos * dim, data + i * dim, dim * sizeof(float));
                
                id_map[i] = new_pos;
                rev_id_map[new_pos] = i;
                
                cluster_filled[cluster]++;
            }

            cluster_data_offsets = cluster_offsets;
            cluster_data_sizes = cluster_sizes;
            
            data_count = n;
            use_rearranged_data = true;
            
            std::cout << "Memory rearrangement completed. Data points organized by clusters.\n";
        }
        
        // 广播数据组织信息到所有进程
        MPI_Bcast(&data_count, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&use_rearranged_data, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            rearranged_data = new float[data_count * dim];
            cluster_data_offsets.resize(n_clusters);
            cluster_data_sizes.resize(n_clusters);
            rev_id_map.resize(data_count);
        }
        
        if (use_rearranged_data) {
            MPI_Bcast(rearranged_data, data_count * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(cluster_data_offsets.data(), n_clusters, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(cluster_data_sizes.data(), n_clusters, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(rev_id_map.data(), data_count, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }

    // MPI并行搜索函数
    std::priority_queue<std::pair<float, int>> search_mpi(
            const float* query, 
            const float* base, 
            size_t k, 
            size_t nprobe = 10) {
        
        if (!trained) {
            if (rank == 0) {
                std::cerr << "Error: Index must be trained before searching\n";
            }
            return std::priority_queue<std::pair<float, int>>();
        }
        
        nprobe = std::min(nprobe, n_clusters);
        
        // 所有进程计算聚类中心距离（保证一致性）
        std::vector<std::pair<float, int>> cluster_distances;
        cluster_distances.reserve(n_clusters);

        for (size_t i = 0; i < n_clusters; i++) {
            float dist = computeDistanceOptimized(query, centroids + i * dim);
            cluster_distances.push_back({dist, static_cast<int>(i)});
        }
        
        // 部分排序，获取前nprobe个最近的簇
        std::partial_sort(
            cluster_distances.begin(), 
            cluster_distances.begin() + nprobe,
            cluster_distances.end()
        );
        
        // MPI任务分配：将nprobe个簇分配给不同进程
        std::vector<int> my_clusters;
        for (size_t i = rank; i < nprobe; i += size) {
            my_clusters.push_back(cluster_distances[i].second);
        }
        
        // 每个进程在分配给它的簇中搜索
        std::priority_queue<std::pair<float, int>> local_results;
        size_t rerank_count = k * 10; // 粗排保留更多候选
        
        for (int cluster_id : my_clusters) {
            if (use_rearranged_data) {
                // 使用重排后的数据
                size_t start = cluster_data_offsets[cluster_id];
                size_t cluster_size = cluster_data_sizes[cluster_id];
                
                for (size_t j = 0; j < cluster_size; j++) {
                    size_t rearranged_id = start + j;
                    float dist = computeDistanceOptimized(query, rearranged_data + rearranged_id * dim);
                    int original_id = rev_id_map[rearranged_id];
                    
                    if (local_results.size() < rerank_count) {
                        local_results.emplace(dist, original_id);
                    } else if (dist < local_results.top().first) {
                        local_results.pop();
                        local_results.emplace(dist, original_id);
                    }
                }
            } else {
                // 主进程负责处理倒排列表（如果没有重排数据）
                if (rank == 0) {
                    const std::vector<int>& ids = invlists[cluster_id];
                    
                    for (int id : ids) {
                        float dist = computeDistanceOptimized(query, base + (size_t)id * dim);
                        
                        if (local_results.size() < rerank_count) {
                            local_results.emplace(dist, id);
                        } else if (dist < local_results.top().first) {
                            local_results.pop();
                            local_results.emplace(dist, id);
                        }
                    }
                }
            }
        }
        
        // 收集各进程的局部结果
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
        
        // 主进程合并结果并返回top-k
        std::priority_queue<std::pair<float, int>> final_results;
        for (int i = 0; i < total_results; i++) {
            if (final_results.size() < k) {
                final_results.emplace(all_dists[i], all_ids[i]);
            } else if (all_dists[i] < final_results.top().first) {
                final_results.pop();
                final_results.emplace(all_dists[i], all_ids[i]);
            }
        }
        
        return final_results;
    }

    // 保存和加载函数（仅主进程执行）
    void saveIndex(const std::string& filename) {
        if (rank != 0) return;
        
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open()) {
            std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
            return;
        }

        // 保存基本参数
        out.write((char*)&dim, sizeof(dim));
        out.write((char*)&n_clusters, sizeof(n_clusters));
        out.write((char*)&use_rearranged_data, sizeof(use_rearranged_data));
        out.write((char*)&data_count, sizeof(data_count));
        
        // 保存聚类中心
        out.write((char*)centroids, dim * n_clusters * sizeof(float));
        
        // 保存倒排列表
        for (size_t i = 0; i < n_clusters; i++) {
            size_t list_size = invlists[i].size();
            out.write((char*)&list_size, sizeof(list_size));
            if (list_size > 0) {
                out.write((char*)invlists[i].data(), list_size * sizeof(int));
            }
        }
        
        // 如果使用了数据重排，保存相关信息
        if (use_rearranged_data) {
            out.write((char*)rearranged_data, data_count * dim * sizeof(float));
            out.write((char*)cluster_data_offsets.data(), n_clusters * sizeof(size_t));
            out.write((char*)cluster_data_sizes.data(), n_clusters * sizeof(size_t));
            out.write((char*)rev_id_map.data(), data_count * sizeof(int));
        }
        
        out.close();
    }

    void loadIndex(const std::string& filename) {
        if (rank == 0) {
            std::ifstream in(filename, std::ios::binary);
            if (!in.is_open()) {
                std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
                return;
            }

            // 读取基本参数
            in.read((char*)&dim, sizeof(dim));
            in.read((char*)&n_clusters, sizeof(n_clusters));
            in.read((char*)&use_rearranged_data, sizeof(use_rearranged_data));
            in.read((char*)&data_count, sizeof(data_count));
            
            // 分配并读取聚类中心
            if (centroids) delete[] centroids;
            centroids = new float[dim * n_clusters];
            in.read((char*)centroids, dim * n_clusters * sizeof(float));
            
            // 读取倒排列表
            invlists.resize(n_clusters);
            for (size_t i = 0; i < n_clusters; i++) {
                size_t list_size;
                in.read((char*)&list_size, sizeof(list_size));
                
                invlists[i].resize(list_size);
                if (list_size > 0) {
                    in.read((char*)invlists[i].data(), list_size * sizeof(int));
                }
            }
            
            // 如果使用了数据重排，读取相关信息
            if (use_rearranged_data) {
                if (rearranged_data) delete[] rearranged_data;
                rearranged_data = new float[data_count * dim];
                in.read((char*)rearranged_data, data_count * dim * sizeof(float));
                
                cluster_data_offsets.resize(n_clusters);
                cluster_data_sizes.resize(n_clusters);
                in.read((char*)cluster_data_offsets.data(), n_clusters * sizeof(size_t));
                in.read((char*)cluster_data_sizes.data(), n_clusters * sizeof(size_t));
                
                rev_id_map.resize(data_count);
                in.read((char*)rev_id_map.data(), data_count * sizeof(int));
                
                id_map.resize(data_count);
                for (size_t i = 0; i < data_count; i++) {
                    id_map[rev_id_map[i]] = i;
                }
            }
            
            in.close();
        }
        
        // 广播加载的数据到所有进程
        MPI_Bcast(&dim, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&n_clusters, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(&use_rearranged_data, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        MPI_Bcast(&data_count, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        
        if (rank != 0) {
            centroids = new float[dim * n_clusters];
            if (use_rearranged_data) {
                rearranged_data = new float[data_count * dim];
                cluster_data_offsets.resize(n_clusters);
                cluster_data_sizes.resize(n_clusters);
                rev_id_map.resize(data_count);
            }
        }
        
        MPI_Bcast(centroids, dim * n_clusters, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        if (use_rearranged_data) {
            MPI_Bcast(rearranged_data, data_count * dim, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(cluster_data_offsets.data(), n_clusters, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(cluster_data_sizes.data(), n_clusters, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(rev_id_map.data(), data_count, MPI_INT, 0, MPI_COMM_WORLD);
        }
        
        trained = true;
    }

public:
    size_t dim;
    size_t n_clusters;
    bool trained;
    float* centroids = nullptr;
    std::vector<std::vector<int>> invlists;

    // 数据重排相关
    float* rearranged_data = nullptr;
    std::vector<int> id_map;
    std::vector<int> rev_id_map;
    std::vector<size_t> cluster_data_offsets;
    std::vector<size_t> cluster_data_sizes;
    bool use_rearranged_data = false;
    size_t data_count = 0;

    // 优化的距离计算函数
    float computeDistanceOptimized(const float* a, const float* b) const {
        float sum = 0.0f;
        size_t d = 0;
        
        // 4x展开循环
        for (; d + 3 < dim; d += 4) {
            float diff0 = a[d] - b[d];
            float diff1 = a[d+1] - b[d+1];
            float diff2 = a[d+2] - b[d+2];
            float diff3 = a[d+3] - b[d+3];
            
            sum += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
        }
        
        for (; d < dim; d++) {
            float diff = a[d] - b[d];
            sum += diff * diff;
        }
        
        return sum;
    }

    int findClosestCentroid(const float* vec) const {
        int closest = 0;
        float min_dist = computeDistanceOptimized(vec, centroids);
        
        for (size_t i = 1; i < n_clusters; i++) {
            float dist = computeDistanceOptimized(vec, centroids + i * dim);
            if (dist < min_dist) {
                min_dist = dist;
                closest = i;
            }
        }
        
        return closest;
    }
    
    void selectInitialCentroids(const float* data, size_t n, std::vector<int>& centroid_ids) {
        centroid_ids.clear();
        centroid_ids.reserve(n_clusters);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, n - 1);
        int first_id = distrib(gen);
        centroid_ids.push_back(first_id);
        
        std::vector<float> min_distances(n, std::numeric_limits<float>::max());
        
        for (size_t c = 1; c < n_clusters; c++) {
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
            
            centroid_ids.push_back(next_id);
        }
    }
};

// MPI版本的IVF搜索函数
std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> 
ivf_search_mpi(float* base, float* query, size_t base_number, size_t vecdim, size_t k, size_t nprobe = 16) {
    static IVF_MPI* ivf = nullptr;
    static bool initialized = false;
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // 首次调用时初始化索引
    if (!initialized) {
        std::string path_index = "files/ivf_mpi.index";
        bool index_exists = false;
        
        if (rank == 0) {
            std::ifstream f(path_index);
            index_exists = f.good();
            f.close();
        }
        
        // 广播索引文件是否存在的信息
        MPI_Bcast(&index_exists, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        
        ivf = new IVF_MPI(vecdim);
        
        if (index_exists) {
            if (rank == 0) {
                std::cout << "Loading existing IVF MPI index...\n";
            }
            ivf->loadIndex(path_index);
        } else {
            if (rank == 0) {
                std::cout << "Building new IVF MPI index...\n";
            }
            ivf->train(base, base_number);
            ivf->addPoints(base, base_number);
            ivf->saveIndex(path_index);
        }
        initialized = true;
    }
    
    // 执行MPI并行搜索
    std::priority_queue<std::pair<float, int>> results = ivf->search_mpi(query, base, k, nprobe);
    
    // 转换为与现有代码兼容的格式
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> output;
    while (!results.empty()) {
        output.push({results.top().first, results.top().second});
        results.pop();
    }
    
    return output;
}