#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <memory>
#include <cassert>
#include <iostream>
#include <fstream>
#include <omp.h>
#include <unordered_map>
#include <queue>
#include <limits>
#include <cstring>
#include <pthread.h>
#include <thread>

// IVF实现 - 倒排文件索引
class IVF {
public:
    // 构造函数
    IVF(size_t d, size_t n_clusters = 1024) 
        : dim(d), n_clusters(n_clusters), trained(false), 
          PARALLEL_THRESHOLD_CLUSTERS(256), PARALLEL_THRESHOLD_NPROBE(8) {
    }

    // 析构函数
    ~IVF() {
        if (centroids) delete[] centroids;
        if (rearranged_data) delete[] rearranged_data;
    }

    // 训练索引 - 使用k-means创建聚类中心
    void train(const float* data, size_t n) {
        if (trained) {
            std::cerr << "Warning: IVF index already trained, skipping training\n";
            return;
        }

        // 分配空间给聚类中心
        centroids = new float[n_clusters * dim];
        // 分配簇内点的存储空间
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
            #pragma omp parallel for
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
        
        trained = true;
    }

    // 添加向量到索引
    void addPoint(const float* vec, int id) {
        if (!trained) {
            std::cerr << "Error: Index must be trained before adding points\n";
            return;
        }
        
        // 找到最近的聚类中心
        int cluster_id = findClosestCentroid(vec);
        
        // 将该点ID添加到对应的倒排列表
        invlists[cluster_id].push_back(id);
    }
    
    // 批量添加向量时进行内存重排
    void addPoints(const float* data, size_t n) {
        if (!trained) {
            std::cerr << "Error: Index must be trained before adding points\n";
            return;
        }

        // 第一步：计算每个点所属的簇
        std::vector<int> point_clusters(n);
    
        for (size_t i = 0; i < n; i++) {
            point_clusters[i] = findClosestCentroid(data + i * dim);
        }
        
        // 更新倒排列表
        for (size_t i = 0; i < n; i++) {
            invlists[point_clusters[i]].push_back(i);
        }

        // 第二步：计算每个簇的大小和起始位置
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

        // 第三步：分配重排内存并复制数据
        rearranged_data = new float[n * dim];
        id_map.resize(n);  // 原始ID到重排后ID的映射
        rev_id_map.resize(n); // 重排后ID到原始ID的映射
        
        // 用于跟踪每个簇已填充的向量数量
        std::vector<size_t> cluster_filled(n_clusters, 0);
        
        for (size_t i = 0; i < n; i++) {
            int cluster = point_clusters[i];
            // 计算该向量在重排后的位置
            size_t new_pos = cluster_offsets[cluster] + cluster_filled[cluster];
            
            // 复制向量数据到新位置
            memcpy(rearranged_data + new_pos * dim, data + i * dim, dim * sizeof(float));
            
            // 更新映射
            id_map[i] = new_pos;
            rev_id_map[new_pos] = i;
            
            // 更新已填充数量
            cluster_filled[cluster]++;
        }

        // 存储簇的偏移量和大小，用于快速访问
        cluster_data_offsets = cluster_offsets;
        cluster_data_sizes = cluster_sizes;
        
        data_count = n;
        use_rearranged_data = true;
        
        std::cout << "Memory rearrangement completed. Data points organized by clusters.\n";
    }

    // 优化的单线程搜索函数
    std::priority_queue<std::pair<float, int>> search(
            const float* query, 
            const float* base, 
            size_t k, 
            size_t nprobe = 10) {
        
        if (!trained) {
            std::cerr << "Error: Index must be trained before searching\n";
            return std::priority_queue<std::pair<float, int>>();
        }
        
        // 限制nprobe不超过簇的数量
        nprobe = std::min(nprobe, n_clusters);
        
        // 找到最近的nprobe个簇
        std::vector<std::pair<float, int>> cluster_distances;
        cluster_distances.reserve(n_clusters);  // 预分配空间避免重新分配

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
        
        // 在选定的簇中查找k个最近邻
        std::priority_queue<std::pair<float, int>> results;
        
        for (size_t i = 0; i < nprobe; i++) {
            int cluster_id = cluster_distances[i].second;
            
            if (use_rearranged_data) {
                // 使用重排后的数据 - 直接访问连续存储的同一簇的数据
                size_t start = cluster_data_offsets[cluster_id];
                size_t size = cluster_data_sizes[cluster_id];
                
                for (size_t j = 0; j < size; j++) {
                    size_t rearranged_id = start + j;
                    float dist = computeDistanceOptimized(query, rearranged_data + rearranged_id * dim);
                    int original_id = rev_id_map[rearranged_id]; // 转换为原始ID
                    
                    if (results.size() < k) {
                        results.emplace(dist, original_id);
                    } else if (dist < results.top().first) {
                        results.pop();
                        results.emplace(dist, original_id);
                    }
                }
            } else {
                // 使用原始方式 - 通过倒排列表访问
                const std::vector<int>& ids = invlists[cluster_id];
                
                for (int id : ids) {
                    float dist = computeDistanceOptimized(query, base + (size_t)id * dim);
                    
                    if (results.size() < k) {
                        results.emplace(dist, id);
                    } else if (dist < results.top().first) {
                        results.pop();
                        results.emplace(dist, id);
                    }
                }
            }
        }
        
        return results;
    }

    // 优化的多线程搜索函数
    std::priority_queue<std::pair<float, int>> search_parallel(
            const float* query, 
            const float* base, 
            size_t k, 
            size_t nprobe = 10) {
        
        if (!trained) {
            std::cerr << "Error: Index must be trained before searching\n";
            return std::priority_queue<std::pair<float, int>>();
        }
        
        // 限制nprobe不超过簇的数量
        nprobe = std::min(nprobe, n_clusters);
        
        // 确定是否使用并行计算聚类中心距离
        bool parallel_centroids = n_clusters > PARALLEL_THRESHOLD_CLUSTERS;
        std::vector<std::pair<float, int>> cluster_distances;
        cluster_distances.reserve(n_clusters); // 预分配空间

        if (parallel_centroids) {
            // 并行计算聚类中心距离
            {
                // 局部存储，减少竞争
                std::vector<std::pair<float, int>> local_distances;
                local_distances.reserve(n_clusters / omp_get_num_threads() + 1);

                for (size_t i = 0; i < n_clusters; i++) {
                    float dist = computeDistanceOptimized(query, centroids + i * dim);
                    local_distances.emplace_back(dist, static_cast<int>(i));
                }

                // 批量合并结果
                {
                    cluster_distances.insert(
                        cluster_distances.end(), 
                        local_distances.begin(), 
                        local_distances.end()
                    );
                }
            }
        } else {
            // 单线程计算聚类中心距离
            for (size_t i = 0; i < n_clusters; i++) {
                float dist = computeDistanceOptimized(query, centroids + i * dim);
                cluster_distances.emplace_back(dist, static_cast<int>(i));
            }
        }
        
        // 部分排序，获取前nprobe个最近的簇
        std::partial_sort(
            cluster_distances.begin(), 
            cluster_distances.begin() + nprobe,
            cluster_distances.end()
        );
        
        // 确定是否使用并行搜索簇
        bool parallel_search = nprobe >= PARALLEL_THRESHOLD_NPROBE;
        std::vector<std::pair<float, int>> all_candidates;
        
        if (parallel_search) {
            // 估计结果数量，避免频繁重分配
            size_t estimated_results = 0;
            for (size_t i = 0; i < nprobe; i++) {
                int cluster_id = cluster_distances[i].second;
                estimated_results += use_rearranged_data ? 
                    cluster_data_sizes[cluster_id] : invlists[cluster_id].size();
            }
            estimated_results = std::min(estimated_results, k * 2);
            all_candidates.reserve(estimated_results);
            
            // 动态确定线程数，避免创建太多线程
            const size_t max_threads = std::min(
                static_cast<size_t>(omp_get_max_threads()),
                nprobe
            );
            
            {
                // 线程局部优先队列
                std::priority_queue<std::pair<float, int>> local_queue;
                
            
                for (size_t i = 0; i < nprobe; i++) {
                    int cluster_id = cluster_distances[i].second;
                    
                    if (use_rearranged_data) {
                        size_t start = cluster_data_offsets[cluster_id];
                        size_t size = cluster_data_sizes[cluster_id];
                        
                        for (size_t j = 0; j < size; j++) {
                            size_t rearranged_id = start + j;
                            float dist = computeDistanceOptimized(query, rearranged_data + rearranged_id * dim);
                            int original_id = rev_id_map[rearranged_id];
                            
                            if (local_queue.size() < k) {
                                local_queue.emplace(dist, original_id);
                            } else if (dist < local_queue.top().first) {
                                local_queue.pop();
                                local_queue.emplace(dist, original_id);
                            }
                        }
                    } else {
                        const std::vector<int>& ids = invlists[cluster_id];
                        
                        for (int id : ids) {
                            float dist = computeDistanceOptimized(query, base + (size_t)id * dim);
                            
                            if (local_queue.size() < k) {
                                local_queue.emplace(dist, id);
                            } else if (dist < local_queue.top().first) {
                                local_queue.pop();
                                local_queue.emplace(dist, id);
                            }
                        }
                    }
                }
                
                // 将局部结果转换为向量
                std::vector<std::pair<float, int>> thread_results;
                thread_results.reserve(local_queue.size());
                
                while (!local_queue.empty()) {
                    thread_results.push_back(local_queue.top());
                    local_queue.pop();
                }
                
                // 批量合并结果
        
                {
                    all_candidates.insert(
                        all_candidates.end(),
                        thread_results.begin(),
                        thread_results.end()
                    );
                }
            }
        } else {
            // 使用单线程搜索簇
            std::priority_queue<std::pair<float, int>> results;
            
            for (size_t i = 0; i < nprobe; i++) {
                int cluster_id = cluster_distances[i].second;
                
                if (use_rearranged_data) {
                    size_t start = cluster_data_offsets[cluster_id];
                    size_t size = cluster_data_sizes[cluster_id];
                    
                    for (size_t j = 0; j < size; j++) {
                        size_t rearranged_id = start + j;
                        float dist = computeDistanceOptimized(query, rearranged_data + rearranged_id * dim);
                        int original_id = rev_id_map[rearranged_id];
                        
                        if (results.size() < k) {
                            results.emplace(dist, original_id);
                        } else if (dist < results.top().first) {
                            results.pop();
                            results.emplace(dist, original_id);
                        }
                    }
                } else {
                    const std::vector<int>& ids = invlists[cluster_id];
                    
                    for (int id : ids) {
                        float dist = computeDistanceOptimized(query, base + (size_t)id * dim);
                        
                        if (results.size() < k) {
                            results.emplace(dist, id);
                        } else if (dist < results.top().first) {
                            results.pop();
                            results.emplace(dist, id);
                        }
                    }
                }
            }
            
            // 转换为向量格式以保持与并行版本一致的处理流程
            while (!results.empty()) {
                all_candidates.push_back(results.top());
                results.pop();
            }
        }
        
        // 合并结果 - 使用部分排序
        std::priority_queue<std::pair<float, int>> final_results;

        if (all_candidates.size() > k) {
            std::partial_sort(
                all_candidates.begin(),
                all_candidates.begin() + k,
                all_candidates.end(),
                [](const std::pair<float, int>& a, const std::pair<float, int>& b) { 
                    return a.first < b.first; 
                }
            );
            all_candidates.resize(k);
        }
        
        // 构建最终结果队列
        for (const auto& candidate : all_candidates) {
            final_results.push(candidate);
        }
        
        return final_results;
    }
    
    // 保存索引时也保存重排后的数据
    void saveIndex(const std::string& filename) {
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
            // 保存重排后的数据
            out.write((char*)rearranged_data, data_count * dim * sizeof(float));
            
            // 保存簇的偏移量和大小
            out.write((char*)cluster_data_offsets.data(), n_clusters * sizeof(size_t));
            out.write((char*)cluster_data_sizes.data(), n_clusters * sizeof(size_t));
            
            // 保存ID映射
            out.write((char*)rev_id_map.data(), data_count * sizeof(int));
        }
        
        out.close();
    }

    // 从文件加载索引
    void loadIndex(const std::string& filename) {
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
            // 读取重排后的数据
            if (rearranged_data) delete[] rearranged_data;
            rearranged_data = new float[data_count * dim];
            in.read((char*)rearranged_data, data_count * dim * sizeof(float));
            
            // 读取簇的偏移量和大小
            cluster_data_offsets.resize(n_clusters);
            cluster_data_sizes.resize(n_clusters);
            in.read((char*)cluster_data_offsets.data(), n_clusters * sizeof(size_t));
            in.read((char*)cluster_data_sizes.data(), n_clusters * sizeof(size_t));
            
            // 读取ID映射
            rev_id_map.resize(data_count);
            in.read((char*)rev_id_map.data(), data_count * sizeof(int));
            
            // 重建id_map (如果需要)
            id_map.resize(data_count);
            for (size_t i = 0; i < data_count; i++) {
                id_map[rev_id_map[i]] = i;
            }
        }
        
        trained = true;
        in.close();
    }

private:
    size_t dim;                          // 向量维度
    size_t n_clusters;                   // 聚类数量
    bool trained;                         // 是否已训练
    float* centroids = nullptr;           // 聚类中心
    std::vector<std::vector<int>> invlists; // 倒排列表

    // 添加新的成员变量
    float* rearranged_data = nullptr;     // 重排后的数据
    std::vector<int> id_map;              // 原始ID到重排后ID的映射
    std::vector<int> rev_id_map;          // 重排后ID到原始ID的映射
    std::vector<size_t> cluster_data_offsets; // 每个簇在重排数据中的起始位置
    std::vector<size_t> cluster_data_sizes;   // 每个簇的大小
    bool use_rearranged_data = false;     // 是否使用重排后的数据
    size_t data_count = 0;                // 数据总数
    
    // 并行阈值
    const size_t PARALLEL_THRESHOLD_CLUSTERS;  // 聚类数量超过此值启用并行计算距离
    const size_t PARALLEL_THRESHOLD_NPROBE;    // nprobe超过此值启用并行搜索

    // 优化的距离计算函数 - 使用循环展开
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
        
        // 处理剩余元素
        for (; d < dim; d++) {
            float diff = a[d] - b[d];
            sum += diff * diff;
        }
        
        return sum;
    }

    // 原始距离计算函数保留供兼容性
    float computeDistance(const float* a, const float* b) const {
        return computeDistanceOptimized(a, b);
    }

    // 找到最近的聚类中心
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
    
    // 选择初始聚类中心点 - k-means++方法
    void selectInitialCentroids(const float* data, size_t n, std::vector<int>& centroid_ids) {
        centroid_ids.clear();
        centroid_ids.reserve(n_clusters);
        
        // 随机选择第一个中心点
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, n - 1);
        int first_id = distrib(gen);
        centroid_ids.push_back(first_id);
        
        // 存储每个点到最近中心的距离
        std::vector<float> min_distances(n, std::numeric_limits<float>::max());
        
        // 选择剩余的中心点
        for (size_t c = 1; c < n_clusters; c++) {
            // 更新每个点到最近中心的距离
            int last_id = centroid_ids.back();
            const float* last_center = data + (size_t)last_id * dim;
            
            float sum_distances = 0.0f;
        
            for (size_t i = 0; i < n; i++) {
                float dist = computeDistanceOptimized(data + i * dim, last_center);
                min_distances[i] = std::min(min_distances[i], dist);
                sum_distances += min_distances[i];
            }
            
            // 按距离概率选择下一个中心
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

// IVF搜索函数，与现有代码兼容
std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> 
ivf_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k, size_t nprobe = 16) {
    static IVF* ivf = nullptr;
    static bool initialized = false;
    
    // 首次调用时初始化索引
    if (!initialized) {
        std::string path_index = "files/ivf.index";
        std::ifstream f(path_index);
        bool index_exists = f.good();
        f.close();
        
        ivf = new IVF(vecdim);
        
        if (index_exists) {
            std::cout << "Loading existing IVF index...\n";
            ivf->loadIndex(path_index);
        } else {
            std::cout << "Building new IVF index...\n";
            ivf->train(base, base_number);
            ivf->addPoints(base, base_number);
            ivf->saveIndex(path_index);
        }
        initialized = true;
    }
    
    // 执行搜索 - 使用自适应搜索策略
    //auto results = ivf->search(query, base, k, nprobe);
    auto results = ivf->search_parallel(query, base, k, nprobe);
    
    // 转换为与现有代码兼容的格式
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> output;
    while (!results.empty()) {
        output.push({results.top().first, results.top().second});
        results.pop();
    }
    
    return output;
}