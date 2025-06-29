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
#include <pthread.h>
#include <thread>
#include <mutex>
#include "IVF.h"
#include "hnswlib/hnswlib/hnswlib.h"

// IVF+HNSW混合算法实现
class IVFHNSW {
private:
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

public:
    // 构造函数
    IVFHNSW(size_t d, size_t nlist = 256, size_t max_elements = 10000, 
            size_t ef_construction = 200, size_t M = 16) 
        : dim(d), nlist(nlist), max_elements_per_cluster(max_elements),
          ef_construction(ef_construction), M(M), trained(false) {
        
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
    ~IVFHNSW() {
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

    // 添加数据点
    void add(const float* data, size_t n, const int* ids = nullptr) {
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
            
            // 线程安全地添加到倒排列表
            {
                std::lock_guard<std::mutex> lock(*cluster_mutexes[cluster_id]);
                invlists[cluster_id].push_back(point_id);
                cluster_sizes[cluster_id]++;
            }
            
            // 如果HNSW索引尚未创建，创建它
            if (!hnsw_indices[cluster_id]) {
                std::lock_guard<std::mutex> lock(*cluster_mutexes[cluster_id]);
                if (!hnsw_indices[cluster_id]) {  // 双重检查
                    hnsw_indices[cluster_id] = new hnswlib::HierarchicalNSW<float>(
                        hnsw_spaces[cluster_id], max_elements_per_cluster, M, ef_construction
                    );
                }
            }
            
            // 添加到对应的HNSW索引
            try {
                hnsw_indices[cluster_id]->addPoint(vec, point_id);
            } catch (const std::exception& e) {
                // 如果HNSW索引已满，扩容或跳过
                std::cerr << "Warning: Failed to add point to HNSW cluster " 
                         << cluster_id << ": " << e.what() << std::endl;
            }
        }
        
        std::cout << "Successfully added vectors to index\n";
    }

    // 搜索函数
    std::priority_queue<std::pair<float, int>> search(
            const float* query, 
            size_t k, 
            size_t nprobe = 10,
            size_t ef = 50) const {
        
        if (!trained) {
            std::cerr << "Error: Index must be trained before searching\n";
            return std::priority_queue<std::pair<float, int>>();
        }
        
        nprobe = std::min(nprobe, nlist);
        
        // 1. 找到最近的nprobe个聚类
        std::vector<std::pair<float, int>> cluster_distances;
        cluster_distances.reserve(nlist);
        
        for (size_t i = 0; i < nlist; i++) {
            float dist = computeDistanceOptimized(query, centroids + i * dim);
            cluster_distances.push_back({dist, static_cast<int>(i)});
        }
        
        std::partial_sort(
            cluster_distances.begin(),
            cluster_distances.begin() + nprobe,
            cluster_distances.end()
        );
        
        // 2. 在选定的聚类中使用HNSW搜索
        std::priority_queue<std::pair<float, int>> global_results;
        
        for (size_t p = 0; p < nprobe; p++) {
            int cluster_id = cluster_distances[p].second;
            
            // 跳过空聚类或没有HNSW索引的聚类
            if (cluster_sizes[cluster_id] == 0 || !hnsw_indices[cluster_id]) {
                continue;
            }
            
            // 设置HNSW搜索参数
            hnsw_indices[cluster_id]->setEf(ef);
            
            try {
                // 在当前聚类的HNSW索引中搜索
                auto cluster_results = hnsw_indices[cluster_id]->searchKnn(query, k);
                
                // 合并结果到全局结果中
                while (!cluster_results.empty()) {
                    auto result = cluster_results.top();
                    cluster_results.pop();
                    
                    if (global_results.size() < k) {
                        global_results.push(result);
                    } else if (result.first < global_results.top().first) {
                        global_results.pop();
                        global_results.push(result);
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Warning: HNSW search failed for cluster " 
                         << cluster_id << ": " << e.what() << std::endl;
            }
        }
        
        return global_results;
    }

    // 带精确重排序的搜索
    std::priority_queue<std::pair<float, int>> searchWithRerank(
            const float* query,
            size_t k,
            size_t nprobe = 10,
            size_t ef = 50,
            size_t rerank_factor = 3) const {
        
        // 先搜索更多候选
        size_t candidate_k = k * rerank_factor;
        auto candidates = search(query, candidate_k, nprobe, ef);
        
        if (!raw_data) {
            return candidates;  // 无法重排序，返回原始结果
        }
        
        // 收集候选结果
        std::vector<std::pair<float, int>> candidate_list;
        while (!candidates.empty()) {
            candidate_list.push_back(candidates.top());
            candidates.pop();
        }
        
        // 使用原始数据进行精确重排序
        std::priority_queue<std::pair<float, int>> reranked_results;
        for (const auto& cand : candidate_list) {
            int id = cand.second;
            
            // 计算精确距离
            float exact_dist = computeDistanceOptimized(query, raw_data + (size_t)id * dim);
            
            if (reranked_results.size() < k) {
                reranked_results.push({exact_dist, id});
            } else if (exact_dist < reranked_results.top().first) {
                reranked_results.pop();
                reranked_results.push({exact_dist, id});
            }
        }
        
        return reranked_results;
    }

    // 保存索引 - 包含HNSW索引的完整保存
    void saveIndex(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open()) {
            std::cerr << "Error: Cannot open file for writing: " << filename << std::endl;
            return;
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
    }

    // 加载索引 - 包含HNSW索引的完整加载
    void loadIndex(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) {
            std::cerr << "Error: Cannot open file for reading: " << filename << std::endl;
            return;
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

    // 获取统计信息
    void printStats() const {
        std::cout << "=== IVF+HNSW Index Statistics ===" << std::endl;
        std::cout << "Dimension: " << dim << std::endl;
        std::cout << "Number of clusters: " << nlist << std::endl;
        std::cout << "Total vectors: " << n_vectors << std::endl;
        
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
        std::cout << "=================================" << std::endl;
    }
};

// 便捷的搜索接口函数 - 优化版，无需每次重建
std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> 
ivfhnsw_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k, 
               size_t nprobe = 16, size_t ef = 50, bool rerank = true, int num_threads = 1) {
    
    static IVFHNSW* index = nullptr;
    static bool initialized = false;
    
    // 首次调用时初始化索引
    if (!initialized) {
        // 创建files目录
        #ifdef _WIN32
            (void)system("mkdir files 2>nul");
        #else
            (void)system("mkdir -p files");
        #endif
        
        std::string path_index = "files/hnsw/ivfhnsw.index";
        std::ifstream f(path_index);
        bool index_exists = f.good();
        f.close();
        
        // 选择合适的参数
        size_t nlist = std::min(size_t(std::sqrt(base_number) * 2), base_number / 50);
        if (nlist < 10) nlist = 10;
        if (nlist > 512) nlist = 512;
        
        index = new IVFHNSW(vecdim, nlist, 20000, 200, 16);
        
        if (index_exists) {
            std::cout << "Loading existing IVF+HNSW index...\n";
            index->loadIndex(path_index);
            // 不再需要重建！HNSW索引已经从文件中加载
        } else {
            std::cout << "Building new IVF+HNSW index...\n";
            index->train(base, base_number);
            index->add(base, base_number);
            index->saveIndex(path_index);
            index->printStats();
        }
        initialized = true;
    }
    
    // 执行搜索
    std::priority_queue<std::pair<float, int>> results;
    if (rerank) {
        results = index->searchWithRerank(query, k, nprobe, ef, 3);
    } else {
        results = index->search(query, k, nprobe, ef);
    }
    
    // 转换为兼容格式
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> output;
    while (!results.empty()) {
        output.push({results.top().first, results.top().second});
        results.pop();
    }
    
    return output;
}