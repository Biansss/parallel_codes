#pragma once
#include "IVF.h"
#include <omp.h>

// IVF的OpenMP加速版本
class IVFomp : public IVF {
public:
    // 使用父类构造函数
    using IVF::IVF;

    // 使用OpenMP加速的搜索方法
    std::priority_queue<std::pair<float, int>> search_omp(
            const float* query, 
            const float* base, 
            size_t k, 
            size_t nprobe = 10,
            int num_threads = 0) {
        
        if (!trained) {
            std::cerr << "Error: Index must be trained before searching\n";
            return std::priority_queue<std::pair<float, int>>();
        }
        
        // 设置OpenMP线程数
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        }
        
        // 限制nprobe不超过簇的数量
        nprobe = std::min(nprobe, n_clusters);
        
        // 1. 找到最近的nprobe个簇
        std::vector<std::pair<float, int>> cluster_distances(n_clusters);
        
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n_clusters; i++) {
            float dist = computeDistanceOptimized(query, centroids + i * dim);
            cluster_distances[i] = {dist, static_cast<int>(i)};
        }
        
        // 部分排序只需在单线程下执行
        std::partial_sort(
            cluster_distances.begin(), 
            cluster_distances.begin() + nprobe,
            cluster_distances.end()
        );
        cluster_distances.resize(nprobe);  // 只保留前nprobe个
        
        // 2. 线程局部结果
        std::vector<std::priority_queue<std::pair<float, int>>> local_results(omp_get_max_threads());
        
        // 使用OpenMP搜索选中的簇
        #pragma omp parallel for schedule(dynamic, 1)
        for (size_t i = 0; i < nprobe; i++) {
            int cluster_id = cluster_distances[i].second;
            int thread_id = omp_get_thread_num();
            auto& thread_results = local_results[thread_id];
            
            if (use_rearranged_data) {
                // 使用重排后的数据 - 直接访问连续存储的同一簇的数据
                size_t start = cluster_data_offsets[cluster_id];
                size_t size = cluster_data_sizes[cluster_id];
                
                for (size_t j = 0; j < size; j++) {
                    size_t rearranged_id = start + j;
                    float dist = computeDistanceOptimized(query, rearranged_data + rearranged_id * dim);
                    int original_id = rev_id_map[rearranged_id]; // 转换为原始ID
                    
                    if (thread_results.size() < k) {
                        thread_results.emplace(dist, original_id);
                    } else if (dist < thread_results.top().first) {
                        thread_results.pop();
                        thread_results.emplace(dist, original_id);
                    }
                }
            } else {
                // 使用原始方式 - 通过倒排列表访问
                const std::vector<int>& ids = invlists[cluster_id];
                
                for (int id : ids) {
                    float dist = computeDistanceOptimized(query, base + (size_t)id * dim);
                    
                    if (thread_results.size() < k) {
                        thread_results.emplace(dist, id);
                    } else if (dist < thread_results.top().first) {
                        thread_results.pop();
                        thread_results.emplace(dist, id);
                    }
                }
            }
        }
        
        // 3. 合并线程局部结果
        std::priority_queue<std::pair<float, int>> final_results;
        
        for (auto& thread_result : local_results) {
            while (!thread_result.empty()) {
                if (final_results.size() < k) {
                    final_results.push(thread_result.top());
                } else if (thread_result.top().first < final_results.top().first) {
                    final_results.pop();
                    final_results.push(thread_result.top());
                }
                thread_result.pop();
            }
        }
        
        return final_results;
    }
};

// 方便使用的封装函数
std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> 
ivf_search_omp(float* base, float* query, size_t base_number, size_t vecdim, size_t k, size_t nprobe = 16, int num_threads = 8) {
    static IVFomp* ivf = nullptr;
    static bool initialized = false;
    
    // 首次调用时初始化索引
    if (!initialized) {
        std::string path_index = "files/ivf.index";
        std::ifstream f(path_index);
        bool index_exists = f.good();
        f.close();
        
        ivf = new IVFomp(vecdim);
        
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
    
    // 执行OpenMP加速的搜索
    std::priority_queue<std::pair<float, int>> results = ivf->search_omp(query, base, k, nprobe, num_threads);
    
    // 转换为与现有代码兼容的格式
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>> output;
    while (!results.empty()) {
        output.push({results.top().first, results.top().second});
        results.pop();
    }
    
    return output;
}