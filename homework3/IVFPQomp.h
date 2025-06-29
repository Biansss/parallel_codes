#pragma once
#include "IVF-PQ.h"
#include <omp.h>

// IVFPQ的OpenMP加速版本
class IVFPQomp : public IVFPQ {
public:
    // 使用父类构造函数
    using IVFPQ::IVFPQ;

    // 使用OpenMP加速的搜索方法
    std::priority_queue<std::pair<float, int>> search_omp(
            const float* query, 
            size_t k, 
            size_t nprobe = 10,
            bool rerank = true,
            size_t rerank_count = 500,
            int num_threads = 0) const {
        
        // 设置OpenMP线程数
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
        }
        
        // 确保nprobe不超过聚类数量
        nprobe = std::min(nprobe, nlist);
        
        // 设置重排序候选数量，如果未指定则使用k的3倍
        if (rerank_count == 0) {
            rerank_count = rerank ? k * 3 : k;
        }
        // 确保重排序候选数量不小于k
        rerank_count = std::max(rerank_count, k);
        
        // 1. 并行寻找nprobe个最近的聚类中心
        std::vector<std::pair<float, int>> cluster_distances(nlist);
        
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nlist; i++) {
            float dist = ivf->computeDistanceOptimized(query, ivf->centroids + i * dim);
            cluster_distances[i] = {dist, static_cast<int>(i)};
        }
        
        // 部分排序只需在单线程下执行
        std::partial_sort(
            cluster_distances.begin(), 
            cluster_distances.begin() + nprobe,
            cluster_distances.end()
        );
        cluster_distances.resize(nprobe);  // 只保留前nprobe个
        
        // 2. 使用线程局部结果收集
        std::vector<std::priority_queue<std::pair<float, int>>> local_results(omp_get_max_threads());
        std::vector<float> residual(dim);
        
        // 为每个线程预分配距离表空间
        std::vector<std::vector<float>> thread_distance_tables(omp_get_max_threads(), 
            std::vector<float>(pq.get_M() * pq.get_Ks()));
        
        // 并行处理选定的簇
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            auto& thread_results = local_results[thread_id];
            auto& distance_tables = thread_distance_tables[thread_id];
            std::vector<float> thread_residual(dim);
            
            #pragma omp for schedule(dynamic, 1)
            for (size_t p = 0; p < nprobe; p++) {
                int cluster_id = cluster_distances[p].second;
                float coarse_dist = cluster_distances[p].first;
                
                // 计算查询向量与该聚类中心的残差
                if (by_residual) {
                    compute_residual(query, thread_residual.data(), cluster_id);
                    
                    // 计算残差的距离表
                    pq.compute_distance_tables(thread_residual.data(), distance_tables.data());
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
                    if (thread_results.size() < rerank_count) {
                        thread_results.push({dist, id});
                    } else if (dist < thread_results.top().first) {
                        thread_results.pop();
                        thread_results.push({dist, id});
                    }
                }
            }
        }
        
        // 3. 合并线程局部结果
        std::vector<std::pair<float, int>> candidates;
        size_t total_candidates = 0;
        
        // 计算所有候选的总数
        for (auto& thread_result : local_results) {
            total_candidates += thread_result.size();
        }
        candidates.reserve(total_candidates);
        
        // 收集所有线程的候选
        for (auto& thread_result : local_results) {
            while (!thread_result.empty()) {
                candidates.push_back(thread_result.top());
                thread_result.pop();
            }
        }
        
        // 根据距离排序所有候选
        std::sort(candidates.begin(), candidates.end());
        
        // 仅保留rerank_count个候选（如果candidates超出）
        if (candidates.size() > rerank_count) {
            candidates.resize(rerank_count);
        }
        
        // 4. 重排序步骤
        std::priority_queue<std::pair<float, int>> final_results;
        
        if (rerank && raw_vectors) {
            // 并行计算精确距离
            std::vector<std::pair<float, int>> reranked_results(candidates.size());
            
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < candidates.size(); i++) {
                int id = candidates[i].second;
                
                // 计算精确距离
                float exact_dist = 0;
                for (size_t d = 0; d < dim; d++) {
                    float diff = query[d] - raw_vectors[id * dim + d];
                    exact_dist += diff * diff;
                }
                
                reranked_results[i] = {exact_dist, id};
            }
            
            // 对重排序结果排序
            std::sort(reranked_results.begin(), reranked_results.end());
            
            // 取前k个结果
            size_t result_count = std::min(k, reranked_results.size());
            for (size_t i = 0; i < result_count; i++) {
                final_results.push(reranked_results[i]);
            }
            
            // 如果向量已重排，将内部ID映射回原始ID
            if (vectors_reordered) {
                std::priority_queue<std::pair<float, int>> mapped_results;
                while (!final_results.empty()) {
                    auto top_result = final_results.top();
                    float dist = top_result.first;
                    int id = top_result.second;
                    final_results.pop();
                    // 将内部ID映射回原始ID
                    int original_id = new_to_original_ids[id];
                    mapped_results.push({dist, original_id});
                }
                return mapped_results;
            }
        } else {
            // 如果不做重排序，直接选择前k个最近的
            size_t result_count = std::min(k, candidates.size());
            for (size_t i = 0; i < result_count; i++) {
                final_results.push(candidates[i]);
            }
            
            // 如果向量已重排，将内部ID映射回原始ID
            if (vectors_reordered) {
                std::priority_queue<std::pair<float, int>> mapped_results;
                std::vector<std::pair<float, int>> temp;
                while (!final_results.empty()) {
                    temp.push_back(final_results.top());
                    final_results.pop();
                }
                for (auto it = temp.rbegin(); it != temp.rend(); ++it) {
                    mapped_results.push({it->first, new_to_original_ids[it->second]});
                }
                return mapped_results;
            }
        }
        
        return final_results;
    }
};

// 方便使用的封装函数
std::priority_queue<std::pair<float, int>> ivfpq_search_omp(
    float* base, float* query, size_t base_number, size_t vecdim, 
    size_t k, size_t nprobe = 9, bool rerank = true,
    bool save_index = true, const std::string& index_file = "files/ivfpq_omp.index",
    bool enable_reordering = true, int num_threads = 0) {
    
    static IVFPQomp* index = nullptr;
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
            IVFPQomp* temp_index = new IVFPQomp(vecdim);
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
            index = new IVFPQomp(vecdim, nlist, m);
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
    size_t rerank_count = 710;
    
    // 执行OpenMP加速的搜索，传入动态计算的重排序候选数量
    return index->search_omp(query, k, nprobe, rerank, rerank_count, num_threads);
}