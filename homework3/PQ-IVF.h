#pragma once

#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <cstring>
#include <cassert>
#include <chrono> // 添加计时库
#include "pq.h"

// PQ-IVF类 - 先对所有数据进行PQ量化，再构建IVF索引
class PQ_IVF {
private:
    size_t dim;                // 向量维度
    size_t n_clusters;         // IVF聚类中心数量
    size_t pq_m;               // PQ子空间数量
    size_t pq_ks;              // PQ每个子空间的聚类数量
    bool trained;              // 是否已训练
    
    ProductQuantizer* pq;                  // PQ量化器
    std::vector<std::vector<int>> invlists; // 倒排列表
    float* centroids;                      // IVF聚类中心（基于PQ编码的聚类）
    size_t code_size;                      // PQ编码大小
    const float* raw_vectors;              // 原始向量指针（用于重排序）
    std::vector<uint8_t> codes;            // 所有向量的PQ编码

    // 计算两个PQ编码之间的距离
    float compute_code_distance(const uint8_t* code1, const uint8_t* code2) const {
        float dist = 0.0f;
        for (size_t m = 0; m < pq_m; m++) {
            // 简单汉明距离（相同为0，不同为1）
            if (code1[m] != code2[m]) {
                dist += 1.0f;
            }
        }
        return dist;
    }

    // 计算PQ编码与聚类中心的距离
    float compute_centroid_distance(const uint8_t* code, int centroid_id) const {
        float dist = 0.0f;
        // 假设centroids存储了每个聚类中心的代表性PQ编码
        const float* centroid_code = centroids + centroid_id * pq_m;
        
        // 计算简化的距离度量
        for (size_t m = 0; m < pq_m; m++) {
            float diff = static_cast<float>(code[m]) - centroid_code[m];
            dist += diff * diff;
        }
        return dist;
    }

    // K-means++ 选择初始中心点
    void select_initial_centroids(const std::vector<uint8_t>& all_codes, size_t n, std::vector<int>& centroid_ids) {
        centroid_ids.clear();
        centroid_ids.reserve(n_clusters);
        
        // 随机选择第一个中心
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> distrib(0, n - 1);
        int first_id = distrib(gen);
        centroid_ids.push_back(first_id);
        
        // 计算剩余中心点
        std::vector<float> min_distances(n, std::numeric_limits<float>::max());
        
        for (size_t c = 1; c < n_clusters; c++) {
            // 更新最小距离
            int last_id = centroid_ids.back();
            const uint8_t* last_center_code = all_codes.data() + (size_t)last_id * pq_m;
            
            float sum_distances = 0.0f;
            for (size_t i = 0; i < n; i++) {
                const uint8_t* code = all_codes.data() + i * pq_m;
                float dist = compute_code_distance(code, last_center_code);
                min_distances[i] = std::min(min_distances[i], dist);
                sum_distances += min_distances[i];
            }
            
            // 按概率选择下一个中心
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

    // 找到最近的聚类中心
    int find_nearest_centroid(const uint8_t* code) const {
        int closest = 0;
        float min_dist = compute_centroid_distance(code, 0);
        
        for (size_t i = 1; i < n_clusters; i++) {
            float dist = compute_centroid_distance(code, i);
            if (dist < min_dist) {
                min_dist = dist;
                closest = i;
            }
        }
        
        return closest;
    }

public:
    PQ_IVF(size_t d, size_t m = 8, size_t ks = 256, size_t clusters = 256)
        : dim(d), n_clusters(clusters), pq_m(m), pq_ks(ks), trained(false), raw_vectors(nullptr) {
        
        // 初始化PQ量化器
        pq = new ProductQuantizer(d, m, ks);
        code_size = m;  // PQ编码大小 = 子空间数量
        
        // 初始化倒排列表
        invlists.resize(n_clusters);
        
        // 预分配聚类中心空间（将存储每个中心对应的编码特征）
        centroids = new float[n_clusters * pq_m];
    }

    ~PQ_IVF() {
        if (pq) delete pq;
        if (centroids) delete[] centroids;
    }

    // 训练索引 - 先PQ再做聚类
    void train(const float* data, size_t n) {
        if (trained) {
            std::cerr << "Warning: PQ-IVF index already trained\n";
            return;
        }

        // 总训练开始计时
        auto total_start = std::chrono::high_resolution_clock::now();

        std::cout << "1. Training PQ quantizer..." << std::endl;
        // 1. 训练PQ量化器
        auto pq_train_start = std::chrono::high_resolution_clock::now();
        pq->train(data, n);
        auto pq_train_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> pq_train_time = pq_train_end - pq_train_start;
        std::cout << "   PQ quantizer training completed in " << pq_train_time.count() << " seconds" << std::endl;
        
        std::cout << "2. Encoding all vectors with PQ..." << std::endl;
        // 2. 使用PQ量化所有向量
        auto encoding_start = std::chrono::high_resolution_clock::now();
        codes.resize(n * code_size);
        for (size_t i = 0; i < n; i++) {
            pq->encode(data + i * dim, codes.data() + i * code_size);
        }
        auto encoding_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> encoding_time = encoding_end - encoding_start;
        std::cout << "   PQ encoding completed in " << encoding_time.count() << " seconds" << std::endl;
        
        std::cout << "3. Training IVF clusters on PQ codes..." << std::endl;
        // 3. 在PQ编码上训练聚类中心
        auto ivf_start = std::chrono::high_resolution_clock::now();
        
        // 选择初始聚类中心
        std::vector<int> centroid_ids;
        select_initial_centroids(codes, n, centroid_ids);
        
        // 初始化聚类中心为选中的PQ编码
        for (size_t i = 0; i < n_clusters; i++) {
            const uint8_t* center_code = codes.data() + centroid_ids[i] * code_size;
            for (size_t j = 0; j < pq_m; j++) {
                centroids[i * pq_m + j] = static_cast<float>(center_code[j]);
            }
        }
        
        // K-means迭代
        std::vector<int> assignments(n, -1);
        std::vector<int> cluster_sizes(n_clusters, 0);
        
        const int max_iter = 20;
        for (int iter = 0; iter < max_iter; iter++) {
            bool converged = true;
            
            // 分配点到最近的聚类
            #pragma omp parallel for
            for (size_t i = 0; i < n; i++) {
                uint8_t* code = codes.data() + i * code_size;
                int closest = find_nearest_centroid(code);
                if (assignments[i] != closest) {
                    assignments[i] = closest;
                    converged = false;
                }
            }
            
            if (converged && iter > 0) break;
            
            // 重新计算聚类中心
            std::fill(cluster_sizes.begin(), cluster_sizes.end(), 0);
            std::vector<float> new_centroids(n_clusters * pq_m, 0);
            
            for (size_t i = 0; i < n; i++) {
                int c = assignments[i];
                cluster_sizes[c]++;
                
                const uint8_t* code = codes.data() + i * code_size;
                float* centroid = new_centroids.data() + c * pq_m;
                
                for (size_t j = 0; j < pq_m; j++) {
                    centroid[j] += static_cast<float>(code[j]);
                }
            }
            
            for (size_t i = 0; i < n_clusters; i++) {
                if (cluster_sizes[i] > 0) {
                    float inv_size = 1.0f / cluster_sizes[i];
                    float* centroid = new_centroids.data() + i * pq_m;
                    for (size_t j = 0; j < pq_m; j++) {
                        centroid[j] *= inv_size;
                    }
                    memcpy(centroids + i * pq_m, centroid, pq_m * sizeof(float));
                }
            }
            
            std::cout << "   K-means iteration " << iter + 1 << " complete" << std::endl;
        }
        
        auto ivf_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> ivf_time = ivf_end - ivf_start;
        std::cout << "   IVF clustering completed in " << ivf_time.count() << " seconds" << std::endl;
        
        // 4. 构建倒排列表
        std::cout << "4. Building inverted lists..." << std::endl;
        auto invlists_start = std::chrono::high_resolution_clock::now();
        
        // 清空倒排列表
        for (auto& list : invlists) {
            list.clear();
        }
        
        for (size_t i = 0; i < n; i++) {
            uint8_t* code = codes.data() + i * code_size;
            int cluster_id = find_nearest_centroid(code);
            invlists[cluster_id].push_back(i);
        }
        
        auto invlists_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> invlists_time = invlists_end - invlists_start;
        std::cout << "   Inverted lists built in " << invlists_time.count() << " seconds" << std::endl;
        
        // 保存原始向量的指针，用于重排序
        raw_vectors = data;
        trained = true;
        
        // 计算总训练时间
        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_time = total_end - total_start;
        std::cout << "PQ-IVF index trained successfully in " << total_time.count() << " seconds!" << std::endl;
        
        // 打印各阶段耗时统计
        std::cout << "\n===== Training Time Summary =====" << std::endl;
        std::cout << "PQ Training:      " << pq_train_time.count() << " seconds" << std::endl;
        std::cout << "PQ Encoding:      " << encoding_time.count() << " seconds" << std::endl;
        std::cout << "IVF Clustering:   " << ivf_time.count() << " seconds" << std::endl;
        std::cout << "Inverted Lists:   " << invlists_time.count() << " seconds" << std::endl;
        std::cout << "Total Training:   " << total_time.count() << " seconds" << std::endl;
        std::cout << "===============================" << std::endl;
    }

    // 添加一批向量到索引
    void add(const float* data, size_t n) {
        if (!trained) {
            std::cerr << "Error: PQ-IVF index must be trained before adding vectors" << std::endl;
            return;
        }
        
        size_t old_size = codes.size() / code_size;
        size_t new_size = old_size + n;
        
        // 调整编码数组大小
        codes.resize(new_size * code_size);
        
        // 编码新向量
        auto encoding_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n; i++) {
            pq->encode(data + i * dim, codes.data() + (old_size + i) * code_size);
        }
        auto encoding_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> encoding_time = encoding_end - encoding_start;
        
        // 将向量添加到倒排列表
        auto assign_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n; i++) {
            uint8_t* code = codes.data() + (old_size + i) * code_size;
            int cluster_id = find_nearest_centroid(code);
            invlists[cluster_id].push_back(old_size + i);
        }
        auto assign_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> assign_time = assign_end - assign_start;
        
        std::cout << "Added " << n << " vectors:" << std::endl;
        std::cout << "  Encoding time: " << encoding_time.count() << " seconds" << std::endl;
        std::cout << "  Assigning time: " << assign_time.count() << " seconds" << std::endl;
        std::cout << "  Total time: " << (encoding_time + assign_time).count() << " seconds" << std::endl;
        
        // 更新原始向量指针（假设调用者保证数据持续有效）
        raw_vectors = data - (old_size * dim);
    }

    // 搜索最近邻
    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query, size_t k, size_t nprobe = 10, bool rerank = true) const {
        
        if (!trained) {
            std::cerr << "Error: PQ-IVF index must be trained before searching" << std::endl;
            return std::priority_queue<std::pair<float, uint32_t>>();
        }
        
        // 限制nprobe不超过总簇数
        nprobe = std::min(nprobe, n_clusters);
        
        // 1. 对查询向量进行PQ编码
        std::vector<uint8_t> query_code(code_size);
        pq->encode(query, query_code.data());
        
        // 2. 计算查询编码与每个簇中心的距离
        std::vector<std::pair<float, int>> cluster_distances;
        cluster_distances.reserve(n_clusters);
        
        for (size_t i = 0; i < n_clusters; i++) {
            float dist = compute_centroid_distance(query_code.data(), i);
            cluster_distances.push_back({dist, static_cast<int>(i)});
        }
        
        // 部分排序，获取最近的nprobe个簇
        std::partial_sort(
            cluster_distances.begin(),
            cluster_distances.begin() + nprobe,
            cluster_distances.end()
        );
        
        // 3. 预计算查询向量的距离表（用于加速PQ距离计算）
        std::vector<float> distance_tables(pq_m * pq_ks);
        pq->compute_distance_tables(query, distance_tables.data());
        
        // 4. 在选定的簇中检索最近邻
        std::priority_queue<std::pair<float, uint32_t>> candidates;
        size_t candidates_count = 500; // 保留前k*10个候选，至少100个
        
        for (size_t p = 0; p < nprobe; p++) {
            int cluster_id = cluster_distances[p].second;
            const std::vector<int>& ids = invlists[cluster_id];
            
            for (int id : ids) {
                const uint8_t* code = codes.data() + (size_t)id * code_size;
                
                // 使用查找表计算PQ近似距离
                float sum_lut = 0;
                for (size_t m = 0; m < pq_m; m++) {
                    uint8_t centroid_idx = code[m];
                    sum_lut += distance_tables[m * pq_ks + centroid_idx];
                }
                
                // 距离是1-内积
                float dist = 1.0f - sum_lut;
                
                if (candidates.size() < candidates_count) {
                    candidates.push({dist, static_cast<uint32_t>(id)});
                } else if (dist < candidates.top().first) {
                    candidates.pop();
                    candidates.push({dist, static_cast<uint32_t>(id)});
                }
            }
        }
        
        // 5. 如果需要，用原始向量重排序
        if (rerank && raw_vectors) {
            std::vector<std::pair<float, uint32_t>> rerank_candidates;
            rerank_candidates.reserve(candidates.size());
            
            // 收集候选项
            while (!candidates.empty()) {
                rerank_candidates.push_back(candidates.top());
                candidates.pop();
            }
            
            // 用原始向量计算精确距离
            std::priority_queue<std::pair<float, uint32_t>> results;
            
            for (const auto& candidate : rerank_candidates) {
                uint32_t id = candidate.second;
                const float* vec = raw_vectors + id * dim;
                
                // 计算精确欧氏距离
                float dist = 0;
                for (size_t d = 0; d < dim; d++) {
                    float diff = query[d] - vec[d];
                    dist += diff * diff;
                }
                
                if (results.size() < k) {
                    results.push({dist, id});
                } else if (dist < results.top().first) {
                    results.pop();
                    results.push({dist, id});
                }
            }
            
            return results;
        } else {
            // 如果不重排序，直接返回候选项的前k个
            std::priority_queue<std::pair<float, uint32_t>> results;
            
            while (!candidates.empty() && results.size() < k) {
                results.push(candidates.top());
                candidates.pop();
            }
            
            return results;
        }
    }

    // 保存索引到文件
    bool save(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open()) return false;
        
        // 保存基本参数
        out.write((char*)&dim, sizeof(dim));
        out.write((char*)&n_clusters, sizeof(n_clusters));
        out.write((char*)&pq_m, sizeof(pq_m));
        out.write((char*)&pq_ks, sizeof(pq_ks));
        
        // 保存PQ量化器
        std::string pq_filename = filename + ".pq";
        pq->save(pq_filename);
        
        // 保存聚类中心
        out.write((char*)centroids, n_clusters * pq_m * sizeof(float));
        
        // 保存所有编码
        size_t n_vectors = codes.size() / code_size;
        out.write((char*)&n_vectors, sizeof(n_vectors));
        out.write((char*)codes.data(), codes.size() * sizeof(uint8_t));
        
        // 保存倒排列表
        for (size_t i = 0; i < n_clusters; i++) {
            size_t list_size = invlists[i].size();
            out.write((char*)&list_size, sizeof(list_size));
            if (list_size > 0) {
                out.write((char*)invlists[i].data(), list_size * sizeof(int));
            }
        }
        
        out.close();
        return true;
    }

    // 从文件加载索引
    bool load(const std::string& filename, const float* base_vectors) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) return false;
        
        // 读取基本参数
        in.read((char*)&dim, sizeof(dim));
        in.read((char*)&n_clusters, sizeof(n_clusters));
        in.read((char*)&pq_m, sizeof(pq_m));
        in.read((char*)&pq_ks, sizeof(pq_ks));
        
        // 释放并重新创建PQ量化器
        if (pq) delete pq;
        pq = new ProductQuantizer(dim, pq_m, pq_ks);
        
        // 加载PQ量化器
        std::string pq_filename = filename + ".pq";
        if (!pq->load(pq_filename)) return false;
        
        // 释放并重新分配聚类中心内存
        if (centroids) delete[] centroids;
        centroids = new float[n_clusters * pq_m];
        
        // 读取聚类中心
        in.read((char*)centroids, n_clusters * pq_m * sizeof(float));
        
        // 读取编码
        size_t n_vectors;
        in.read((char*)&n_vectors, sizeof(n_vectors));
        code_size = pq_m;
        codes.resize(n_vectors * code_size);
        in.read((char*)codes.data(), n_vectors * code_size * sizeof(uint8_t));
        
        // 读取倒排列表
        invlists.clear();
        invlists.resize(n_clusters);
        for (size_t i = 0; i < n_clusters; i++) {
            size_t list_size;
            in.read((char*)&list_size, sizeof(list_size));
            invlists[i].resize(list_size);
            if (list_size > 0) {
                in.read((char*)invlists[i].data(), list_size * sizeof(int));
            }
        }
        
        // 设置原始向量指针
        raw_vectors = base_vectors;
        trained = true;
        
        in.close();
        return true;
    }
};

// 提供一个简单的搜索接口函数
std::priority_queue<std::pair<float, uint32_t>> pq_ivf_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k, size_t nprobe = 16,
    bool save_index = true, const std::string& index_file = "files/pq_ivf_index", bool load_index = true) {
    
    static PQ_IVF* index = nullptr;
    static float* last_base = nullptr;
    static size_t last_base_number = 0;
    static size_t last_vecdim = 0;
    
    // 创建files文件夹
    #ifdef _WIN32
    if (system("if not exist files mkdir files") != 0) {
        // 处理可能的错误
    }
    #else
    if (system("mkdir -p files") != 0) {
        // 处理可能的错误
    }
    #endif
    
    bool index_loaded = false;
    
    // 尝试加载索引
    if (load_index && 
        (index == nullptr || base != last_base || 
         base_number != last_base_number || vecdim != last_vecdim)) {
        
        // 默认PQ参数
        size_t m = 8;      // PQ子空间数量
        size_t ks = 256;   // 每个子空间的聚类数
        size_t clusters = 1024; // IVF聚类数量
        
        PQ_IVF* temp_index = new PQ_IVF(vecdim, m, ks, clusters);
        
        if (temp_index->load(index_file, base)) {
            if (index != nullptr) delete index;
            index = temp_index;
            last_base = base;
            last_base_number = base_number;
            last_vecdim = vecdim;
            index_loaded = true;
            std::cout << "Loaded PQ-IVF index from " << index_file << std::endl;
        } else {
            delete temp_index;
        }
    }
    
    // 如果没有加载到索引，则构建新索引
    if (!index_loaded && 
        (index == nullptr || base != last_base || 
         base_number != last_base_number || vecdim != last_vecdim)) {
        
        if (index != nullptr) delete index;
        
        // 默认PQ参数
        size_t m = 8;      // PQ子空间数量
        size_t ks = 256;   // 每个子空间的聚类数
        size_t clusters = 1024; // IVF聚类数量
        
        index = new PQ_IVF(vecdim, m, ks, clusters);
        
        // 训练和构建索引
        std::cout << "Building new PQ-IVF index..." << std::endl;
        index->train(base, base_number);
        
        // 如果需要保存索引
        if (save_index) {
            if (index->save(index_file)) {
                std::cout << "Saved PQ-IVF index to " << index_file << std::endl;
            }
        }
        
        last_base = base;
        last_base_number = base_number;
        last_vecdim = vecdim;
    }
    
    // 执行搜索
    return index->search(query, k, nprobe, true);
}