#pragma once

#include <vector>
#include <algorithm>
#include <queue>
#include <random>
#include <cstring>
#include <limits>
#include <cassert>
#include <fstream>
#include <string>
#include <arm_neon.h>
#include "simd.h"
#include <stdlib.h>
#include <malloc.h>
#include <stdint.h>
#include <new>
#include <iostream>

// 产品量化器类
class ProductQuantizer {
private:
    size_t dim;        // 向量维度
    size_t M;          // 子空间数量
    size_t Ks;         // 每个子空间的聚类数量
    size_t dsub;       // 每个子空间的维度
    std::vector<float> centroids; // 存储所有聚类中心

    // 原始 K-means 训练函数中的并行指令需要移除
    void train_kmeans(const float* data, size_t n, size_t sub_dim, 
                     size_t k, float* centroids, size_t max_iter = 20) {
        std::vector<size_t> indices(n);
        for (size_t i = 0; i < n; i++) {
            indices[i] = i;
        }
        
        // 随机初始化聚类中心
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        
        for (size_t i = 0; i < k; i++) {
            memcpy(centroids + i * sub_dim, data + indices[i] * sub_dim, 
                   sub_dim * sizeof(float));
        }
        
        std::vector<size_t> assign(n);
        std::vector<size_t> cluster_size(k);
        std::vector<float> cluster_sum(k * sub_dim);
        
        // k-means迭代优化
        for (size_t iter = 0; iter < max_iter; iter++) {
            // 分配向量到最近的中心 (移除 #pragma omp parallel for)
            for (size_t i = 0; i < n; i++) {
                float min_dist = std::numeric_limits<float>::max();
                size_t min_idx = 0;
                
                for (size_t j = 0; j < k; j++) {
                    float dist = 0;
                    for (size_t d = 0; d < sub_dim; d++) {
                        float diff = data[i * sub_dim + d] - centroids[j * sub_dim + d];
                        dist += diff * diff;
                    }
                    
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_idx = j;
                    }
                }
                assign[i] = min_idx;
            }
            
            // 重新计算聚类中心
            std::fill(cluster_size.begin(), cluster_size.end(), 0);
            std::fill(cluster_sum.begin(), cluster_sum.end(), 0);
            
            for (size_t i = 0; i < n; i++) {
                size_t cluster_id = assign[i];
                cluster_size[cluster_id]++;
                for (size_t d = 0; d < sub_dim; d++) {
                    cluster_sum[cluster_id * sub_dim + d] += data[i * sub_dim + d];
                }
            }
            
            for (size_t j = 0; j < k; j++) {
                if (cluster_size[j] > 0) {
                    for (size_t d = 0; d < sub_dim; d++) {
                        centroids[j * sub_dim + d] = cluster_sum[j * sub_dim + d] / cluster_size[j];
                    }
                }
            }
        }
    }
public:
    ProductQuantizer(size_t d, size_t m = 8, size_t ks = 256) : dim(d), M(m), Ks(ks) {
        assert(d % m == 0); // 确保维度能被子空间数量整除
        dsub = d / m;
        centroids.resize(M * Ks * dsub);
    }
    
    // 训练量化器
    void train(const float* vectors, size_t n) {
        std::vector<float> sub_vectors(n * dsub);
        
        for (size_t m = 0; m < M; m++) {
            // 提取第m个子空间的所有向量
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < dsub; j++) {
                    sub_vectors[i * dsub + j] = vectors[i * dim + m * dsub + j];
                }
            }
            
            // 训练该子空间的聚类中心
            train_kmeans(sub_vectors.data(), n, dsub, Ks, 
                         centroids.data() + m * Ks * dsub);
        }
    }
    
    // 编码单个向量
    void encode(const float* vector, uint8_t* code) const {
        for (size_t m = 0; m < M; m++) {
            float min_dist = std::numeric_limits<float>::max();
            uint8_t min_idx = 0;
            
            for (size_t k = 0; k < Ks; k++) {
                float dist = 0;
                for (size_t d = 0; d < dsub; d++) {
                    float diff = vector[m * dsub + d] - centroids[(m * Ks + k) * dsub + d];
                    dist += diff * diff;
                }
                
                if (dist < min_dist) {
                    min_dist = dist;
                    min_idx = k;
                }
            }
            code[m] = min_idx;
        }
    }
    
    // 计算查询向量与每个聚类中心的距离表
    void compute_distance_tables(const float* query, float* tables) const {
        // 根据图片，我们需要为每个子空间计算查找表
        for (size_t m = 0; m < M; m++) {
            for (size_t k = 0; k < Ks; k++) {
                const float* centroid = centroids.data() + (m * Ks + k) * dsub;
                float ip = 0; // 内积
                for (size_t d = 0; d < dsub; d++) {
                    ip += query[m * dsub + d] * centroid[d];
                }
                // 存储内积到查找表
                tables[m * Ks + k] = ip;
            }
        }
    }
    
    size_t get_code_size() const { return M; }
    size_t get_M() const { return M; }
    size_t get_Ks() const { return Ks; }

    // 保存量化器参数到文件
    bool save(const std::string& filename) const {
        std::ofstream out(filename, std::ios::binary);
        if (!out.is_open()) return false;
        
        // 保存基本参数
        out.write((char*)&dim, sizeof(dim));
        out.write((char*)&M, sizeof(M));
        out.write((char*)&Ks, sizeof(Ks));
        out.write((char*)&dsub, sizeof(dsub));
        
        // 保存聚类中心
        size_t centroids_size = centroids.size();
        out.write((char*)&centroids_size, sizeof(centroids_size));
        out.write((char*)centroids.data(), centroids.size() * sizeof(float));
        
        return true;
    }
    
    // 从文件加载量化器参数
    bool load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in.is_open()) return false;
        
        // 读取基本参数
        in.read((char*)&dim, sizeof(dim));
        in.read((char*)&M, sizeof(M));
        in.read((char*)&Ks, sizeof(Ks));
        in.read((char*)&dsub, sizeof(dsub));
        
        // 读取聚类中心
        size_t centroids_size;
        in.read((char*)&centroids_size, sizeof(centroids_size));
        centroids.resize(centroids_size);
        in.read((char*)centroids.data(), centroids_size * sizeof(float));
        
        return true;
    }
};

// PQ索引类
class PQIndex {
private:
    ProductQuantizer pq;
    size_t dim;
    size_t n_vectors;
    std::vector<uint8_t> codes;
    std::vector<uint8_t> transposed_codes;  // 新增："子空间优先"编码
    const float* raw_vectors; // 保存原始向量用于重排序

    // 分配对齐内存的辅助函数
    static float* allocate_aligned_float(size_t num_elements, size_t alignment) {
        void* ptr = nullptr;
        #if defined(__linux__) || defined(__ANDROID__) || (defined(__APPLE__) && defined(__MACH__)) || (defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L)
            if (posix_memalign(&ptr, alignment, num_elements * sizeof(float)) != 0) {
                ptr = nullptr;
            }
        #elif defined(__linux__) || defined(__ANDROID__)
            ptr = memalign(alignment, num_elements * sizeof(float));
        #elif defined(_WIN32) && defined(_M_ARM)
            ptr = _aligned_malloc(num_elements * sizeof(float), alignment);
        #else
            #if __cplusplus >= 201103L && defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && !defined(__APPLE__)
                ptr = aligned_alloc(alignment, ((num_elements * sizeof(float) + alignment - 1) / alignment) * alignment);
            #else
                ptr = malloc(num_elements * sizeof(float));
                std::cerr << "Warning: Using potentially unaligned memory allocation." << std::endl;
            #endif
        #endif

        if (!ptr) {
            throw std::bad_alloc();
        }
        return static_cast<float*>(ptr);
    }

    // 释放对齐内存的辅助函数
    static void free_aligned(float* ptr) {
        if (!ptr) return;
        #if defined(_WIN32) && defined(_M_ARM)
            _aligned_free(ptr);
        #else
            free(ptr);
        #endif
    }

public:
    PQIndex(size_t d, size_t m = 8) : pq(d, m), dim(d), n_vectors(0), raw_vectors(nullptr) {}
    
    // 构建索引 - 修改此函数添加转置编码
    void build(const float* vectors, size_t n) {
        n_vectors = n;
        size_t code_size = pq.get_code_size();
        codes.resize(n * code_size);
        raw_vectors = vectors;
        
        // 训练PQ量化器
        pq.train(vectors, n);
        
        // 编码所有向量
        for (size_t i = 0; i < n; i++) {    
            pq.encode(vectors + i * dim, codes.data() + i * code_size);
        }
        
        // 创建转置编码 - 用于SIMD优化
        create_transposed_codes();
    }
    // 新增：创建转置编码
    void create_transposed_codes() {
        size_t code_size = pq.get_code_size();
        size_t M = pq.get_M();
        
        // 确保代码一致性
        assert(M == code_size);
        
        // 分配转置编码空间
        transposed_codes.resize(n_vectors * code_size);
        
        // 执行转置：从"向量优先"到"子空间优先"
        for (size_t m = 0; m < M; m++) {
            for (size_t i = 0; i < n_vectors; i++) {
                // 原始编码中的位置: i * code_size + m
                // 转置后的位置: m * n_vectors + i
                transposed_codes[m * n_vectors + i] = codes[i * code_size + m];
            }
        }
    }
// ANN搜索，带重排序功能
    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query, size_t k, size_t rerank_count = 0) const {
        
        // 如果rerank_count为0，设置默认值为k的3倍
        if (rerank_count == 0) rerank_count = k * 3;
        
        // 确保rerank_count不小于k
        rerank_count = std::max(rerank_count, k);
        
        // 第一阶段：使用PQ量化索引找出候选
        std::priority_queue<std::pair<float, uint32_t>> candidates;
        size_t code_size = pq.get_code_size();
        
        // 预计算距离表
        std::vector<float> distance_tables(pq.get_M() * pq.get_Ks());
        pq.compute_distance_tables(query, distance_tables.data());
        
        // 计算每个向量的近似距离
        for (size_t i = 0; i < n_vectors; i++) {
            const uint8_t* code = codes.data() + i * code_size;
            
            // 使用查找表法快速计算距离：δ(x,q) = 1 - ∑(LUT[k,o_k])
            float sum_lut = 0;
            for (size_t m = 0; m < pq.get_M(); m++) {
                uint8_t centroid_idx = code[m]; // o_k
                sum_lut += distance_tables[m * pq.get_Ks() + centroid_idx];
            }
            
            // 计算距离 (1 - 内积)
            float dist = 1 - sum_lut;
            
            if (candidates.size() < rerank_count) {
                candidates.push({dist, static_cast<uint32_t>(i)});
            } else if (dist < candidates.top().first) {
                candidates.push({dist, static_cast<uint32_t>(i)});
                candidates.pop();
            }
        }
        
        // 第二阶段：对候选项进行全精度重排序
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
            uint32_t idx = candidate.second;
            const float* vec = raw_vectors + idx * dim;
            
            // 计算精确内积
            float ip = 0;
            for (size_t d = 0; d < dim; d++) {
                ip += query[d] * vec[d];
            }
            
            float dist = 1 - ip;
            
            if (results.size() < k) {
                results.push({dist, idx});
            } else if (dist < results.top().first) {
                results.push({dist, idx});
                results.pop();
            }
        }
        
        return results;
    }
    // 使用转置编码的SIMD搜索实现
    std::priority_queue<std::pair<float, uint32_t>> simdsearch(
        const float* query, size_t k, size_t rerank_count = 0) const {
        
        // 如果rerank_count为0，设置默认值为k的3倍
        if (rerank_count == 0) rerank_count = k * 3;
        
        // 确保rerank_count不小于k
        rerank_count = std::max(rerank_count, k);
        
        // 第一阶段：使用PQ量化索引找出候选
        std::priority_queue<std::pair<float, uint32_t>> candidates;
        size_t code_size = pq.get_code_size();
        const size_t alignment = 16; // ARM Neon 通常需要 16 字节 (128位) 对齐

        // 检查转置编码是否存在
        if (transposed_codes.empty()) {
             std::cerr << "错误: 转置编码未生成，无法执行 SIMD 搜索。" << std::endl;
             return {}; // 返回空结果
        }

        // 预计算距离表 - 使用对齐内存分配
        size_t distance_tables_size = pq.get_M() * pq.get_Ks();
        float* distance_tables = nullptr;
        try {
            // 使用辅助函数分配对齐内存
            distance_tables = allocate_aligned_float(distance_tables_size, alignment);
        } catch (const std::bad_alloc& e) {
            std::cerr << "错误: 分配距离表对齐内存失败。" << std::endl;
            return {}; // 返回空结果
        }

        // 计算查找表 (标量操作，写入到对齐的 distance_tables)
        pq.compute_distance_tables(query, distance_tables);

        // 主循环：计算近似距离，分组SIMD处理
        for (size_t i = 0; i < n_vectors; i += 8) {
            // 确保栈上的临时数组是对齐的
            alignas(alignment) float MResult1[8] = {0};
            alignas(alignment) float MResult2[8] = {0};
            alignas(alignment) float MResult3[8] = {0};
            alignas(alignment) float MResult4[8] = {0};
            alignas(alignment) float MResult5[8] = {0};
            alignas(alignment) float MResult6[8] = {0};
            alignas(alignment) float MResult7[8] = {0};
            alignas(alignment) float MResult8[8] = {0};

            size_t batch_size = std::min(size_t(8), n_vectors - i);

            // 获取指向当前批次转置编码的指针 (这些指针本身不需要特殊对齐，它们指向的数据在 transposed_codes 中)
            const uint8_t* subspace0_codes = transposed_codes.data() + 0 * n_vectors + i;
            const uint8_t* subspace1_codes = transposed_codes.data() + 1 * n_vectors + i;
            const uint8_t* subspace2_codes = transposed_codes.data() + 2 * n_vectors + i;
            const uint8_t* subspace3_codes = transposed_codes.data() + 3 * n_vectors + i;
            const uint8_t* subspace4_codes = transposed_codes.data() + 4 * n_vectors + i;
            const uint8_t* subspace5_codes = transposed_codes.data() + 5 * n_vectors + i;
            const uint8_t* subspace6_codes = transposed_codes.data() + 6 * n_vectors + i;
            const uint8_t* subspace7_codes = transposed_codes.data() + 7 * n_vectors + i;

            // 从连续内存加载编码并查找距离表值 (标量操作)
            for (size_t j = 0; j < batch_size; j++) {
                MResult1[j] = distance_tables[0 * pq.get_Ks() + subspace0_codes[j]];
                MResult2[j] = distance_tables[1 * pq.get_Ks() + subspace1_codes[j]];
                MResult3[j] = distance_tables[2 * pq.get_Ks() + subspace2_codes[j]];
                MResult4[j] = distance_tables[3 * pq.get_Ks() + subspace3_codes[j]];
                MResult5[j] = distance_tables[4 * pq.get_Ks() + subspace4_codes[j]];
                MResult6[j] = distance_tables[5 * pq.get_Ks() + subspace5_codes[j]];
                MResult7[j] = distance_tables[6 * pq.get_Ks() + subspace6_codes[j]];
                MResult8[j] = distance_tables[7 * pq.get_Ks() + subspace7_codes[j]];
            }

            // SIMD 计算部分
            simd8float32 tmp1(MResult1), tmp2(MResult2), tmp3(MResult3), tmp4(MResult4),
                         tmp5(MResult5), tmp6(MResult6), tmp7(MResult7), tmp8(MResult8);
            simd8float32 result(1.0f);
            simd8float32 sum = tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6 + tmp7 + tmp8;
            result = result - sum;
            float dist[8];
            result.storeu(dist);

            // 收集一批结果
            std::pair<float, uint32_t> batch_results[8];
            for(size_t j = 0; j < batch_size; j++) {
                batch_results[j] = {dist[j], i + j};
            }

            // 批量更新优先队列
            for(size_t j = 0; j < batch_size; j++) {
                if(candidates.size() < rerank_count || batch_results[j].first < candidates.top().first) {
                    candidates.push(batch_results[j]);
                    if (candidates.size() > rerank_count) {
                        candidates.pop();
                    }
                }
            }
        }

        // 释放对齐分配的距离表内存
        free_aligned(distance_tables);

        // 第二阶段：对候选项进行全精度重排序
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
            uint32_t idx = candidate.second;
            float dist = InnerProductSIMDNeon(
                query, 
                raw_vectors + idx * dim, 
                dim
            );
            if (results.size() < k) {
                results.push({dist, idx});
            } else if (dist < results.top().first) {
                results.push({dist, idx});
                results.pop();
            }
        }
        
        return results;
    }
   // 保存函数也需要保存转置编码
   bool save(const std::string& filename) const {
    // 保存PQ量化器
    std::string pq_filename = filename + ".pq";
    if (!pq.save(pq_filename)) return false;
    
    // 保存编码后的向量
    std::string codes_filename = filename + ".codes";
    std::ofstream out(codes_filename, std::ios::binary);
    if (!out.is_open()) return false;
    
    // 保存基本参数
    out.write((char*)&dim, sizeof(dim));
    out.write((char*)&n_vectors, sizeof(n_vectors));
    
    // 保存编码向量
    size_t code_size = pq.get_code_size();
    out.write((char*)&code_size, sizeof(code_size));
    out.write((char*)codes.data(), n_vectors * code_size * sizeof(uint8_t));
    
    // 保存转置编码
    out.write((char*)transposed_codes.data(), n_vectors * code_size * sizeof(uint8_t));
    
    return true;
}

// 加载函数也需要加载转置编码
bool load(const std::string& filename, const float* vectors) {
    // 加载PQ量化器
    std::string pq_filename = filename + ".pq";
    if (!pq.load(pq_filename)) return false;
    
    // 加载编码后的向量
    std::string codes_filename = filename + ".codes";
    std::ifstream in(codes_filename, std::ios::binary);
    if (!in.is_open()) return false;
    
    // 读取基本参数
    in.read((char*)&dim, sizeof(dim));
    in.read((char*)&n_vectors, sizeof(n_vectors));
    
    // 读取编码向量
    size_t code_size;
    in.read((char*)&code_size, sizeof(code_size));
    codes.resize(n_vectors * code_size);
    in.read((char*)codes.data(), n_vectors * code_size * sizeof(uint8_t));
    
    // 尝试读取转置编码 - 如果文件中有的话
    transposed_codes.resize(n_vectors * code_size);
    if (!in.eof()) {
        in.read((char*)transposed_codes.data(), n_vectors * code_size * sizeof(uint8_t));
    } else {
        // 如果旧格式文件没有转置编码，则创建
        create_transposed_codes();
    }
    
    // 设置原始向量引用（用于重排序）
    raw_vectors = vectors;
    
    return true;
}
};

// 修改pq_search函数，实现在files文件夹中保存索引
std::priority_queue<std::pair<float, uint32_t>> pq_search(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k, size_t rerank = 100,
    bool save_index = true, const std::string& index_file = "files", bool load_index = true) {
    
    static PQIndex* index = nullptr;
    static float* last_base = nullptr;
    static size_t last_base_number = 0;
    static size_t last_vecdim = 0;
    
    // 修正创建files文件夹部分，处理system返回值
    #ifdef _WIN32
        if (system("if not exist files mkdir files") != 0) {
            // 可以添加错误处理，或者忽略但不触发警告
        }
        #else
        if (system("mkdir -p files") != 0) {
            // 可以添加错误处理，或者忽略但不触发警告
        }
    #endif
    
    std::string actual_index_file = index_file;
    if (actual_index_file == "files") {
        // 在files文件夹中使用数据特征自动生成索引文件名
        actual_index_file = "files/index_" + std::to_string(base_number) + "_" + std::to_string(vecdim);
    } else {
        // 如果提供了自定义文件名，也放在files文件夹中
        if (actual_index_file.find('/') == std::string::npos && 
            actual_index_file.find('\\') == std::string::npos) {
            actual_index_file = "files/" + actual_index_file;
        }
    }
    
    bool index_loaded = false;
    
    // 尝试加载索引，如果启用了加载和提供了文件名
    if (load_index && !actual_index_file.empty() && 
        (index == nullptr || base != last_base || 
         base_number != last_base_number || vecdim != last_vecdim)) {
        
        // 定义子空间数量
        size_t m = 8;
        
        PQIndex* temp_index = new PQIndex(vecdim, m);
        
        // 尝试加载索引
        if (temp_index->load(actual_index_file, base)) {
            // 释放旧索引（如果有）
            if (index != nullptr) {
                delete index;
            }
            
            index = temp_index;
            
            // 保存当前数据信息
            last_base = base;
            last_base_number = base_number;
            last_vecdim = vecdim;
            
            index_loaded = true;
        } else {
            // 加载失败，删除临时索引
            delete temp_index;
        }
    }
    
    // 如果没有加载到索引，或者数据发生变化，则需要重建索引
    if (!index_loaded && (index == nullptr || base != last_base || 
        base_number != last_base_number || vecdim != last_vecdim)) {
        
        // 释放旧索引
        if (index != nullptr) {
            delete index;
        }
        
        // 定义子空间数量
        size_t m = 8;
        
        index = new PQIndex(vecdim, m);
        
        // 构建索引
        index->build(base, base_number);
        
        // 如果需要保存索引
        if (save_index && !actual_index_file.empty()) {
            index->save(actual_index_file);
        }
        
        // 保存当前数据信息
        last_base = base;
        last_base_number = base_number;
        last_vecdim = vecdim;
    }
    
    return index->search(query, k, rerank);
}
// SIMD优化的pq_search函数
std::priority_queue<std::pair<float, uint32_t>> pq_search_simd(
    float* base, float* query, size_t base_number, size_t vecdim, size_t k, size_t rerank = 100,
    bool save_index = true, const std::string& index_file = "files", bool load_index = true) {
    
    static PQIndex* index = nullptr;
    static float* last_base = nullptr;
    static size_t last_base_number = 0;
    static size_t last_vecdim = 0;
    
    // 修正创建files文件夹部分，处理system返回值
    #ifdef _WIN32
        if (system("if not exist files mkdir files") != 0) {
            // 可以添加错误处理，或者忽略但不触发警告
        }
        #else
        if (system("mkdir -p files") != 0) {
            // 可以添加错误处理，或者忽略但不触发警告
        }
    #endif
    
    std::string actual_index_file = index_file;
    if (actual_index_file == "files") {
        // 在files文件夹中使用数据特征自动生成索引文件名
        actual_index_file = "files/index_" + std::to_string(base_number) + "_" + std::to_string(vecdim);
    } else {
        // 如果提供了自定义文件名，也放在files文件夹中
        if (actual_index_file.find('/') == std::string::npos && 
            actual_index_file.find('\\') == std::string::npos) {
            actual_index_file = "files/" + actual_index_file;
        }
    }
    
    bool index_loaded = false;
    
    // 尝试加载索引，如果启用了加载和提供了文件名
    if (load_index && !actual_index_file.empty() && 
        (index == nullptr || base != last_base || 
         base_number != last_base_number || vecdim != last_vecdim)) {
        
        // 定义子空间数量
        size_t m = 8;
        
        PQIndex* temp_index = new PQIndex(vecdim, m);
        
        // 尝试加载索引
        if (temp_index->load(actual_index_file, base)) {
            // 释放旧索引（如果有）
            if (index != nullptr) {
                delete index;
            }
            index = temp_index;
            // 保存当前数据信息
            last_base = base;
            last_base_number = base_number;
            last_vecdim = vecdim;
            index_loaded = true;
        } else {
            // 加载失败，删除临时索引
            delete temp_index;
        }
    }
    // 如果没有加载到索引，或者数据发生变化，则需要重建索引
    if (!index_loaded && (index == nullptr || base != last_base || 
        base_number != last_base_number || vecdim != last_vecdim)) {
        
        // 释放旧索引
        if (index != nullptr) {
            delete index;
        }
        
        // 定义子空间数量
        size_t m = 8;
        
        index = new PQIndex(vecdim, m);
        
        // 构建索引
        index->build(base, base_number);
        
        // 如果需要保存索引
        if (save_index && !actual_index_file.empty()) {
            index->save(actual_index_file);
        }
        
        // 保存当前数据信息
        last_base = base;
        last_base_number = base_number;
        last_vecdim = vecdim;
    }
    return index->simdsearch(query, k, rerank);
}