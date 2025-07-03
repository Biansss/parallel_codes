#include "IVF_GPU.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cfloat>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// cuBLAS错误检查宏
#define CUBLAS_CHECK(call) \
do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d - %d\n", \
                __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// 计算查询与聚类中心的距离
__global__ void compute_cluster_distances_kernel(
    const float* centroids,
    const float* query,
    float* distances,
    size_t n_clusters,
    size_t dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_clusters) {
        float dist = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = centroids[idx * dim + d] - query[d];
            dist += diff * diff;
        }
        distances[idx] = dist;
    }
}

// 批量计算查询与聚类中心的距离
__global__ void batch_compute_cluster_distances_kernel(
    const float* centroids,
    const float* queries,
    float* distances,
    size_t n_clusters,
    size_t query_num,
    size_t dim
) {
    int cluster_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y;
    
    if (cluster_idx < n_clusters && query_idx < query_num) {
        float dist = 0.0f;
        const float* query = queries + query_idx * dim;
        const float* centroid = centroids + cluster_idx * dim;
        
        for (int d = 0; d < dim; d++) {
            float diff = centroid[d] - query[d];
            dist += diff * diff;
        }
        
        distances[query_idx * n_clusters + cluster_idx] = dist;
    }
}

// 查询特定簇中的向量
__global__ void search_cluster_vectors_kernel(
    const float* rearranged_data,
    const float* query,
    const int* cluster_ids,
    const size_t* cluster_offsets,
    const size_t* cluster_sizes,
    float* distances,
    int* indices,
    size_t dim,
    size_t nprobe,
    size_t k
) {
    // 每个线程块处理一个簇
    int probe_idx = blockIdx.x;
    if (probe_idx >= nprobe) return;
    
    // 获取当前簇信息
    int cluster_id = cluster_ids[probe_idx];
    size_t offset = cluster_offsets[cluster_id];
    size_t size = cluster_sizes[cluster_id];
    
    // 共享内存用于TopK结果
    extern __shared__ float s_mem[];
    float* s_dists = s_mem;
    int* s_indices = (int*)(s_dists + k);
    
    // 初始化
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        s_dists[i] = FLT_MAX;
        s_indices[i] = -1;
    }
    __syncthreads();
    
    // 每个线程计算一部分向量的距离
    for (size_t i = threadIdx.x; i < size; i += blockDim.x) {
        size_t vec_idx = offset + i;
        const float* vec = rearranged_data + vec_idx * dim;
        
        // 计算欧氏距离
        float dist = 0.0f;
        for (int d = 0; d < dim; d++) {
            float diff = vec[d] - query[d];
            dist += diff * diff;
        }
        
        // 尝试插入到TopK
        for (int j = 0; j < k; j++) {
            if (dist < s_dists[j]) {
                // 向后移动元素
                for (int l = k-1; l > j; l--) {
                    s_dists[l] = s_dists[l-1];
                    s_indices[l] = s_indices[l-1];
                }
                s_dists[j] = dist;
                s_indices[j] = vec_idx;
                break;
            }
        }
    }
    __syncthreads();
    
    // 第一个线程写回结果
    if (threadIdx.x == 0) {
        for (int i = 0; i < k; i++) {
            distances[probe_idx * k + i] = s_dists[i];
            indices[probe_idx * k + i] = s_indices[i];
        }
    }
}

// 选择全局TopK近邻
__global__ void merge_topk_kernel(
    const float* probe_distances,
    const int* probe_indices,
    const int* rev_id_map,
    float* final_distances,
    int* final_indices,
    size_t nprobe,
    size_t k
) {
    // 单线程块实现
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // 初始化
        for (int i = 0; i < k; i++) {
            final_distances[i] = FLT_MAX;
            final_indices[i] = -1;
        }
        
        // 合并所有probe结果
        for (int p = 0; p < nprobe; p++) {
            for (int i = 0; i < k; i++) {
                float dist = probe_distances[p * k + i];
                int idx = probe_indices[p * k + i];
                
                if (idx >= 0) {
                    // 原始ID
                    int orig_id = rev_id_map[idx];
                    
                    // 尝试插入到最终TopK
                    for (int j = 0; j < k; j++) {
                        if (dist < final_distances[j]) {
                            // 向后移动元素
                            for (int l = k-1; l > j; l--) {
                                final_distances[l] = final_distances[l-1];
                                final_indices[l] = final_indices[l-1];
                            }
                            final_distances[j] = dist;
                            final_indices[j] = orig_id;
                            break;
                        }
                    }
                }
            }
        }
    }
}

// 构造函数
IVF_GPU::IVF_GPU(size_t dim, size_t n_clusters)
    : dim(dim), n_clusters(n_clusters), initialized(false), data_count(0),
      d_centroids(nullptr), d_rearranged_data(nullptr), d_rev_id_map(nullptr),
      d_cluster_offsets(nullptr), d_cluster_sizes(nullptr) {
    
    // 创建cuBLAS句柄
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
}

// 析构函数
IVF_GPU::~IVF_GPU() {
    // 释放GPU内存
    if (d_centroids) cudaFree(d_centroids);
    if (d_rearranged_data) cudaFree(d_rearranged_data);
    if (d_rev_id_map) cudaFree(d_rev_id_map);
    if (d_cluster_offsets) cudaFree(d_cluster_offsets);
    if (d_cluster_sizes) cudaFree(d_cluster_sizes);
    
    // 销毁cuBLAS句柄
    if (cublas_handle) cublasDestroy(cublas_handle);
}

// 从CPU版本的IVF索引加载
void IVF_GPU::loadFromCPU(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Could not open file for reading: " << filename << std::endl;
        return;
    }
    
    // 读取基本参数
    size_t file_dim, file_n_clusters;
    bool use_rearranged;
    size_t file_data_count;
    
    in.read((char*)&file_dim, sizeof(file_dim));
    in.read((char*)&file_n_clusters, sizeof(file_n_clusters));
    in.read((char*)&use_rearranged, sizeof(use_rearranged));
    in.read((char*)&file_data_count, sizeof(file_data_count));
    
    // 检查参数是否匹配
    if (file_dim != dim || file_n_clusters != n_clusters) {
        std::cerr << "Error: Index parameters do not match." << std::endl;
        in.close();
        return;
    }
    
    // 读取聚类中心
    float* h_centroids = new float[dim * n_clusters];
    in.read((char*)h_centroids, dim * n_clusters * sizeof(float));
    
    // 跳过倒排列表信息
    for (size_t i = 0; i < n_clusters; i++) {
        size_t list_size;
        in.read((char*)&list_size, sizeof(list_size));
        if (list_size > 0) {
            in.seekg(list_size * sizeof(int), std::ios::cur);
        }
    }
    
    // 检查是否使用重排后的数据
    if (!use_rearranged) {
        std::cerr << "Error: CPU index does not use rearranged data, which is required for GPU acceleration." << std::endl;
        delete[] h_centroids;
        in.close();
        return;
    }
    
    // 读取重排后的数据
    data_count = file_data_count;
    float* h_rearranged_data = new float[data_count * dim];
    in.read((char*)h_rearranged_data, data_count * dim * sizeof(float));
    
    // 读取簇偏移量和大小
    std::vector<size_t> h_cluster_offsets(n_clusters);
    std::vector<size_t> h_cluster_sizes(n_clusters);
    in.read((char*)h_cluster_offsets.data(), n_clusters * sizeof(size_t));
    in.read((char*)h_cluster_sizes.data(), n_clusters * sizeof(size_t));
    
    // 读取ID映射
    std::vector<int> h_rev_id_map(data_count);
    in.read((char*)h_rev_id_map.data(), data_count * sizeof(int));
    
    in.close();
    
    // 将数据传输到GPU
    CUDA_CHECK(cudaMalloc(&d_centroids, dim * n_clusters * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rearranged_data, data_count * dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rev_id_map, data_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cluster_offsets, n_clusters * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_cluster_sizes, n_clusters * sizeof(size_t)));
    
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids, dim * n_clusters * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rearranged_data, h_rearranged_data, data_count * dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rev_id_map, h_rev_id_map.data(), data_count * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cluster_offsets, h_cluster_offsets.data(), n_clusters * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cluster_sizes, h_cluster_sizes.data(), n_clusters * sizeof(size_t), cudaMemcpyHostToDevice));
    
    // 清理
    delete[] h_centroids;
    delete[] h_rearranged_data;
    
    initialized = true;
    std::cout << "Successfully loaded IVF index to GPU." << std::endl;
}

// 在GPU上找到最近的nprobe个簇
void IVF_GPU::findNearestClusters(
    const float* query,
    int* cluster_ids,
    float* cluster_dists,
    size_t nprobe
) {
    if (!initialized) {
        std::cerr << "Error: IVF_GPU not initialized" << std::endl;
        return;
    }
    
    // 分配GPU内存
    float* d_query;
    float* d_distances;
    CUDA_CHECK(cudaMalloc(&d_query, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_distances, n_clusters * sizeof(float)));
    
    // 复制查询向量到GPU
    CUDA_CHECK(cudaMemcpy(d_query, query, dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // 启动核函数计算距离
    int block_size = 256;
    int grid_size = (n_clusters + block_size - 1) / block_size;
    compute_cluster_distances_kernel<<<grid_size, block_size>>>(
        d_centroids, d_query, d_distances, n_clusters, dim);
    CUDA_CHECK(cudaGetLastError());
    
    // 复制距离回主机
    std::vector<float> h_distances(n_clusters);
    CUDA_CHECK(cudaMemcpy(h_distances.data(), d_distances, n_clusters * sizeof(float), cudaMemcpyDeviceToHost));
    
    // 在主机上找出最近的nprobe个簇
    std::vector<std::pair<float, int>> cluster_distances;
    cluster_distances.reserve(n_clusters);
    for (size_t i = 0; i < n_clusters; i++) {
        cluster_distances.push_back({h_distances[i], static_cast<int>(i)});
    }
    
    // 部分排序，找出前nprobe个
    std::partial_sort(
        cluster_distances.begin(),
        cluster_distances.begin() + nprobe,
        cluster_distances.end()
    );
    
    // 提取结果
    for (size_t i = 0; i < nprobe; i++) {
        cluster_ids[i] = cluster_distances[i].second;
        cluster_dists[i] = cluster_distances[i].first;
    }
    
    // 释放资源
    cudaFree(d_query);
    cudaFree(d_distances);
}

// 在GPU上搜索最近邻
std::priority_queue<std::pair<float, uint32_t>> IVF_GPU::search(
    const float* query,
    size_t k,
    size_t nprobe
) {
    if (!initialized) {
        std::cerr << "Error: IVF_GPU not initialized" << std::endl;
        return std::priority_queue<std::pair<float, uint32_t>>();
    }
    
    nprobe = std::min(nprobe, n_clusters);
    std::priority_queue<std::pair<float, uint32_t>> results;
    
    // 步骤1：找到最近的nprobe个簇
    std::vector<int> h_cluster_ids(nprobe);
    std::vector<float> h_cluster_dists(nprobe);
    
    findNearestClusters(query, h_cluster_ids.data(), h_cluster_dists.data(), nprobe);
    
    // 步骤2：分配GPU内存
    float* d_query;
    int* d_cluster_ids;
    float* d_probe_distances;
    int* d_probe_indices;
    float* d_final_distances;
    int* d_final_indices;
    
    CUDA_CHECK(cudaMalloc(&d_query, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cluster_ids, nprobe * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_probe_distances, nprobe * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probe_indices, nprobe * k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_final_distances, k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_final_indices, k * sizeof(int)));
    
    // 复制数据到GPU
    CUDA_CHECK(cudaMemcpy(d_query, query, dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cluster_ids, h_cluster_ids.data(), nprobe * sizeof(int), cudaMemcpyHostToDevice));
    
    // 步骤3：在选定的簇中搜索
    int threads_per_block = 256;
    size_t shared_mem_size = k * (sizeof(float) + sizeof(int));
    
    search_cluster_vectors_kernel<<<nprobe, threads_per_block, shared_mem_size>>>(
        d_rearranged_data,
        d_query,
        d_cluster_ids,
        d_cluster_offsets,
        d_cluster_sizes,
        d_probe_distances,
        d_probe_indices,
        dim,
        nprobe,
        k
    );
    CUDA_CHECK(cudaGetLastError());
    
    // 步骤4：合并来自不同簇的结果
    merge_topk_kernel<<<1, 1>>>(
        d_probe_distances,
        d_probe_indices,
        d_rev_id_map,
        d_final_distances,
        d_final_indices,
        nprobe,
        k
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 步骤5：复制结果回主机
    std::vector<float> h_distances(k);
    std::vector<int> h_indices(k);
    
    CUDA_CHECK(cudaMemcpy(h_distances.data(), d_final_distances, k * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_indices.data(), d_final_indices, k * sizeof(int), cudaMemcpyDeviceToHost));
    
    // 构建优先队列结果
    for (size_t i = 0; i < k; i++) {
        if (h_indices[i] >= 0) {
            results.push({h_distances[i], static_cast<uint32_t>(h_indices[i])});
        }
    }
    
    // 释放资源
    cudaFree(d_query);
    cudaFree(d_cluster_ids);
    cudaFree(d_probe_distances);
    cudaFree(d_probe_indices);
    cudaFree(d_final_distances);
    cudaFree(d_final_indices);
    
    return results;
}

// 批量查询
std::vector<std::priority_queue<std::pair<float, uint32_t>>> IVF_GPU::batch_search(
    const float* queries,
    size_t query_num,
    size_t k,
    size_t nprobe,
    int batch_size
) {
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(query_num);
    
    // 分批处理查询
    for (size_t i = 0; i < query_num; i += batch_size) {
        size_t current_batch = std::min(batch_size, static_cast<int>(query_num - i));
        
        // 处理每个批次的每个查询
        for (size_t j = 0; j < current_batch; j++) {
            const float* query = queries + (i + j) * dim;
            results[i + j] = search(query, k, nprobe);
        }
    }
    
    return results;
}

// 单例模式的外部接口
std::priority_queue<std::pair<float, uint32_t>> ivf_gpu_search(
    const float* query,
    size_t vecdim,
    size_t k,
    size_t nprobe
) {
    static IVF_GPU* ivf_gpu = nullptr;
    static bool initialized = false;
    
    // 首次调用时初始化
    if (!initialized) {
        std::string cpu_index_path = "files/ivf.index";
        
        ivf_gpu = new IVF_GPU(vecdim);
        ivf_gpu->loadFromCPU(cpu_index_path);
        
        initialized = true;
    }
    
    return ivf_gpu->search(query, k, nprobe);
}

// 批量查询外部接口
std::vector<std::priority_queue<std::pair<float, uint32_t>>> ivf_gpu_batch_search(
    const float* queries,
    size_t query_num,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    int batch_size
) {
    static IVF_GPU* ivf_gpu = nullptr;
    static bool initialized = false;
    
    // 首次调用时初始化
    if (!initialized) {
        std::string cpu_index_path = "files/ivf.index";
        
        ivf_gpu = new IVF_GPU(vecdim);
        ivf_gpu->loadFromCPU(cpu_index_path);
        
        initialized = true;
    }
    
    return ivf_gpu->batch_search(queries, query_num, k, nprobe, batch_size);
}

// 添加到现有函数之后

// 计算两个查询向量的聚类中心重合度
int compute_centroid_overlap(const int* clusters1, const int* clusters2, size_t nprobe) {
    int overlap = 0;
    for (size_t i = 0; i < nprobe; i++) {
        for (size_t j = 0; j < nprobe; j++) {
            if (clusters1[i] == clusters2[j]) {
                overlap++;
                break; // 找到一个匹配就跳出内层循环
            }
        }
    }
    return overlap;
}

// 批量查询-聚类中心相似度分组策略
std::vector<std::priority_queue<std::pair<float, uint32_t>>> IVF_GPU::batch_search_cluster_similarity(
    const float* queries,
    size_t query_num,
    size_t k,
    size_t nprobe,
    int batch_size
) {
    if (!initialized) {
        std::cerr << "Error: IVF_GPU not initialized" << std::endl;
        return std::vector<std::priority_queue<std::pair<float, uint32_t>>>(query_num);
    }
    
    nprobe = std::min(nprobe, n_clusters);
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(query_num);
    
    // 1. 预先计算所有查询向量的最近簇
    std::vector<std::vector<int>> all_nearest_clusters(query_num);
    std::vector<std::vector<float>> all_cluster_dists(query_num);
    
    // 为每个查询找到最近的nprobe个簇
    #pragma omp parallel for
    for (size_t i = 0; i < query_num; i++) {
        std::vector<int> cluster_ids(nprobe);
        std::vector<float> cluster_dists(nprobe);
        findNearestClusters(queries + i * dim, cluster_ids.data(), cluster_dists.data(), nprobe);
        all_nearest_clusters[i] = cluster_ids;
        all_cluster_dists[i] = cluster_dists;
    }
    
    // 2. 使用聚类中心相似度进行查询分组
    std::vector<bool> processed(query_num, false);
    std::vector<std::vector<size_t>> batches;
    
    while (std::find(processed.begin(), processed.end(), false) != processed.end()) {
        // 找到第一个未处理的查询
        size_t seed_idx = 0;
        for (; seed_idx < query_num; seed_idx++) {
            if (!processed[seed_idx]) break;
        }
        
        // 创建新批次，以该查询为种子
        std::vector<size_t> batch = {seed_idx};
        processed[seed_idx] = true;
        
        // 计算与其他查询的簇重合度
        std::vector<std::pair<size_t, int>> similarity_scores;
        for (size_t i = 0; i < query_num; i++) {
            if (!processed[i]) {
                // 计算簇重合数量
                int overlap = compute_centroid_overlap(
                    all_nearest_clusters[seed_idx].data(),
                    all_nearest_clusters[i].data(),
                    nprobe
                );
                similarity_scores.push_back(std::make_pair(i, overlap));
            }
        }
        
        // 按重合度排序（从高到低）
        std::sort(similarity_scores.begin(), similarity_scores.end(),
            [](const std::pair<size_t, int>& a, const std::pair<size_t, int>& b) { 
                return a.second > b.second; 
            });
        
        // 添加最相似的查询到批次
        for (size_t i = 0; i < similarity_scores.size(); i++) {
            if (batch.size() >= batch_size) break;
            
            batch.push_back(similarity_scores[i].first);
            processed[similarity_scores[i].first] = true;
        }
        
        batches.push_back(batch);
    }
    
    // 3. 对每个批次进行处理
    for (size_t batch_idx = 0; batch_idx < batches.size(); batch_idx++) {
        const std::vector<size_t>& batch = batches[batch_idx];
        int current_batch_size = batch.size();
        
        // 统计此批次的簇使用频率
        // 使用普通数组而不是std::map
        int* cluster_freq = new int[n_clusters];
        memset(cluster_freq, 0, n_clusters * sizeof(int));
        
        for (size_t i = 0; i < batch.size(); i++) {
            size_t idx = batch[i];
            for (size_t j = 0; j < all_nearest_clusters[idx].size(); j++) {
                int cluster_id = all_nearest_clusters[idx][j];
                cluster_freq[cluster_id]++;
            }
        }
        
        // 找出此批次中最常用的簇
        std::vector<std::pair<int, int>> sorted_clusters;
        for (size_t i = 0; i < n_clusters; i++) {
            if (cluster_freq[i] > 0) {
                sorted_clusters.push_back(std::make_pair((int)i, cluster_freq[i]));
            }
        }
        
        std::sort(sorted_clusters.begin(), sorted_clusters.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) { 
                return a.second > b.second; 
            });
        
        delete[] cluster_freq;
        
        // 确定要搜索的簇数量（可能超过nprobe，以覆盖所有查询需要的簇）
        size_t search_clusters = std::min(sorted_clusters.size(), nprobe * 2UL); // 最多搜索2倍的nprobe簇
        std::vector<int> common_clusters(search_clusters);
        for (size_t i = 0; i < search_clusters; i++) {
            common_clusters[i] = sorted_clusters[i].first;
        }
        
        // 为每个查询在共同簇中搜索
        for (size_t i = 0; i < batch.size(); i++) {
            size_t query_idx = batch[i];
            const float* query = queries + query_idx * dim;
            
            // 使用优化的方法在共同簇中搜索
            results[query_idx] = search_in_specific_clusters(query, k, common_clusters);
        }
    }
    
    return results;
}

// 在指定簇中搜索
std::priority_queue<std::pair<float, uint32_t>> IVF_GPU::search_in_specific_clusters(
    const float* query,
    size_t k,
    const std::vector<int>& cluster_ids
) {
    size_t nprobe = cluster_ids.size();
    std::priority_queue<std::pair<float, uint32_t>> results;
    
    // 步骤1：分配GPU内存
    float* d_query;
    int* d_cluster_ids;
    float* d_probe_distances;
    int* d_probe_indices;
    float* d_final_distances;
    int* d_final_indices;
    
    CUDA_CHECK(cudaMalloc(&d_query, dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cluster_ids, nprobe * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_probe_distances, nprobe * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_probe_indices, nprobe * k * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_final_distances, k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_final_indices, k * sizeof(int)));
    
    // 复制数据到GPU
    CUDA_CHECK(cudaMemcpy(d_query, query, dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cluster_ids, cluster_ids.data(), nprobe * sizeof(int), cudaMemcpyHostToDevice));
    
    // 步骤2：在选定的簇中搜索
    int threads_per_block = 256;
    size_t shared_mem_size = k * (sizeof(float) + sizeof(int));
    
    search_cluster_vectors_kernel<<<nprobe, threads_per_block, shared_mem_size>>>(
        d_rearranged_data,
        d_query,
        d_cluster_ids,
        d_cluster_offsets,
        d_cluster_sizes,
        d_probe_distances,
        d_probe_indices,
        dim,
        nprobe,
        k
    );
    CUDA_CHECK(cudaGetLastError());
    
    // 步骤3：合并来自不同簇的结果
    merge_topk_kernel<<<1, 1>>>(
        d_probe_distances,
        d_probe_indices,
        d_rev_id_map,
        d_final_distances,
        d_final_indices,
        nprobe,
        k
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 步骤4：复制结果回主机
    std::vector<float> h_distances(k);
    std::vector<int> h_indices(k);
    
    CUDA_CHECK(cudaMemcpy(h_distances.data(), d_final_distances, k * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_indices.data(), d_final_indices, k * sizeof(int), cudaMemcpyDeviceToHost));
    
    // 构建优先队列结果
    for (size_t i = 0; i < k; i++) {
        if (h_indices[i] >= 0) {
            results.push({h_distances[i], static_cast<uint32_t>(h_indices[i])});
        }
    }
    
    // 释放资源
    cudaFree(d_query);
    cudaFree(d_cluster_ids);
    cudaFree(d_probe_distances);
    cudaFree(d_probe_indices);
    cudaFree(d_final_distances);
    cudaFree(d_final_indices);
    
    return results;
}

// 修改批量查询外部接口，使用聚类中心相似度分组策略
std::vector<std::priority_queue<std::pair<float, uint32_t>>> ivf_gpu_batch_search_cluster_similarity(
    const float* queries,
    size_t query_num,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    int batch_size
) {
    static IVF_GPU* ivf_gpu = nullptr;
    static bool initialized = false;
    
    // 首次调用时初始化
    if (!initialized) {
        std::string cpu_index_path = "files/ivf.index";
        
        ivf_gpu = new IVF_GPU(vecdim);
        ivf_gpu->loadFromCPU(cpu_index_path);
        
        initialized = true;
    }
    
    return ivf_gpu->batch_search_cluster_similarity(queries, query_num, k, nprobe, batch_size);
}
