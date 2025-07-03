#include "gpu_search.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cfloat>  // 用于FLT_MAX

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

// 计算点积距离的CUDA核函数
__global__ void compute_distances_kernel(float* base, float* query, float* distances, 
                                        size_t base_number, size_t vecdim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < base_number) {
        float dot_product = 0.0f;
        for (int d = 0; d < vecdim; ++d) {
            dot_product += base[idx * vecdim + d] * query[d];
        }
        // DEEP100K使用IP距离: 1 - 内积
        distances[idx] = 1.0f - dot_product;
    }
}

// 添加1-内积转换的核函数
__global__ void convert_dot_to_distance_kernel(float* dot_products, size_t total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        dot_products[idx] = 1.0f - dot_products[idx]; // 将内积转为距离
    }
}

// 找出k个最近邻的CUDA核函数
__global__ void find_topk_kernel(float* distances, int* indices, size_t base_number, size_t k) {
    // 共享内存用于存储当前最近的k个点
    extern __shared__ float shared_mem[];
    float* shared_dists = shared_mem;
    int* shared_indices = (int*)(shared_dists + k);
    
    // 初始化为最大值
    if (threadIdx.x < k) {
        shared_dists[threadIdx.x] = FLT_MAX;
        shared_indices[threadIdx.x] = -1;
    }
    __syncthreads();
    
    // 每个线程处理数据库中的一部分向量
    for (int i = threadIdx.x; i < base_number; i += blockDim.x) {
        float dist = distances[i];
        
        // 尝试将当前向量插入到TopK中
        for (int j = 0; j < k; j++) {
            if (dist < shared_dists[j]) {
                // 向后移动元素为新元素腾出位置
                for (int l = k - 1; l > j; l--) {
                    shared_dists[l] = shared_dists[l-1];
                    shared_indices[l] = shared_indices[l-1];
                }
                shared_dists[j] = dist;
                shared_indices[j] = i;
                break;
            }
        }
        __syncthreads();
    }
    
    // 线程0负责将结果写回全局内存
    if (threadIdx.x == 0) {
        for (int i = 0; i < k; i++) {
            distances[i] = shared_dists[i];
            indices[i] = shared_indices[i];
        }
    }
}

// 批量查询的TopK核函数
__global__ void batch_find_topk_kernel(float* distances, int* indices, 
                                      size_t base_number, size_t query_num, size_t k) {
    // 每个线程块处理一个查询
    int query_idx = blockIdx.x;
    if (query_idx >= query_num) return;
    
    // 每个线程块使用共享内存来存储当前查询的TopK结果
    extern __shared__ float shared_mem[];
    float* shared_dists = shared_mem;
    int* shared_indices = (int*)(shared_dists + k);
    
    // 初始化共享内存
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        shared_dists[i] = FLT_MAX;
        shared_indices[i] = -1;
    }
    __syncthreads();
    
    // 所有线程共同处理一个查询的所有向量
    for (int i = threadIdx.x; i < base_number; i += blockDim.x) {
        // 获取当前查询到当前向量的距离
        float dist = distances[query_idx * base_number + i];
        
        // 插入排序，放入合适的位置
        for (int j = 0; j < k; j++) {
            if (dist < shared_dists[j]) {
                // 需要插入，进行元素移动
                for (int l = k - 1; l > j; l--) {
                    shared_dists[l] = shared_dists[l-1];
                    shared_indices[l] = shared_indices[l-1];
                }
                shared_dists[j] = dist;
                shared_indices[j] = i;
                break;
            }
        }
        __syncthreads(); // 确保所有线程完成一轮处理
    }
    
    // 将结果写回全局内存，结果布局为每个查询连续存储其k个最近邻
    if (threadIdx.x < k) {
        int idx = query_idx * k + threadIdx.x;
        distances[idx] = shared_dists[threadIdx.x];
        indices[idx] = shared_indices[threadIdx.x];
    }
}

// 实现高效的批处理版本
std::vector<std::priority_queue<std::pair<float, uint32_t>>> 
gpu_batch_search(float* base, float* queries, size_t base_number, size_t vecdim, 
                size_t query_num, size_t k, int batch_size, int block_size) {
    
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results(query_num);
    
    // 创建cuBLAS句柄
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // 为批处理分配GPU内存
    float *d_base = nullptr;
    float *d_queries = nullptr;
    float *d_distances = nullptr;
    int *d_indices = nullptr;
    
    // 为所有数据分配GPU内存
    CUDA_CHECK(cudaMalloc(&d_base, base_number * vecdim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_base, base, base_number * vecdim * sizeof(float), cudaMemcpyHostToDevice));
    
    // 批处理参数
    int num_batches = (query_num + batch_size - 1) / batch_size;
    
    // 为当前批次分配内存 - 修复类型不匹配问题
    int max_batch = (query_num < static_cast<size_t>(batch_size)) ? 
                     static_cast<int>(query_num) : batch_size;
    
    CUDA_CHECK(cudaMalloc(&d_queries, max_batch * vecdim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_distances, base_number * max_batch * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_indices, max_batch * k * sizeof(int)));
    
    // 为结果分配主机内存
    float *h_distances = new float[max_batch * k];
    int *h_indices = new int[max_batch * k];
    
    // 处理每个批次
    for (int b = 0; b < num_batches; b++) {
        // 计算当前批次的实际大小 - 修复类型不匹配问题
        int current_batch_size = (b * batch_size + batch_size <= query_num) ? 
                                batch_size : static_cast<int>(query_num - b * batch_size);
        
        // 复制当前批次的查询到GPU
        CUDA_CHECK(cudaMemcpy(d_queries, 
                            queries + (b * batch_size) * vecdim, 
                            current_batch_size * vecdim * sizeof(float), 
                            cudaMemcpyHostToDevice));
        
        // 使用cuBLAS计算距离矩阵
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // 计算内积矩阵
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                              base_number, current_batch_size, vecdim,
                              &alpha, d_base, vecdim, 
                              d_queries, vecdim,
                              &beta, d_distances, base_number));
        
        // 将内积转换为距离
        dim3 block(block_size);
        dim3 grid((base_number * current_batch_size + block.x - 1) / block.x);
        convert_dot_to_distance_kernel<<<grid, block>>>(d_distances, base_number * current_batch_size);
        CUDA_CHECK(cudaGetLastError());
        
        // 为每个查询找出TopK近邻
        int shared_mem_size = k * (sizeof(float) + sizeof(int));
        batch_find_topk_kernel<<<current_batch_size, block_size, shared_mem_size>>>(
            d_distances, d_indices, base_number, current_batch_size, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 复制结果回主机
        CUDA_CHECK(cudaMemcpy(h_distances, d_distances, current_batch_size * k * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_indices, d_indices, current_batch_size * k * sizeof(int), cudaMemcpyDeviceToHost));
        
        // 构建优先队列结果
        for (int i = 0; i < current_batch_size; i++) {
            int query_idx = b * batch_size + i;
            std::priority_queue<std::pair<float, uint32_t>> pq;
            
            for (int j = 0; j < k; j++) {
                int idx = i * k + j;
                if (h_indices[idx] >= 0) {
                    pq.push({h_distances[idx], (uint32_t)h_indices[idx]});
                }
            }
            
            results[query_idx] = std::move(pq);
        }
    }
    
    // 释放资源
    delete[] h_distances;
    delete[] h_indices;
    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_queries));
    CUDA_CHECK(cudaFree(d_distances));
    CUDA_CHECK(cudaFree(d_indices));
    CUBLAS_CHECK(cublasDestroy(handle));
    
    return results;
}

// 保留旧接口作为批处理的简化版
std::priority_queue<std::pair<float, uint32_t>> 
gpu_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    auto results = gpu_batch_search(base, query, base_number, vecdim, 1, k, 1);
    return results[0];
}