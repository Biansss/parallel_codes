#pragma once
#include <vector>
#include <queue>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>

class IVF_GPU {
public:
    // 构造函数
    IVF_GPU(size_t dim, size_t n_clusters = 1024);
    
    // 析构函数
    ~IVF_GPU();
    
    // 从CPU版本的IVF索引加载
    void loadFromCPU(const std::string& filename);
    
    // 直接加载已保存的GPU索引
    void loadIndex(const std::string& filename);
    
    // 保存GPU索引
    void saveIndex(const std::string& filename);
    
    // 在GPU上搜索最近邻
    std::priority_queue<std::pair<float, uint32_t>> search(
        const float* query, 
        size_t k, 
        size_t nprobe = 10
    );
    
    // 批量查询
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> batch_search(
        const float* queries,
        size_t query_num,
        size_t k,
        size_t nprobe = 10,
        int batch_size = 64
    );
    
    // 基于聚类中心相似度的批处理查询
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> batch_search_cluster_similarity(
        const float* queries,
        size_t query_num,
        size_t k,
        size_t nprobe,
        int batch_size
    );

    // 在特定簇中搜索
    std::priority_queue<std::pair<float, uint32_t>> search_in_specific_clusters(
        const float* query,
        size_t k,
        const std::vector<int>& cluster_ids
    );

private:
    // 在GPU上找到最近的簇
    void findNearestClusters(
        const float* query,
        int* cluster_ids,
        float* cluster_dists,
        size_t nprobe
    );
    
    // 在GPU上批量查找最近的簇
    void batchFindNearestClusters(
        const float* queries,
        int* cluster_ids,
        float* cluster_dists,
        size_t query_num,
        size_t nprobe
    );

    size_t dim;                  // 向量维度
    size_t n_clusters;           // 聚类数量
    bool initialized;            // 是否已初始化
    size_t data_count;           // 数据总数
    
    // GPU内存指针
    float* d_centroids;          // 聚类中心
    float* d_rearranged_data;    // 重排后的数据库
    int* d_rev_id_map;           // ID映射
    size_t* d_cluster_offsets;   // 簇偏移量
    size_t* d_cluster_sizes;     // 簇大小
    
    // cuBLAS句柄
    cublasHandle_t cublas_handle;
};

// 外部接口函数
std::priority_queue<std::pair<float, uint32_t>> ivf_gpu_search(
    const float* query,
    size_t vecdim,
    size_t k,
    size_t nprobe = 16
);

std::vector<std::priority_queue<std::pair<float, uint32_t>>> ivf_gpu_batch_search(
    const float* queries,
    size_t query_num,
    size_t vecdim,
    size_t k,
    size_t nprobe = 16,
    int batch_size = 64
);

// 使用聚类中心相似度分组策略的批处理查询接口
std::vector<std::priority_queue<std::pair<float, uint32_t>>> ivf_gpu_batch_search_cluster_similarity(
    const float* queries,
    size_t query_num,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    int batch_size
);
