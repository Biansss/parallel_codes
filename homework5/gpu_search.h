#pragma once
#include <queue>
#include <vector>
#include <cstdint>

// GPU加速的ANN搜索函数声明（实现在.cu文件中）
std::priority_queue<std::pair<float, uint32_t>> gpu_search(
    float* base, 
    float* query, 
    size_t base_number, 
    size_t vecdim, 
    size_t k
);

// 批量查询版本（可选使用）
std::vector<std::priority_queue<std::pair<float, uint32_t>>> gpu_batch_search(
    float* base, 
    float* queries, 
    size_t base_number, 
    size_t vecdim,
    size_t query_num, 
    size_t k, 
    int batch_size = 64,
    int block_size = 256  // 新增线程块大小参数
);