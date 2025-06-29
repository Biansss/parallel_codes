#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include <mpi.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
// 条件编译：只在ARM架构下包含ARM特定头文件
#ifdef __aarch64__
#include <arm_neon.h>
#include <arm_acle.h>
#elif defined(__x86_64__)
#include <immintrin.h>
#endif
#include "IVF-MPI.h"
#include "IVF-PQ-MPI.h"
#include "IVF-HNSW-MPI.h"
#include "IVF-HNSW-MPI-1.h"
using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

// 计算单个查询的召回率
float calculate_recall(const std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, std::greater<std::pair<float, int>>>& result,
                      const int* gt, size_t k, size_t gt_start_idx) {
    std::set<uint32_t> gtset;
    for(size_t j = 0; j < k; ++j){
        int t = gt[j + gt_start_idx];
        gtset.insert(t);
    }

    size_t acc = 0;
    auto temp_result = result; // 复制结果队列
    while (!temp_result.empty()) {   
        int x = temp_result.top().second;
        if(gtset.find(x) != gtset.end()){
            ++acc;
        }
        temp_result.pop();
    }
    
    return (float)acc / k;
}

int main(int argc, char *argv[])
{
    // 初始化MPI环境
    MPI_Init(&argc, &argv);
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    // 只在0号进程上输出进程信息
    if (world_rank == 0) {
        std::cout << "Running with " << world_size << " MPI processes" << std::endl;
    }

    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;
    float* test_query = nullptr;
    int* test_gt = nullptr;
    float* base = nullptr;

    // 所有进程都需要加载数据
    std::string data_path = "/anndata/"; 
    test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    // 检查是否加载成功
    if(test_query == nullptr || test_gt == nullptr || base == nullptr) {
        if (world_rank == 0) {
            std::cerr << "load data failed\n";
        }
        MPI_Finalize();
        return -1;
    }
    
    // 只测试前2000条查询
    test_number = std::min(test_number, (size_t)2000);
    
    const size_t k = 10;
    
    // 所有进程都需要处理所有查询（MPI并行在IVF内部实现）
    std::vector<SearchResult> results;
    results.resize(test_number);

    // 只在0号进程上执行查询和计算准确率
    if (world_rank == 0) {
        for(size_t i = 0; i < test_number; ++i) {
            const unsigned long Converter = 1000 * 1000;
            struct timeval val;
            int ret = gettimeofday(&val, NULL);

            // 使用MPI版本的IVF搜索（内部已经实现MPI并行）
            auto res = ivf_search_mpi(base, test_query + i*vecdim, base_number, vecdim, k);
            //auto res = ivfpq_search_mpi(base, test_query + i*vecdim, base_number, vecdim, k, 16, true);
            //auto res = ivfhnsw_search_mpi(base, test_query + i*vecdim, base_number, vecdim, k, 16, true);
            //auto res = hnsw_mpi_distributed_search(base, test_query + i*vecdim, base_number, vecdim, k, 16);
            struct timeval newVal;
            ret = gettimeofday(&newVal, NULL);
            int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

            // 计算召回率
            float recall = calculate_recall(res, test_gt, k, i * test_gt_d);

            results[i] = {recall, diff};
            
            // 每100个查询输出一次进度
            if ((i + 1) % 100 == 0) {
                std::cout << "Processed " << (i + 1) << "/" << test_number << " queries" << std::endl;
            }
        }
        
        // 计算并输出平均结果
        float avg_recall = 0, avg_latency = 0;
        for(size_t i = 0; i < test_number; ++i) {
            avg_recall += results[i].recall;
            avg_latency += results[i].latency;
        }

        std::cout << "average recall: " << avg_recall / test_number << "\n";
        std::cout << "average latency (us): " << avg_latency / test_number << "\n";
    } else {
        // 其他进程需要参与IVF搜索的MPI并行计算
        // 但不需要执行查询循环，只需要等待MPI通信
        for(size_t i = 0; i < test_number; ++i) {
            // 调用搜索函数参与MPI计算，但不使用返回结果
            ivf_search_mpi(base, test_query + i*vecdim, base_number, vecdim, k);
            //ivfpq_search_mpi(base, test_query + i*vecdim, base_number, vecdim, k, 16, true);
            //ivfhnsw_search_mpi(base, test_query + i*vecdim, base_number, vecdim, k, 16, true);
            //hnsw_mpi_distributed_search(base, test_query + i*vecdim, base_number, vecdim, k, 16);
            
        }
    }

    // 清理内存
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    
    // 终止MPI环境
    MPI_Finalize();
    return 0;
}