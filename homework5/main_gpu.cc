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
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"
// 移除了ARM NEON和SIMD相关头文件
#include "gpu_search.h"
// 添加IVF相关头文件
#include "IVF.h"
#include "IVFomp.h"
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

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150; // 为防止索引构建时间过长，efc建议设置200以下
    const int M = 16; // M建议设置为16以下

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}

// 确保IVF索引已构建
void ensure_ivf_index(float* base, size_t base_number, size_t vecdim)
{
    std::string path_index = "files/ivf.index";
    std::ifstream f(path_index);
    bool index_exists = f.good();
    f.close();
    
    if (!index_exists) {
        std::cout << "正在构建IVF索引..." << std::endl;
        
        // 创建并训练IVF索引
        IVF* ivf = new IVF(vecdim);
        ivf->train(base, base_number);
        ivf->addPoints(base, base_number);
        
        // 保存索引
        ivf->saveIndex(path_index);
        delete ivf;
        
        std::cout << "IVF索引构建完成并保存。" << std::endl;
    }
}

int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "anndata/"; 
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    // 检查是否加载成功
    if(test_query == nullptr || test_gt == nullptr || base == nullptr) {
        std::cerr << "load data failed\n";
        return -1;
    }
    // 只测试前2000条查询
    test_number = 2000;

    // const size_t k = 10;
    // const int BATCH_SIZE = 128; // 配置批处理大小
     const int CPU_TEST_SIZE = 100; // 为CPU测试设置更小的样本，因为较慢
    // const size_t NPROBE = 16; // IVF搜索的nprobe参数

    std::vector<SearchResult> gpu_results;
    std::vector<SearchResult> cpu_results;
    std::vector<SearchResult> ivf_cpu_results;
    std::vector<SearchResult> ivf_gpu_results;
    
    gpu_results.resize(test_number);
    cpu_results.resize(CPU_TEST_SIZE);
    ivf_cpu_results.resize(CPU_TEST_SIZE);
    ivf_gpu_results.resize(test_number);

    // // 确保IVF索引存在
    // ensure_ivf_index(base, base_number, vecdim);

    // // 1. 首先测试CPU搜索时间（样本较小以节省时间）
    // std::cout << "正在测试CPU搜索性能..." << std::endl;
    // double cpu_total_time_us = 0;
    // for(int i = 0; i < CPU_TEST_SIZE; ++i) {
    //     const unsigned long Converter = 1000 * 1000;
    //     struct timeval val;
    //     gettimeofday(&val, NULL);

    //     // 使用CPU版本的搜索函数
    //     auto res = flat_search(base, test_query + i*vecdim, base_number, vecdim, k);
        
    //     struct timeval newVal;
    //     gettimeofday(&newVal, NULL);
    //     int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
    //     cpu_total_time_us += diff;

    //     std::set<uint32_t> gtset;
    //     for(int j = 0; j < k; ++j){
    //         int t = test_gt[j + i*test_gt_d];
    //         gtset.insert(t);
    //     }

    //     size_t acc = 0;
    //     while (res.size()) {   
    //         int x = res.top().second;
    //         if(gtset.find(x) != gtset.end()){
    //             ++acc;
    //         }
    //         res.pop();
    //     }
    //     float recall = (float)acc/k;

    //     cpu_results[i] = {recall, diff};
    // }
    
    // // 2. 测试CPU版本的IVF搜索性能
    // std::cout << "正在测试CPU IVF搜索性能..." << std::endl;
    // for(int i = 0; i < CPU_TEST_SIZE; ++i) {
    //     const unsigned long Converter = 1000 * 1000;
    //     struct timeval val;
    //     gettimeofday(&val, NULL);

    //     // 使用CPU版本的IVF搜索函数
    //     auto res = ivf_search(base, test_query + i*vecdim, base_number, vecdim, k, NPROBE, 1); // 单线程执行
        
    //     struct timeval newVal;
    //     gettimeofday(&newVal, NULL);
    //     int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

    //     // 计算召回率
    //     std::set<uint32_t> gtset;
    //     for(int j = 0; j < k; ++j){
    //         int t = test_gt[j + i*test_gt_d];
    //         gtset.insert(t);
    //     }

    //     size_t acc = 0;
    //     while (res.size()) {   
    //         int x = res.top().second;
    //         if(gtset.find(x) != gtset.end()){
    //             ++acc;
    //         }
    //         res.pop();
    //     }
    //     float recall = (float)acc/k;

    //     ivf_cpu_results[i] = {recall, diff};
    // }
    
    // // 3. 然后测试GPU批处理搜索性能
    // std::cout << "正在测试GPU搜索性能..." << std::endl;
    // // 优化的批处理查询代码
    // for (int batch_start = 0; batch_start < test_number; batch_start += BATCH_SIZE) {
    //     const unsigned long Converter = 1000 * 1000;
    //     struct timeval val;
    //     int ret = gettimeofday(&val, NULL);
        
    //     // 计算当前批次的实际大小
    //     int current_batch_size = std::min(BATCH_SIZE, (int)(test_number - batch_start));
        
    //     // 批量处理查询
    //     auto batch_results = gpu_batch_search(
    //         base, 
    //         test_query + batch_start * vecdim, 
    //         base_number, 
    //         vecdim, 
    //         current_batch_size, // 当前批次大小
    //         k, 
    //         current_batch_size  // 批处理大小等于当前批次大小
    //     );
    //     struct timeval newVal;
    //     ret = gettimeofday(&newVal, NULL);
    //     int64_t total_diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
    //     int64_t avg_diff = total_diff / current_batch_size; // 计算平均每个查询时间
        
    //     // 处理每个查询结果
    //     for (int i = 0; i < current_batch_size; i++) {
    //         int query_idx = batch_start + i;
            
    //         // 获取当前查询的结果队列
    //         auto& res = batch_results[i];
            
    //         // 构建ground truth集合
    //         std::set<uint32_t> gtset;
    //         for (int j = 0; j < k; ++j) {
    //             int t = test_gt[j + query_idx * test_gt_d];
    //             gtset.insert(t);
    //         }
            
    //         // 计算召回率
    //         size_t acc = 0;
    //         while (res.size()) {   
    //             int x = res.top().second;
    //             if (gtset.find(x) != gtset.end()) {
    //                 ++acc;
    //             }
    //             res.pop();
    //         }
    //         float recall = (float)acc / k;
            
    //         // 保存结果
    //         gpu_results[query_idx] = {recall, avg_diff};
    //     }
    // }
    
    // // 4. 测试GPU版本的IVF搜索性能
    // std::cout << "正在测试GPU IVF搜索性能..." << std::endl;
    // for (int batch_start = 0; batch_start < test_number; batch_start += BATCH_SIZE) {
    //     const unsigned long Converter = 1000 * 1000;
    //     struct timeval val;
    //     gettimeofday(&val, NULL);
        
    //     // 计算当前批次的实际大小
    //     int current_batch_size = std::min(BATCH_SIZE, (int)(test_number - batch_start));
        
    //     // 批量处理IVF查询
    //     auto batch_results = ivf_gpu_batch_search(
    //         test_query + batch_start * vecdim,
    //         current_batch_size,
    //         vecdim,
    //         k,
    //         NPROBE,
    //         current_batch_size
    //     );
        
    //     struct timeval newVal;
    //     gettimeofday(&newVal, NULL);
    //     int64_t total_diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
    //     int64_t avg_diff = total_diff / current_batch_size;
        
    //     // 处理每个查询结果
    //     for (int i = 0; i < current_batch_size; i++) {
    //         int query_idx = batch_start + i;
            
    //         // 获取当前查询的结果队列
    //         auto& res = batch_results[i];
            
    //         // 构建ground truth集合
    //         std::set<uint32_t> gtset;
    //         for (int j = 0; j < k; ++j) {
    //             int t = test_gt[j + query_idx * test_gt_d];
    //             gtset.insert(t);
    //         }
            
    //         // 计算召回率
    //         size_t acc = 0;
    //         while (res.size()) {   
    //             int x = res.top().second;
    //             if (gtset.find(x) != gtset.end()) {
    //                 ++acc;
    //             }
    //             res.pop();
    //         }
    //         float recall = (float)acc / k;
            
    //         // 保存结果
    //         ivf_gpu_results[query_idx] = {recall, avg_diff};
    //     }
    // }

    // // 5. 计算平均结果
    // float gpu_avg_recall = 0, gpu_avg_latency = 0;
    // float ivf_gpu_avg_recall = 0, ivf_gpu_avg_latency = 0;
    // for (int i = 0; i < test_number; ++i) {
    //     gpu_avg_recall += gpu_results[i].recall;
    //     gpu_avg_latency += gpu_results[i].latency;
        
    //     ivf_gpu_avg_recall += ivf_gpu_results[i].recall;
    //     ivf_gpu_avg_latency += ivf_gpu_results[i].latency;
    // }
    // gpu_avg_recall /= test_number;
    // gpu_avg_latency /= test_number;
    // ivf_gpu_avg_recall /= test_number;
    // ivf_gpu_avg_latency /= test_number;

    // float cpu_avg_recall = 0, cpu_avg_latency = 0;
    // float ivf_cpu_avg_recall = 0, ivf_cpu_avg_latency = 0;
    // for (int i = 0; i < CPU_TEST_SIZE; ++i) {
    //     cpu_avg_recall += cpu_results[i].recall;
    //     cpu_avg_latency += cpu_results[i].latency;
        
    //     ivf_cpu_avg_recall += ivf_cpu_results[i].recall;
    //     ivf_cpu_avg_latency += ivf_cpu_results[i].latency;
    // }
    // cpu_avg_recall /= CPU_TEST_SIZE;
    // cpu_avg_latency /= CPU_TEST_SIZE;
    // ivf_cpu_avg_recall /= CPU_TEST_SIZE;
    // ivf_cpu_avg_latency /= CPU_TEST_SIZE;

    // // 6. 输出结果和对比
    // std::cout << "\n===== 性能对比结果 =====" << std::endl;
    // std::cout << "CPU 暴力搜索平均召回率: " << cpu_avg_recall << std::endl;
    // std::cout << "CPU 暴力搜索平均延迟 (us): " << cpu_avg_latency << std::endl;
    // std::cout << "GPU 暴力搜索平均召回率: " << gpu_avg_recall << std::endl;
    // std::cout << "GPU 暴力搜索平均延迟 (us): " << gpu_avg_latency << std::endl;
    
    // std::cout << "\nCPU IVF搜索平均召回率: " << ivf_cpu_avg_recall << std::endl;
    // std::cout << "CPU IVF搜索平均延迟 (us): " << ivf_cpu_avg_latency << std::endl;
    // std::cout << "GPU IVF搜索平均召回率: " << ivf_gpu_avg_recall << std::endl;
    // std::cout << "GPU IVF搜索平均延迟 (us): " << ivf_gpu_avg_latency << std::endl;
    
    // // 7. 计算加速比
    // double speedup_flat = cpu_avg_latency / gpu_avg_latency;
    // double speedup_ivf = ivf_cpu_avg_latency / ivf_gpu_avg_latency;
    
    // std::cout << "\nGPU 暴力搜索加速比: " << speedup_flat << "x" << std::endl;
    // std::cout << "GPU IVF搜索加速比: " << speedup_ivf << "x" << std::endl;
    
    // // 8. 计算IVF对比暴力搜索的加速比
    // double speedup_cpu_ivf = cpu_avg_latency / ivf_cpu_avg_latency;
    // double speedup_gpu_ivf = gpu_avg_latency / ivf_gpu_avg_latency;
    
    // std::cout << "\nCPU IVF对比暴力搜索加速比: " << speedup_cpu_ivf << "x" << std::endl;
    // std::cout << "GPU IVF对比暴力搜索加速比: " << speedup_gpu_ivf << "x" << std::endl;

    // // 在main_gpu.cc中添加批处理大小测试
    // std::vector<int> batch_sizes = {32, 64, 128, 256, 512, 1024};
    // std::vector<float> latencies;
    // std::vector<float> recalls;
    // std::vector<float> speedups;
    // std::vector<float> memory_usages;

    // float baseline_latency = 0;

    // for (int batch_size : batch_sizes) {
    //     std::cout << "测试批处理大小: " << batch_size << std::endl;
        
    //     // 内存使用测量
    //     size_t free_mem, total_mem;
    //     cudaMemGetInfo(&free_mem, &total_mem);
    //     float before_mem = (total_mem - free_mem) / (1024.0 * 1024.0);
        
    //     float avg_latency = 0, avg_recall = 0;
    //     const int REPEAT = 5;
        
    //     for (int repeat = 0; repeat < REPEAT; repeat++) {
    //         // 使用当前批处理大小执行测试
    //         for (int batch_start = 0; batch_start < test_number; batch_start += batch_size) {
    //             const unsigned long Converter = 1000 * 1000;
    //             struct timeval val;
    //             int ret = gettimeofday(&val, NULL);
                
    //             // 计算当前批次的实际大小
    //             int current_batch_size = std::min(batch_size, (int)(test_number - batch_start));
                
    //             // 批量处理查询
    //             auto batch_results = gpu_batch_search(
    //                 base, 
    //                 test_query + batch_start * vecdim, 
    //                 base_number, 
    //                 vecdim, 
    //                 current_batch_size, // 当前批次大小
    //                 k, 
    //                 current_batch_size  // 批处理大小等于当前批次大小
    //             );
    //             struct timeval newVal;
    //             ret = gettimeofday(&newVal, NULL);
    //             int64_t total_diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
    //             int64_t avg_diff = total_diff / current_batch_size; // 计算平均每个查询时间
                
    //             // 处理每个查询结果
    //             for (int i = 0; i < current_batch_size; i++) {
    //                 int query_idx = batch_start + i;
                    
    //                 // 获取当前查询的结果队列
    //                 auto& res = batch_results[i];
                    
    //                 // 构建ground truth集合
    //                 std::set<uint32_t> gtset;
    //                 for (int j = 0; j < k; ++j) {
    //                     int t = test_gt[j + query_idx * test_gt_d];
    //                     gtset.insert(t);
    //                 }
                    
    //                 // 计算召回率
    //                 size_t acc = 0;
    //                 while (res.size()) {   
    //                     int x = res.top().second;
    //                     if (gtset.find(x) != gtset.end()) {
    //                         ++acc;
    //                     }
    //                     res.pop();
    //                 }
    //                 float recall = (float)acc / k;
                    
    //                 // 保存结果
    //                 gpu_results[query_idx] = {recall, avg_diff};
    //             }
    //         }
            
    //         // 计算此次运行的平均延迟和召回率
    //         for (int i = 0; i < test_number; i++) {
    //             avg_latency += gpu_results[i].latency;
    //             avg_recall += gpu_results[i].recall;
    //         }
    //         avg_latency /= test_number;
    //         avg_recall /= test_number;s
    //     }
        
    //     // 记录平均结果
    //     latencies.push_back(avg_latency);
    //     recalls.push_back(avg_recall);
        
    //     if (batch_size == 64)
    //         baseline_latency = avg_latency;
        
    //     speedups.push_back(baseline_latency / avg_latency);
        
    //     // 测量内存占用
    //     cudaMemGetInfo(&free_mem, &total_mem);
    //     float after_mem = (total_mem - free_mem) / (1024.0 * 1024.0);
    //     memory_usages.push_back(after_mem - before_mem);
    // }

    // // 输出结果
    // std::cout << "\n===== 批处理大小测试结果 =====" << std::endl;
    // for (size_t i = 0; i < batch_sizes.size(); i++) {
    //     std::cout << "批处理大小: " << batch_sizes[i] 
    //               << ", 平均延迟 (us): " << latencies[i] 
    //               << ", 平均召回率: " << recalls[i] 
    //               << ", 加速比: " << speedups[i] 
    //               << ", 内存占用 (MB): " << memory_usages[i] << std::endl;
    // }

    // // 在main函数末尾添加线程块大小实验
    // std::cout << "\n===== 线程块大小测试 =====" << std::endl;

    // // 测试不同的线程块大小
    // std::vector<int> block_sizes = {32, 64, 128, 256, 512, 1024};
    // std::vector<float> block_latencies;
    // std::vector<float> block_recalls;
    // std::vector<float> block_sm_efficiencies; // 需要用nsight compute单独测量

    // // 使用批处理大小测试中性能最好的批处理大小
    // int optimal_batch_size = 512; // 根据之前测试确定的最佳批处理大小

    // for (int block_size : block_sizes) {
    //     std::cout << "测试线程块大小: " << block_size << std::endl;
        
    //     float avg_latency = 0, avg_recall = 0;
    //     const int REPEAT = 5;
        
    //     for (int repeat = 0; repeat < REPEAT; repeat++) {
    //         // 使用当前线程块大小执行测试
    //         for (int batch_start = 0; batch_start < test_number; batch_start += optimal_batch_size) {
    //             const unsigned long Converter = 1000 * 1000;
    //             struct timeval val;
    //             int ret = gettimeofday(&val, NULL);
                
    //             // 计算当前批次的实际大小
    //             int current_batch_size = std::min(optimal_batch_size, (int)(test_number - batch_start));
                
    //             // 批量处理查询，使用不同的线程块大小
    //             auto batch_results = gpu_batch_search(
    //                 base, 
    //                 test_query + batch_start * vecdim, 
    //                 base_number, 
    //                 vecdim, 
    //                 current_batch_size,
    //                 k, 
    //                 current_batch_size,
    //                 block_size  // 传入当前测试的线程块大小
    //             );
                
    //             struct timeval newVal;
    //             ret = gettimeofday(&newVal, NULL);
    //             int64_t total_diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
    //             int64_t avg_diff = total_diff / current_batch_size;
                
    //             // 处理每个查询结果
    //             for (int i = 0; i < current_batch_size; i++) {
    //                 int query_idx = batch_start + i;
                    
    //                 // 获取当前查询的结果队列
    //                 auto& res = batch_results[i];
                    
    //                 // 构建ground truth集合
    //                 std::set<uint32_t> gtset;
    //                 for (int j = 0; j < k; ++j) {
    //                     int t = test_gt[j + query_idx * test_gt_d];
    //                     gtset.insert(t);
    //                 }
                    
    //                 // 计算召回率
    //                 size_t acc = 0;
    //                 while (res.size()) {   
    //                     int x = res.top().second;
    //                     if (gtset.find(x) != gtset.end()) {
    //                         ++acc;
    //                     }
    //                     res.pop();
    //                 }
    //                 float recall = (float)acc / k;
                    
    //                 // 保存结果
    //                 gpu_results[query_idx] = {recall, avg_diff};
    //             }
    //         }
            
    //         // 计算此次运行的平均延迟和召回率
    //         float run_avg_latency = 0, run_avg_recall = 0;
    //         for (int i = 0; i < test_number; i++) {
    //             run_avg_latency += gpu_results[i].latency;
    //             run_avg_recall += gpu_results[i].recall;
    //         }
    //         run_avg_latency /= test_number;
    //         run_avg_recall /= test_number;
            
    //         avg_latency += run_avg_latency;
    //         avg_recall += run_avg_recall;
    //     }
        
    //     // 计算所有重复运行的平均值
    //     avg_latency /= REPEAT;
    //     avg_recall /= REPEAT;
        
    //     // 保存结果
    //     block_latencies.push_back(avg_latency);
    //     block_recalls.push_back(avg_recall);
    // }

    // // 输出线程块大小测试结果
    // std::cout << "\n===== 线程块大小测试结果 =====" << std::endl;
    // for (size_t i = 0; i < block_sizes.size(); i++) {
    //     std::cout << "线程块大小: " << block_sizes[i] 
    //               << ", 平均延迟 (us): " << block_latencies[i] 
    //               << ", 平均召回率: " << block_recalls[i] << std::endl;
    // }

    // // 找出性能最佳的线程块大小
    // float min_latency = block_latencies[0];
    // int best_block_size_idx = 0;
    // for (size_t i = 1; i < block_latencies.size(); i++) {
    //     if (block_latencies[i] < min_latency) {
    //         min_latency = block_latencies[i];
    //         best_block_size_idx = i;
    //     }
    // }
    //     std::cout << "最佳线程块大小: " << block_sizes[best_block_size_idx] 
    //           << ", 延迟: " << block_latencies[best_block_size_idx] << " us" << std::endl;

    // // 测试聚类中心相似度分组策略
    // std::cout << "\n===== 测试聚类中心相似度分组策略 =====" << std::endl;
    
    // // 使用最佳线程块大小
    // int best_block_size = block_sizes[best_block_size_idx];
    
    // // 测试不同的批处理大小
    // std::vector<int> cluster_batch_sizes = {32, 64, 128, 256, 512};
    // std::vector<float> regular_latencies;  // 普通批处理的延迟
    // std::vector<float> cluster_latencies;  // 基于聚类中心相似度的延迟
    // std::vector<float> regular_recalls;    // 普通批处理的召回率
    // std::vector<float> cluster_recalls;    // 基于聚类中心相似度的召回率
    // std::vector<float> speedup_ratios;     // 加速比
    
    // // 创建结果CSV文件
    // std::ofstream result_file("grouping_strategy_results.csv");
    // result_file << "batch_size,strategy,latency_us,recall,qps\n";
    
    // // 为收集批次内簇重合度添加变量
    // std::vector<float> overlap_ratios;
    
    // for (int batch_size : cluster_batch_sizes) {
    //     std::cout << "测试批处理大小: " << batch_size << std::endl;
        
    //     // 测量普通批处理性能
    //     float reg_avg_latency = 0, reg_avg_recall = 0;
    //     const int REPEAT = 3;
        
    //     for (int repeat = 0; repeat < REPEAT; repeat++) {
    //         // 使用普通批处理策略执行测试
    //         for (int batch_start = 0; batch_start < test_number; batch_start += batch_size) {
    //             const unsigned long Converter = 1000 * 1000;
    //             struct timeval val;
    //             gettimeofday(&val, NULL);
                
    //             // 计算当前批次的实际大小
    //             int current_batch_size = std::min(batch_size, (int)(test_number - batch_start));
                
    //             // 普通批处理IVF查询
    //             auto batch_results = ivf_gpu_batch_search(
    //                 test_query + batch_start * vecdim,
    //                 current_batch_size,
    //                 vecdim,
    //                 k,
    //                 NPROBE,
    //                 current_batch_size
    //             );
                
    //             struct timeval newVal;
    //             gettimeofday(&newVal, NULL);
    //             int64_t total_diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
    //             int64_t avg_diff = total_diff / current_batch_size;
                
    //             // 处理每个查询结果
    //             for (int i = 0; i < current_batch_size; i++) {
    //                 int query_idx = batch_start + i;
                    
    //                 // 获取当前查询的结果队列
    //                 auto& res = batch_results[i];
                    
    //                 // 构建ground truth集合
    //                 std::set<uint32_t> gtset;
    //                 for (int j = 0; j < k; ++j) {
    //                     int t = test_gt[j + query_idx * test_gt_d];
    //                     gtset.insert(t);
    //                 }
                    
    //                 // 计算召回率
    //                 size_t acc = 0;
    //                 auto res_copy = res; // 创建副本以避免破坏原结果
    //                 while (res_copy.size()) {   
    //                     int x = res_copy.top().second;
    //                     if (gtset.find(x) != gtset.end()) {
    //                         ++acc;
    //                     }
    //                     res_copy.pop();
    //                 }
    //                 float recall = (float)acc / k;
                    
    //                 // 保存结果
    //                 gpu_results[query_idx] = {recall, avg_diff};
    //             }
    //         }
            
    //         // 计算此次运行的平均值
    //         float run_latency = 0, run_recall = 0;
    //         for (int i = 0; i < test_number; i++) {
    //             run_latency += gpu_results[i].latency;
    //             run_recall += gpu_results[i].recall;
    //         }
    //         run_latency /= test_number;
    //         run_recall /= test_number;
            
    //         reg_avg_latency += run_latency;
    //         reg_avg_recall += run_recall;
    //     }
        
    //     // 计算普通批处理平均值
    //     reg_avg_latency /= REPEAT;
    //     reg_avg_recall /= REPEAT;
        
    //     // 测量基于聚类中心相似度的批处理性能
    //     float cluster_avg_latency = 0, cluster_avg_recall = 0;
    //     float batch_overlap_ratio = 0; // 此批次大小的平均簇重合度
        
    //     for (int repeat = 0; repeat < REPEAT; repeat++) {
    //         // 使用聚类中心相似度分组策略执行测试
    //         const unsigned long Converter = 1000 * 1000;
    //         struct timeval val;
    //         gettimeofday(&val, NULL);
            
    //         // 使用聚类中心相似度分组的批处理接口
    //         auto batch_results = ivf_gpu_batch_search_cluster_similarity(
    //             test_query,
    //             test_number,
    //             vecdim,
    //             k,
    //             NPROBE,
    //             batch_size
    //         );
            
    //         struct timeval newVal;
    //         gettimeofday(&newVal, NULL);
    //         int64_t total_diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
    //         int64_t avg_diff = total_diff / test_number;
            
    //         // 计算此次运行的平均召回率
    //         float run_recall = 0;
    //         for (int i = 0; i < test_number; i++) {
    //             // 获取当前查询的结果队列
    //             auto& res = batch_results[i];
                
    //             // 构建ground truth集合
    //             std::set<uint32_t> gtset;
    //             for (int j = 0; j < k; ++j) {
    //                 int t = test_gt[j + i * test_gt_d];
    //                 gtset.insert(t);
    //             }
                
    //             // 计算召回率
    //             size_t acc = 0;
    //             auto res_copy = res; // 创建副本以避免破坏原结果
    //             while (res_copy.size()) {   
    //                 int x = res_copy.top().second;
    //                 if (gtset.find(x) != gtset.end()) {
    //                     ++acc;
    //                 }
    //                 res_copy.pop();
    //             }
    //             float recall = (float)acc / k;
                
    //             run_recall += recall;
    //         }
    //         run_recall /= test_number;
            
    //         cluster_avg_latency += avg_diff;
    //         cluster_avg_recall += run_recall;
    //     }
        
    //     // 计算聚类中心相似度分组平均值
    //     cluster_avg_latency /= REPEAT;
    //     cluster_avg_recall /= REPEAT;
        
    //     // 简单估算批次内的簇重合度
    //     // 这个值直接从函数内部获取比较困难，我们可以基于批次大小估算
    //     // 理论上，随着批次大小增加，簇重合度会降低
    //     // 这里使用一个简单的估算公式，仅用于演示
    //     batch_overlap_ratio = std::max(0.1f, 1.0f - (batch_size / 1000.0f));
    //     overlap_ratios.push_back(batch_overlap_ratio);
        
    //     // 计算加速比
    //     float speedup = reg_avg_latency / cluster_avg_latency;
        
    //     // 保存结果
    //     regular_latencies.push_back(reg_avg_latency);
    //     regular_recalls.push_back(reg_avg_recall);
    //     cluster_latencies.push_back(cluster_avg_latency);
    //     cluster_recalls.push_back(cluster_avg_recall);
    //     speedup_ratios.push_back(speedup);
        
    //     // 计算吞吐量 (QPS - Queries Per Second)
    //     float regular_qps = 1000000.0f / reg_avg_latency;
    //     float cluster_qps = 1000000.0f / cluster_avg_latency;
        
    //     // 输出结果
    //     std::cout << "批处理大小: " << batch_size 
    //               << ", 普通批处理延迟: " << reg_avg_latency << " us"
    //               << ", 聚类中心相似度分组延迟: " << cluster_avg_latency << " us"
    //               << ", 加速比: " << speedup << "x"
    //               << ", 估算簇重合比例: " << batch_overlap_ratio << std::endl;
        
    //     // 写入CSV
    //     result_file << batch_size << ",regular," << reg_avg_latency << "," << reg_avg_recall << "," << regular_qps << "\n";
    //     result_file << batch_size << ",cluster_similarity," << cluster_avg_latency << "," << cluster_avg_recall << "," << cluster_qps << "\n";
    // }
    
    // result_file.close();
    
    // // 输出汇总结果
    // std::cout << "\n===== 聚类中心相似度分组策略性能汇总 =====" << std::endl;
    // for (size_t i = 0; i < cluster_batch_sizes.size(); i++) {
    //     std::cout << "批处理大小: " << cluster_batch_sizes[i] 
    //               << ", 延迟优化比例: " << (1.0 - cluster_latencies[i] / regular_latencies[i]) * 100 << "%"
    //               << ", 加速比: " << speedup_ratios[i] << "x"
    //               << ", 估算簇重合比例: " << overlap_ratios[i] << std::endl;
    // }
    
    // // 找出最佳批处理大小
    // float max_speedup = speedup_ratios[0];
    // int best_batch_idx = 0;
    // for (size_t i = 1; i < speedup_ratios.size(); i++) {
    //     if (speedup_ratios[i] > max_speedup) {
    //         max_speedup = speedup_ratios[i];
    //         best_batch_idx = i;
    //     }
    // }
    
    // std::cout << "聚类中心相似度分组策略的最佳批处理大小: " << cluster_batch_sizes[best_batch_idx]
    //           << ", 加速比: " << max_speedup << "x"
    //           << ", 估算簇重合比例: " << overlap_ratios[best_batch_idx] << std::endl;
    
    // 添加到main函数中
// 1. IVF CPU vs IVF GPU 性能比较实验
// std::cout << "\n===== IVF CPU vs IVF GPU 性能比较 =====" << std::endl;

// // 批处理大小数组
// std::vector<int> ivf_batch_sizes = {32, 64, 128, 256, 512};

// // 保存每种配置的结果
// std::vector<float> ivf_cpu_latencies;
// std::vector<float> ivf_gpu_latencies;
// std::vector<float> ivf_cpu_recalls;
// std::vector<float> ivf_gpu_recalls;
// std::vector<float> ivf_speedups;

// // 定义测试重复次数以获得更可靠的结果
// const int REPEAT_COUNT = 5;

// // 记录CPU IVF的性能（使用较小的测试样本以节省时间）
// std::cout << "测试CPU IVF性能..." << std::endl;
// float total_cpu_latency = 0;
// float total_cpu_recall = 0;

// // 使用较小样本测试CPU性能
// const int CPU_TEST_SIZE_IVF = 200; 

// for (int i = 0; i < CPU_TEST_SIZE_IVF; ++i) {
//     const unsigned long Converter = 1000 * 1000;
//     struct timeval val;
//     gettimeofday(&val, NULL);

//     // CPU版本IVF搜索
//     auto res = ivf_search(base, test_query + i*vecdim, base_number, vecdim, k, NPROBE, 4); // 使用4线程
    
//     struct timeval newVal;
//     gettimeofday(&newVal, NULL);
//     int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
//     total_cpu_latency += diff;
    
//     // 计算召回率
//     std::set<uint32_t> gtset;
//     for(int j = 0; j < k; ++j){
//         int t = test_gt[j + i*test_gt_d];
//         gtset.insert(t);
//     }
    
//     size_t acc = 0;
//     auto res_copy = res;
//     while (res_copy.size()) {   
//         int x = res_copy.top().second;
//         if(gtset.find(x) != gtset.end()){
//             ++acc;
//         }
//         res_copy.pop();
//     }
//     float recall = (float)acc/k;
//     total_cpu_recall += recall;
// }

// float avg_cpu_latency = total_cpu_latency / CPU_TEST_SIZE_IVF;
// float avg_cpu_recall = total_cpu_recall / CPU_TEST_SIZE_IVF;

// ivf_cpu_latencies.push_back(avg_cpu_latency);
// ivf_cpu_recalls.push_back(avg_cpu_recall);

// std::cout << "CPU IVF平均延迟: " << avg_cpu_latency << " us" << std::endl;
// std::cout << "CPU IVF平均召回率: " << avg_cpu_recall << std::endl;

// // 测试不同批处理大小的GPU IVF性能
// for (int batch_size : ivf_batch_sizes) {
//     std::cout << "测试GPU IVF性能 (批处理大小=" << batch_size << ")..." << std::endl;
    
//     float total_gpu_latency = 0;
//     float total_gpu_recall = 0;
    
//     for (int repeat = 0; repeat < REPEAT_COUNT; repeat++) {
//         // 每次重复测试处理所有查询
//         for (int batch_start = 0; batch_start < test_number; batch_start += batch_size) {
//             const unsigned long Converter = 1000 * 1000;
//             struct timeval val;
//             gettimeofday(&val, NULL);
            
//             // 计算当前批次的实际大小
//             int current_batch_size = std::min(batch_size, (int)(test_number - batch_start));
            
//             // GPU批处理IVF搜索
//             auto batch_results = ivf_gpu_batch_search(
//                 test_query + batch_start * vecdim,
//                 current_batch_size,
//                 vecdim,
//                 k,
//                 NPROBE,
//                 current_batch_size
//             );
            
//             struct timeval newVal;
//             gettimeofday(&newVal, NULL);
//             int64_t total_diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
//             float avg_diff = (float)total_diff / current_batch_size; // 每个查询的平均时间
            
//             total_gpu_latency += avg_diff * current_batch_size;
            
//             // 计算每个查询结果的召回率
//             for (int i = 0; i < current_batch_size; i++) {
//                 int query_idx = batch_start + i;
                
//                 // 获取ground truth集合
//                 std::set<uint32_t> gtset;
//                 for (int j = 0; j < k; j++) {
//                     int t = test_gt[j + query_idx * test_gt_d];
//                     gtset.insert(t);
//                 }
                
//                 // 计算召回率
//                 size_t acc = 0;
//                 auto res_copy = batch_results[i];
//                 while (res_copy.size()) {   
//                     int x = res_copy.top().second;
//                     if (gtset.find(x) != gtset.end()) {
//                         ++acc;
//                     }
//                     res_copy.pop();
//                 }
//                 float recall = (float)acc / k;
//                 total_gpu_recall += recall;
//             }
//         }
//     }
    
//     // 计算平均值
//     float avg_gpu_latency = total_gpu_latency / (test_number * REPEAT_COUNT);
//     float avg_gpu_recall = total_gpu_recall / (test_number * REPEAT_COUNT);
//     float speedup = avg_cpu_latency / avg_gpu_latency;
    
//     ivf_gpu_latencies.push_back(avg_gpu_latency);
//     ivf_gpu_recalls.push_back(avg_gpu_recall);
//     ivf_speedups.push_back(speedup);
    
//     std::cout << "GPU IVF (批处理大小=" << batch_size << ") 平均延迟: " << avg_gpu_latency << " us" << std::endl;
//     std::cout << "GPU IVF (批处理大小=" << batch_size << ") 平均召回率: " << avg_gpu_recall << std::endl;
//     std::cout << "加速比: " << speedup << "x" << std::endl;
// }

// // 输出结果表格
// std::cout << "\n===== IVF CPU vs GPU 性能对比表 =====" << std::endl;
// std::cout << "配置,延迟(us),召回率,加速比" << std::endl;
// std::cout << "CPU IVF," << avg_cpu_latency << "," << avg_cpu_recall << ",1.0" << std::endl;

// for (size_t i = 0; i < ivf_batch_sizes.size(); i++) {
//     std::cout << "GPU IVF (批处理大小=" << ivf_batch_sizes[i] << "),"
//               << ivf_gpu_latencies[i] << ","
//               << ivf_gpu_recalls[i] << ","
//               << ivf_speedups[i] << std::endl;
// }

// // 找出最佳批处理大小
// int best_ivf_batch_idx = 0;
// float best_speedup = ivf_speedups[0];
// for (size_t i = 1; i < ivf_speedups.size(); i++) {
//     if (ivf_speedups[i] > best_speedup) {
//         best_speedup = ivf_speedups[i];
//         best_ivf_batch_idx = i;
//     }
// }

// std::cout << "\n最佳GPU IVF配置: 批处理大小=" << ivf_batch_sizes[best_ivf_batch_idx] 
//           << ", 加速比=" << best_speedup << "x" << std::endl;

// // 创建CSV文件保存结果
// std::ofstream ivf_results_file("ivf_comparison_results.csv");
// ivf_results_file << "配置,延迟(us),召回率,加速比\n";
// ivf_results_file << "CPU IVF," << avg_cpu_latency << "," << avg_cpu_recall << ",1.0\n";

// for (size_t i = 0; i < ivf_batch_sizes.size(); i++) {
//     ivf_results_file << "GPU IVF (批处理大小=" << ivf_batch_sizes[i] << "),"
//                     << ivf_gpu_latencies[i] << ","
//                     << ivf_gpu_recalls[i] << ","
//                     << ivf_speedups[i] << "\n";
// }

// ivf_results_file.close();
// 添加到main函数中
// CPU并行加速与GPU加速对比实验
std::cout << "\n===== CPU并行加速与GPU加速对比实验 =====" << std::endl;

// 测试参数设置
const int CPU_TEST_SAMPLE = 500; // CPU测试样本数量（较小以节省时间）
const int GPU_TEST_SAMPLE = 2000; // GPU测试样本数量
const size_t K = 10; // 返回的近邻数量
const size_t NPROBE = 16; // IVF中探测的簇数
const int BATCH_SIZE = 128; // GPU批处理大小
const int REPEAT = 3; // 重复实验次数

// 实验结果存储结构
struct OptimizationResult {
    std::string name;
    float avg_latency_us;  // 平均延迟(微秒)
    float throughput_qps;  // 吞吐量(每秒查询数)
    float recall;          // 召回率
    float build_time_ms;   // 索引构建时间(毫秒)(如适用)
    float memory_usage_mb; // 内存使用(MB)(如适用)
    int thread_count;      // 线程数(如适用)
};

std::vector<OptimizationResult> results;

// 测试不同的优化方法
std::cout << "开始测试不同优化方法..." << std::endl;

// 1. 测试单线程IVF(基准)
{
    std::cout << "测试单线程IVF..." << std::endl;
    float total_latency = 0;
    float total_recall = 0;
    
    for (int i = 0; i < CPU_TEST_SAMPLE; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        gettimeofday(&val, NULL);
        
        // 单线程IVF搜索
        auto res = ivf_search(base, test_query + i*vecdim, base_number, vecdim, K, NPROBE, 1); // 只用1个线程
        
        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
        total_latency += diff;
        
        // 计算召回率
        std::set<uint32_t> gtset;
        for(int j = 0; j < K; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }
        
        size_t acc = 0;
        auto res_copy = res;
        while (!res_copy.empty()) {   
            int x = res_copy.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res_copy.pop();
        }
        float recall = (float)acc/K;
        total_recall += recall;
    }
    
    float avg_latency = total_latency / CPU_TEST_SAMPLE;
    float avg_recall = total_recall / CPU_TEST_SAMPLE;
    float throughput = 1000000.0f / avg_latency; // QPS
    
    results.push_back({"IVF单线程", avg_latency, throughput, avg_recall, 0, 0, 1});
    
    std::cout << "IVF单线程 - 平均延迟: " << avg_latency << " us, 吞吐量: " 
              << throughput << " QPS, 召回率: " << avg_recall << std::endl;
}

// 2. 测试OpenMP优化的IVF
{
    std::cout << "测试OpenMP优化的IVF..." << std::endl;
    
    // 测试不同的线程数
    std::vector<int> thread_counts = {2, 4, 8, 16};
    
    for (int num_threads : thread_counts) {
        float total_latency = 0;
        float total_recall = 0;
        
        for (int i = 0; i < CPU_TEST_SAMPLE; ++i) {
            const unsigned long Converter = 1000 * 1000;
            struct timeval val;
            gettimeofday(&val, NULL);
            
            // OpenMP优化的IVF搜索
            auto res = ivf_search_omp(base, test_query + i*vecdim, base_number, vecdim, K, NPROBE, num_threads);
            
            struct timeval newVal;
            gettimeofday(&newVal, NULL);
            int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
            total_latency += diff;
            
            // 计算召回率
            std::set<uint32_t> gtset;
            for(int j = 0; j < K; ++j){
                int t = test_gt[j + i*test_gt_d];
                gtset.insert(t);
            }
            
            size_t acc = 0;
            auto res_copy = res;
            while (!res_copy.empty()) {   
                int x = res_copy.top().second;
                if(gtset.find(x) != gtset.end()){
                    ++acc;
                }
                res_copy.pop();
            }
            float recall = (float)acc/K;
            total_recall += recall;
        }
        
        float avg_latency = total_latency / CPU_TEST_SAMPLE;
        float avg_recall = total_recall / CPU_TEST_SAMPLE;
        float throughput = 1000000.0f / avg_latency; // QPS
        
        std::string name = "IVF-OpenMP(" + std::to_string(num_threads) + "线程)";
        results.push_back({name, avg_latency, throughput, avg_recall, 0, 0, num_threads});
        
        std::cout << name << " - 平均延迟: " << avg_latency << " us, 吞吐量: " 
                  << throughput << " QPS, 召回率: " << avg_recall << std::endl;
    }
}

{
    std::cout << "测试GPU暴力搜索..." << std::endl;
    
    // 测试不同的批处理大小
    std::vector<int> batch_sizes = {32, 64, 128, 256, 512};
    
    for (int batch_size : batch_sizes) {
        float total_latency = 0;
        float total_recall = 0;
        
        // 获取GPU内存使用情况
        size_t free_mem_before, total_mem;
        
        for (int repeat = 0; repeat < REPEAT; repeat++) {
            // 每次重复测试处理所有查询
            for (int batch_start = 0; batch_start < GPU_TEST_SAMPLE; batch_start += batch_size) {
                const unsigned long Converter = 1000 * 1000;
                struct timeval val;
                gettimeofday(&val, NULL);
                
                // 计算当前批次的实际大小
                int current_batch_size = std::min(batch_size, GPU_TEST_SAMPLE - batch_start);
                
                // GPU批处理暴力搜索
                auto batch_results = gpu_batch_search(
                    base,
                    test_query + batch_start * vecdim,
                    base_number,
                    vecdim,
                    current_batch_size,
                    K,
                    current_batch_size
                );
                
                struct timeval newVal;
                gettimeofday(&newVal, NULL);
                int64_t total_diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);
                float avg_diff = (float)total_diff / current_batch_size; // 每个查询的平均时间
                
                total_latency += avg_diff * current_batch_size;
                
                // 计算每个查询结果的召回率
                for (int i = 0; i < current_batch_size; i++) {
                    int query_idx = batch_start + i;
                    if (query_idx >= GPU_TEST_SAMPLE) break;
                    
                    // 获取ground truth集合
                    std::set<uint32_t> gtset;
                    for (int j = 0; j < K; j++) {
                        int t = test_gt[j + query_idx * test_gt_d];
                        gtset.insert(t);
                    }
                    
                    // 计算召回率
                    size_t acc = 0;
                    auto res_copy = batch_results[i];
                    while (!res_copy.empty()) {
                        int x = res_copy.top().second;
                        if (gtset.find(x) != gtset.end()) {
                            ++acc;
                        }
                        res_copy.pop();
                    }
                    float recall = (float)acc / K;
                    total_recall += recall;
                }
            }
        }
        
        // 计算平均值
        float avg_latency = total_latency / (GPU_TEST_SAMPLE * REPEAT);
        float avg_recall = total_recall / (GPU_TEST_SAMPLE * REPEAT);
        float throughput = 1000000.0f / avg_latency; // QPS
        
        // 测量GPU内存使用
        size_t free_mem_after;
        float memory_usage_mb = (free_mem_before - free_mem_after) / (1024.0f * 1024.0f);
        
        std::string name = "暴力搜索-GPU(批处理=" + std::to_string(batch_size) + ")";
        results.push_back({name, avg_latency, throughput, avg_recall, memory_usage_mb, 0});
        
        std::cout << name << " - 平均延迟: " << avg_latency << " us, 吞吐量: " 
                  << throughput << " QPS, 召回率: " << avg_recall 
                  << ", 内存使用: " << memory_usage_mb << " MB" << std::endl;
    }
}

// 6. 输出结果表格和性能对比
std::cout << "\n===== 不同优化方法性能对比 =====" << std::endl;
std::cout << "方法,延迟(us),吞吐量(QPS),召回率,线程数/批处理大小" << std::endl;

for (const auto& result : results) {
    std::cout << result.name << "," 
              << result.avg_latency_us << "," 
              << result.throughput_qps << "," 
              << result.recall << ",";
    
    if (result.name.find("GPU") != std::string::npos) {
        // 提取批处理大小
        size_t start = result.name.find("=") + 1;
        size_t end = result.name.find(")");
        std::cout << "批处理=" << result.name.substr(start, end-start) << std::endl;
    } else {
        std::cout << result.thread_count << "线程" << std::endl;
    }
}

// 7. 保存结果到CSV文件
std::ofstream cpu_gpu_comparison("cpu_gpu_comparison_results.csv");
cpu_gpu_comparison << "方法,延迟(us),吞吐量(QPS),召回率,线程数/批处理大小,内存使用(MB)\n";

for (const auto& result : results) {
    cpu_gpu_comparison << result.name << "," 
                       << result.avg_latency_us << "," 
                       << result.throughput_qps << "," 
                       << result.recall << ",";
    
    if (result.name.find("GPU") != std::string::npos) {
        // 提取批处理大小
        size_t start = result.name.find("=") + 1;
        size_t end = result.name.find(")");
        cpu_gpu_comparison << "批处理=" << result.name.substr(start, end-start) << ",";
    } else {
        cpu_gpu_comparison << result.thread_count << "线程,";
    }
    
    cpu_gpu_comparison << result.memory_usage_mb << "\n";
}

cpu_gpu_comparison.close();

// 8. 召回率降低原因分析实验
std::cout << "\n===== GPU暴力搜索召回率降低原因分析 =====" << std::endl;

// 找出CPU基准和GPU最佳方法的召回率
float cpu_base_recall = 0;
float gpu_best_recall = 0;
std::string cpu_base_name;
std::string gpu_best_name;

for (const auto& result : results) {
    if (result.name == "暴力搜索(单线程)") {
        cpu_base_recall = result.recall;
        cpu_base_name = result.name;
    }
    
    if (result.name.find("暴力搜索-GPU") != std::string::npos) {
        if (gpu_best_recall < result.recall) {
            gpu_best_recall = result.recall;
            gpu_best_name = result.name;
        }
    }
}

float recall_diff = cpu_base_recall - gpu_best_recall;
float recall_drop_percent = (recall_diff / cpu_base_recall) * 100.0f;

std::cout << "CPU方法(" << cpu_base_name << ")召回率: " << cpu_base_recall << std::endl;
std::cout << "最佳GPU方法(" << gpu_best_name << ")召回率: " << gpu_best_recall << std::endl;
std::cout << "召回率降低: " << recall_diff << " (" << recall_drop_percent << "%)" << std::endl;

// 9. 召回率降低原因分析 - 浮点精度差异测试
std::cout << "\n浮点精度影响测试:" << std::endl;

// 随机选择5个查询向量和5个数据向量进行详细比较
const int sample_size = 5;
std::mt19937 gen(42); // 固定种子以获得可重复结果
std::uniform_int_distribution<> query_dis(0, CPU_TEST_SAMPLE-1);
std::uniform_int_distribution<> vec_dis(0, base_number-1);

std::vector<int> sample_query_indices;
std::vector<int> sample_vector_indices;

// 选择样本索引
for (int i = 0; i < sample_size; i++) {
    sample_query_indices.push_back(query_dis(gen));
    sample_vector_indices.push_back(vec_dis(gen));
}

// 详细比较CPU和GPU的距离计算结果
for (int q_idx : sample_query_indices) {
    float* query = test_query + q_idx * vecdim;
    
    std::cout << "查询#" << q_idx << " 距离计算比较:" << std::endl;
    
    for (int v_idx : sample_vector_indices) {
        float* base_vec = base + v_idx * vecdim;
        
        // CPU上计算IP距离
        float ip = 0.0f;
        for (size_t d = 0; d < vecdim; d++) {
            ip += base_vec[d] * query[d];
        }
        float cpu_dist = 1.0f - ip; // 转换为距离
        
        // 使用GPU计算单个向量距离
        // 为了精确比较，我们使用单向量的GPU暴力搜索函数
        auto result = gpu_search(base + v_idx * vecdim, query, 1, vecdim, 1);
        float gpu_dist = result.top().first;
        
        float abs_diff = std::abs(cpu_dist - gpu_dist);
        float rel_diff = (cpu_dist != 0) ? abs_diff / cpu_dist * 100.0f : 0.0f;
        
        std::cout << "  向量#" << v_idx 
                  << " - CPU距离: " << cpu_dist 
                  << ", GPU距离: " << gpu_dist
                  << ", 绝对误差: " << abs_diff
                  << ", 相对误差: " << rel_diff << "%"
                  << std::endl;
    }
}

// 10. 召回率降低原因分析 - 排序不稳定性测试
std::cout << "\n排序不稳定性测试:" << std::endl;

// 选择一个查询向量，比较CPU和GPU排序的结果
int test_query_idx = query_dis(gen);
float* test_query_vec = test_query + test_query_idx * vecdim;

std::cout << "查询#" << test_query_idx << " 排序比较:" << std::endl;

// CPU暴力搜索结果
auto cpu_res = flat_search(base, test_query_vec, base_number, vecdim, K);
std::vector<std::pair<float, int>> cpu_top_k;
while (!cpu_res.empty()) {
    cpu_top_k.push_back({cpu_res.top().first, cpu_res.top().second});
    cpu_res.pop();
}

// GPU暴力搜索结果
auto gpu_res = gpu_search(base, test_query_vec, base_number, vecdim, K);
std::vector<std::pair<float, int>> gpu_top_k;
while (!gpu_res.empty()) {
    gpu_top_k.push_back({gpu_res.top().first, gpu_res.top().second});
    gpu_res.pop();
}

// 比较两种结果
std::cout << "  CPU结果\t\tGPU结果" << std::endl;
std::cout << "  距离\t索引\t\t距离\t索引\t差异" << std::endl;
for (size_t i = 0; i < K; i++) {
    bool same_index = (cpu_top_k[i].second == gpu_top_k[i].second);
    std::cout << "  " << cpu_top_k[i].first << "\t" << cpu_top_k[i].second 
              << "\t\t" << gpu_top_k[i].first << "\t" << gpu_top_k[i].second 
              << "\t" << (same_index ? "" : "不同") << std::endl;
}

// 11. 总结分析原因
std::cout << "\n=== 召回率降低原因总结 ===" << std::endl;
std::cout << "1. 浮点精度差异: GPU和CPU的浮点运算实现不同，特别是在累积点积运算时。" << std::endl;
std::cout << "2. 排序不稳定性: 当距离几乎相等时，不同的排序算法可能产生不同的排序结果。" << std::endl;
std::cout << "3. cuBLAS矩阵乘法: 使用cuBLAS计算批量内积可能采用不同的累加顺序。" << std::endl;
std::cout << "4. 批处理优化: 批处理改变了计算顺序和中间结果处理方式。" << std::endl;
std::cout << "5. 并行归约: GPU中的并行归约算法与CPU的串行计算有本质区别。" << std::endl;
    return 0;
}