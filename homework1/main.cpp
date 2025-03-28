#include <iostream>
#include <ctime>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include <vector>
using namespace std;
// 定义与典型缓存大小相关的测试大小
    vector<int> matrix_sizes = {64, 128, 256, 512, 1024, 2048};
    vector<int> array_sizes = {1024, 16*1024, 256*1024, 1024*1024, 4*1024*1024};
    vector<int> memory_pressure = {0, 8, 32, 128}; 
    
    const int repeat_mat = 100;  // 矩阵测试重复次数
    const int repeat_sum = 100; // 求和测试重复次数
double* multiply(double* vec, double** mat, int n, int m) {
    double* sum = new double[m];
    for(int i=0; i<m; i++) {
        sum[i]=0;
        for(int j=0; j<n; j++) {
            sum[i]+=vec[j]*mat[j][i];
        }
    }
    return sum;
}

double* cache_multiply(double* vec, double** mat, int n, int m) {
    double* sum = new double[m];
    for(int i=0; i<m; i++) {
        sum[i]=0;
    }
    for(int i=0; i<n; i++) {
        for(int j=0; j<m; j++) {
            sum[j]+=vec[i]*mat[i][j];
        }
    }
    return sum;
}

double nsum(double* arr, int n) {
    double sum=0;
    for(int i=0; i<n; i++) {
        sum+=arr[i];
    }
    return sum;
}

double nsum_unrolled(double* arr, int n) {
    double sum=0;
    int i = 0;

    for(; i<n-3; i+=4) {
        sum += arr[i] + arr[i+1] + arr[i+2] + arr[i+3];
    }
 
    for(; i<n; i++) {
        sum += arr[i];
    }
    return sum;
}

double superscalar_nsum1(double* arr, int n) {
    double sum1=0, sum2=0;
    for(int i=0; i<n; i+=2) {
        sum1+=arr[i];
        if (i+1 < n) { 
            sum2+=arr[i+1];
        }
    }
    return sum1 + sum2;
}

double superscalar_nsum1_quad(double* arr, int n) {
    double sum1=0, sum2=0, sum3=0, sum4=0;
    for(int i=0; i<n; i+=4) {
        if (i < n) sum1 += arr[i];
        if (i+1 < n) sum2 += arr[i+1];
        if (i+2 < n) sum3 += arr[i+2];
        if (i+3 < n) sum4 += arr[i+3];
    }
    return sum1 + sum2 + sum3 + sum4;
}

double superscalar_nsum2(double* arr, int n, int m) {
    if(n==1) {
        return arr[0];
    }
    else if(m==1) {
        for(int i=0; i<n/2; i++) {
            arr[i]+=arr[i+n/2];
        }
        return superscalar_nsum2(arr, n/2, 1);
    }
    else if(m==2) {
        for(int i=n; i>1; i/=2) {
            for(int j=0; j<i/2; j++) {
                arr[j]=arr[j*2]+arr[j*2+1];
            }
        }
        return arr[0];
    }
    else {
        return -1;
    }
}


void fill_memory(size_t megabytes) {
    if (megabytes == 0) return;
    
    size_t size = megabytes * 1024 * 1024 / sizeof(double);
    double* mem_filler = new double[size];
    

    for (size_t i = 0; i < size; i++) {
        mem_filler[i] = (double)i;
    }
    

    volatile double sum = 0;
    for (size_t i = 0; i < size; i += 1024) {
        sum += mem_filler[i];
    }
}
int main() {
    //利用大模型生成规范的测试框架
    clock_t start, end;
    
    cout << "============= Matrix-Vector Multiplication Performance Test =============" << endl;
    cout << "Size | Column-wise (ms) | Row-wise (ms) | Speedup | Theoretical Operations" << endl;
    cout << "----------------------------------------------------------------------" << endl;
    
    for (int n : matrix_sizes) {
        // 创建测试数据：matrix[i][j] = i+j, vector[i] = i
        double** matrix = new double*[n];
        for (int i = 0; i < n; i++) {
            matrix[i] = new double[n];
            for (int j = 0; j < n; j++) {
                matrix[i][j] = i + j;
            }
        }
        
        double* vec = new double[n];
        for (int i = 0; i < n; i++) {
            vec[i] = i;
        }
        
        double* dot_out_trivial = nullptr;
        double* dot_out_cache = nullptr;
        
        // 测试列式访问（朴素算法）
        start = clock();
        for (int r = 0; r < repeat_mat; r++) {
            if (dot_out_trivial) delete[] dot_out_trivial;
            dot_out_trivial = multiply(vec, matrix, n, n);
        }
        end = clock();
        double time_trivial = double(end - start) / CLOCKS_PER_SEC * 1000;
        
        // 测试行式访问（缓存优化）
        start = clock();
        for (int r = 0; r < repeat_mat; r++) {
            if (dot_out_cache) delete[] dot_out_cache;
            dot_out_cache = cache_multiply(vec, matrix, n, n);
        }
        end = clock();
        double time_cache = double(end - start) / CLOCKS_PER_SEC * 1000;
        
        // 计算加速比和理论操作次数
        double speedup = time_trivial / time_cache;
        long long operations = 2LL * n * n; // 每个元素需要1次乘法和1次加法
        
        // 打印结果
        cout << setw(4) << n << " | "
             << fixed << setprecision(2) << setw(15) << time_trivial << " | "
             << setw(13) << time_cache << " | "
             << setw(7) << speedup << " | "
             << setw(10) << operations << endl;
             
        // 清理资源
        for (int i = 0; i < n; i++) {
            delete[] matrix[i];
        }
        delete[] matrix;
        delete[] vec;
        delete[] dot_out_trivial;
        delete[] dot_out_cache;
    }
    
    // 用不同问题规模测试数组求和
    cout << "\n================ Array Summation Performance Test ================" << endl;
    cout << "Size(K) | Naive(ms) | Unrolled(ms) | Dual-Chain(ms) | Quad-Chain(ms) | Recursive(ms)" << endl;
    cout << "---------------------------------------------------------------------------" << endl;
    
    for (int n : array_sizes) {
        // 生成测试数据：所有元素 = 1.0，方便验证
        double* arr = new double[n];
        for (int i = 0; i < n; i++) {
            arr[i] = 1.0;
        }
        
        double time_naive = 0, time_unrolled = 0, time_dual = 0, time_quad = 0, time_recursive = 0;
        double sum_naive = 0, sum_unrolled = 0, sum_dual = 0, sum_quad = 0, sum_recursive = 0;
        
        // 测试朴素求和
        start = clock();
        for (int r = 0; r < repeat_sum; r++) {
            sum_naive = nsum(arr, n);
        }
        end = clock();
        time_naive = double(end - start) / CLOCKS_PER_SEC * 1000;
        
        // 测试展开循环求和
        start = clock();
        for (int r = 0; r < repeat_sum; r++) {
            sum_unrolled = nsum_unrolled(arr, n);
        }
        end = clock();
        time_unrolled = double(end - start) / CLOCKS_PER_SEC * 1000;
        
        // 测试双链求和
        start = clock();
        for (int r = 0; r < repeat_sum; r++) {
            sum_dual = superscalar_nsum1(arr, n);
        }
        end = clock();
        time_dual = double(end - start) / CLOCKS_PER_SEC * 1000;
        
        // 测试四链求和
        start = clock();
        for (int r = 0; r < repeat_sum; r++) {
            sum_quad = superscalar_nsum1_quad(arr, n);
        }
        end = clock();
        time_quad = double(end - start) / CLOCKS_PER_SEC * 1000;
        
        // 测试递归求和
        double* arr_copy = new double[n];
        start = clock();
        for (int r = 0; r < repeat_sum; r++) {
            memcpy(arr_copy, arr, n * sizeof(double));
            sum_recursive = superscalar_nsum2(arr_copy, n, 1);
        }
        end = clock();
        time_recursive = double(end - start) / CLOCKS_PER_SEC * 1000;
        
        // 打印结果
        cout << setw(7) << n/1024 << " | " 
             << fixed << setprecision(2) << setw(9) << time_naive << " | "
             << setw(12) << time_unrolled << " | "
             << setw(14) << time_dual << " | "
             << setw(14) << time_quad << " | "
             << setw(13) << time_recursive << endl;
             
        // 检查结果一致性
        if (abs(sum_naive - n) > 0.1 || abs(sum_unrolled - n) > 0.1 || 
            abs(sum_dual - n) > 0.1 || abs(sum_quad - n) > 0.1 || 
            abs(sum_recursive - n) > 0.1) {
            cout << "Error: Inconsistent results! "
                 << sum_naive << ", " << sum_unrolled << ", " 
                 << sum_dual << ", " << sum_quad << ", " 
                 << sum_recursive << " (expected: " << n << ")" << endl;
        }
        
        // 清理资源
        delete[] arr;
        delete[] arr_copy;
    }
    
    // 测试内存压力对缓存效果的影响
    cout << "\n============= Cache Sensitivity Test =============" << endl;
    cout << "Memory(MB) | Column-wise(ms) | Row-wise(ms) | Naive Sum(ms) | Dual-Chain(ms)" << endl;
    cout << "------------------------------------------------------------------------" << endl;
    
    const int fixed_n = 1024; // 缓存测试的固定问题规模
    
    for (int mem : memory_pressure) {
        // 分配内存以在缓存上创建压力
        fill_memory(mem);
        
        // 创建测试数据
        double** matrix = new double*[fixed_n];
        for (int i = 0; i < fixed_n; i++) {
            matrix[i] = new double[fixed_n];
            for (int j = 0; j < fixed_n; j++) {
                matrix[i][j] = i + j;
            }
        }
        
        double* vec = new double[fixed_n];
        for (int i = 0; i < fixed_n; i++) {
            vec[i] = i;
        }
        
        double* arr = new double[fixed_n * fixed_n];
        for (int i = 0; i < fixed_n * fixed_n; i++) {
            arr[i] = 1.0;
        }
        
        double* dot_out_trivial = nullptr;
        double* dot_out_cache = nullptr;
        double time_trivial = 0, time_cache = 0, time_sum_naive = 0, time_sum_dual = 0;
        
        // 列式矩阵乘法测试
        start = clock();
        for (int r = 0; r < repeat_mat; r++) {
            if (dot_out_trivial) delete[] dot_out_trivial;
            dot_out_trivial = multiply(vec, matrix, fixed_n, fixed_n);
        }
        end = clock();
        time_trivial = double(end - start) / CLOCKS_PER_SEC * 1000;
        
        // 行式矩阵乘法测试
        start = clock();
        for (int r = 0; r < repeat_mat; r++) {
            if (dot_out_cache) delete[] dot_out_cache;
            dot_out_cache = cache_multiply(vec, matrix, fixed_n, fixed_n);
        }
        end = clock();
        time_cache = double(end - start) / CLOCKS_PER_SEC * 1000;
        
        // 朴素数组求和测试
        double sum_naive = 0;
        start = clock();
        for (int r = 0; r < repeat_sum; r++) {
            sum_naive = nsum(arr, fixed_n * fixed_n);
        }
        end = clock();
        time_sum_naive = double(end - start) / CLOCKS_PER_SEC * 1000;
        
        // 双链数组求和测试
        double sum_dual = 0;
        start = clock();
        for (int r = 0; r < repeat_sum; r++) {
            sum_dual = superscalar_nsum1(arr, fixed_n * fixed_n);
        }
        end = clock();
        time_sum_dual = double(end - start) / CLOCKS_PER_SEC * 1000;
        
        // 打印结果
        cout << setw(10) << mem << " | " 
             << fixed << setprecision(2) << setw(14) << time_trivial << " | "
             << setw(12) << time_cache << " | "
             << setw(13) << time_sum_naive << " | "
             << setw(14) << time_sum_dual << endl;
             
        // 清理资源
        for (int i = 0; i < fixed_n; i++) {
            delete[] matrix[i];
        }
        delete[] matrix;
        delete[] vec;
        delete[] arr;
        delete[] dot_out_trivial;
        delete[] dot_out_cache;
    }
    return 0;
}