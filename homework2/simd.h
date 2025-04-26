#include <arm_neon.h>
#include <stdio.h>
#include <assert.h>
#include <arm_acle.h>
#include "sim8float32.h"
#pragma once
#include <queue>
#include <iomanip>
#include <omp.h>
#include <iostream>
float InnerProductSIMDNeon(const float* b1, const float* b2, size_t vecdim){
    assert(vecdim % 8 == 0);
    float zeros[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    simd8float32 sum(zeros);
    for (size_t i = 0; i < vecdim; i += 8) {
        simd8float32 a1(b1 + i);
        simd8float32 a2(b2 + i);
        simd8float32 prod = a1 * a2;
        sum = sum + prod;
    }
    float tmp[8];
    sum.storeu(tmp);
    float dis = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
    return 1-dis;
}

std::priority_queue<std::pair<float, uint32_t>> simflat_search(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    
    // 移除了并行处理，直接遍历所有基础向量
    for(int i = 0; i < base_number; ++i) {
        float dis = 0;
        // 计算内积，使用SIMD优化
        dis = InnerProductSIMDNeon(base + i * vecdim, query, vecdim);
        
        if(q.size() < k) {
            q.push({dis, i});
        } else {
            if(dis < q.top().first) {
                q.push({dis, i});
                q.pop();
            }
        }
    }
    
    return q;
}