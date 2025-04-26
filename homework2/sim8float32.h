#include <arm_neon.h>

struct simd8float32 {
    float32x4x2_t data;

    simd8float32() = default;

    simd8float32(const float* x, const float* y) : data{vld1q_f32(x), vld1q_f32(y)} {}

    explicit simd8float32(const float* x) : data{vld1q_f32(x), vld1q_f32(x+4)} {}
    explicit simd8float32(float value) {
        data.val[0] = vdupq_n_f32(value);
        data.val[1] = vdupq_n_f32(value);
    }
    simd8float32 operator*(const simd8float32& other) const{
        simd8float32 result;
        result.data.val[0] = vmulq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vmulq_f32(data.val[1], other.data.val[1]);
        return result;
    }

    simd8float32 operator+(const simd8float32& other) const{
        simd8float32 result;
        result.data.val[0] = vaddq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vaddq_f32(data.val[1], other.data.val[1]);
        return result;
    }
    simd8float32 operator-(const simd8float32& other) const{
        simd8float32 result;
        result.data.val[0] = vsubq_f32(data.val[0], other.data.val[0]);
        result.data.val[1] = vsubq_f32(data.val[1], other.data.val[1]);
        return result;
    }

    void storeu(float* x) const{
        vst1q_f32(x, data.val[0]);
        vst1q_f32(x + 4, data.val[1]);
    }
    void store(float* x) const{
        vst1q_f32(x, data.val[0]);
        vst1q_f32(x + 4, data.val[1]);
    }
};
