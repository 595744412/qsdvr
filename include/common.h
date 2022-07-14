#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <pybind11/numpy.h>
#include <iostream>

using torch::Tensor;

static const int commonBlockSize = 128;

inline unsigned int GetGridSize(unsigned int blockSize, unsigned int threadSize)
{
    return (threadSize - 1) / blockSize + 1;
}

struct RayInfo
{
    unsigned int sdfCount;
    unsigned int sdfOffset;
    unsigned int renderCount;
    unsigned int renderOffset;
};

struct vec3f
{
    float x, y, z;
    __device__ __forceinline__ vec3f() : x(0.0f), y(0.0f), z(0.0f) {}
    __device__ __forceinline__ vec3f(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}
    __device__ __forceinline__ vec3f operator+(const float &a)
    {
        return vec3f(x + a, y + a, z + a);
    }
    __device__ __forceinline__ vec3f operator+(const vec3f &a)
    {
        return vec3f(x + a.x, y + a.y, z + a.z);
    }
    __device__ __forceinline__ vec3f operator-(const float &a)
    {
        return vec3f(x - +a, y - +a, z + -a);
    }
    __device__ __forceinline__ vec3f operator-(const vec3f &a)
    {
        return vec3f(x - a.x, y - a.y, z - a.z);
    }
    __device__ __forceinline__ vec3f operator/(const float &a)
    {
        return vec3f(x / a, y / a, z / a);
    }
    __device__ __forceinline__ vec3f operator*(const float &a)
    {
        return vec3f(x * a, y * a, z * a);
    }
    __device__ __forceinline__ vec3f operator*(const vec3f &a)
    {
        return vec3f(x * a.x, y * a.y, z * a.z);
    }
    __device__ __forceinline__ vec3f &Normalize()
    {
        float n = Norm();
        x /= n;
        y /= n;
        z /= n;
        return *this;
    }
    __device__ __forceinline__ float Norm()
    {
        return sqrtf(x * x + y * y + z * z);
    }
    __device__ __forceinline__ float Sum()
    {
        return x + y + z;
    }
};