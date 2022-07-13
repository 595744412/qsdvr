#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h>

struct CameraArgs
{
    unsigned int width;
    unsigned int height;
    float pos[3];
    float r[9];
    float fx;
    float fy;
    float cx;
    float cy;
    float near;
    float far;
    float scale[3];
};

torch::Tensor GenerateRayPoints(CameraArgs &args, float3 *dir, float3 *origin, float *adaptiveInterval, RayInfo *rayInfo, float interval, unsigned int pixelCount);