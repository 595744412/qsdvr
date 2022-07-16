#pragma once
#include <common.h>

struct CameraArgs
{
    unsigned int width;
    unsigned int height;
    float rotation[9];
    vec3f translation;
    vec3f scale;
    float fx;
    float fy;
    float cx;
    float cy;
    float near;
    float far;
};

struct PyCameraArgs
{
    unsigned int width;
    unsigned int height;
    pybind11::array_t<float> rotation;
    pybind11::array_t<float> translation;
    float fx;
    float fy;
    float cx;
    float cy;
    float near;
    float far;
    unsigned int pixelCount;
    unsigned int sdfCount;
    unsigned int renderCount;
};

inline CameraArgs ToCameraArgs(PyCameraArgs &pyargs)
{
    CameraArgs args;
    args.width = pyargs.width;
    args.height = pyargs.height;
    float *temp = (float *)pyargs.rotation.request().ptr;
    for (int i = 0; i < 9; i += 1)
    {
        args.rotation[i] = temp[i];
    }
    temp = (float *)pyargs.translation.request().ptr;
    args.translation = {temp[0], temp[1], temp[2]};
    args.fx = pyargs.fx;
    args.fy = pyargs.fy;
    args.cx = pyargs.cx;
    args.cy = pyargs.cy;
    args.near = pyargs.near;
    args.far = pyargs.far;
    return args;
}

struct Range
{
    float min;
    float max;
};

void RayCut(PyCameraArgs &args, Tensor &mask, Tensor &rayList, Tensor &originList, Tensor &rangeList, Tensor &dirList, float interval);

void GenerateRayPoints(PyCameraArgs &args, Tensor &sdfPointList, Tensor &sdfIndexList, Tensor &renderPointList, Tensor &renderIndexList, Tensor &viewDirList, Tensor &rayList, Tensor &originList, Tensor &rangeList, Tensor &dirList, float interval, unsigned int baseVolumeReso);