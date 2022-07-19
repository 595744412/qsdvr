#include <camera.h>

__device__ __forceinline__ void TransCameraToWorld(CameraArgs &camera, vec3f &p)
{
    float x = (p.x + camera.cx) / camera.fx * p.z;
    float y = (p.y + camera.cy) / camera.fy * p.z;
    p.x = camera.rotation[0] * x + camera.rotation[1] * y + camera.rotation[2] * p.z + camera.translation.x;
    p.y = camera.rotation[3] * x + camera.rotation[4] * y + camera.rotation[5] * p.z + camera.translation.y;
    p.z = camera.rotation[6] * x + camera.rotation[7] * y + camera.rotation[8] * p.z + camera.translation.z;
}

__global__ void RayCutKernel(CameraArgs camera, vec3f *dirCuda, vec3f *originCuda, RayInfo *rayCuda, float interval, unsigned int pixelCount)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= pixelCount)
    {
        return;
    }
    float x = (float)(id % camera.width) + 0.5f;
    float y = (float)(id / camera.width) + 0.5f;
    vec3f nearPoint(x, y, camera.near);
    vec3f farPoint(x, y, camera.far);
    TransCameraToWorld(camera, nearPoint);
    TransCameraToWorld(camera, farPoint);
    vec3f dir = (farPoint - nearPoint).Normalize();
    float end;
    if (dir.x != 0.0f)
    {
        end = (farPoint.x - nearPoint.x) / dir.x;
    }
    else if (dir.y != 0.0f)
    {
        end = (farPoint.y - nearPoint.y) / dir.y;
    }
    else
    {
        end = (farPoint.z - nearPoint.z) / dir.z;
    }

    unsigned int tempRayPointCount = end / interval;
    unsigned int rayPointCount = 0;
    vec3f newOri;
    for (unsigned int j = 0; j < tempRayPointCount; j += 1)
    {
        vec3f p = nearPoint + dir * (float)j * interval;
        if (p.x < 0.9999f && p.x > -0.9999f && p.y < 0.9999f && p.y > -0.9999f && p.z < 0.9999f && p.z > -0.9999f)
        {
            if (rayPointCount == 0)
            {
                newOri = p;
            }
            rayPointCount += 1;
        }
    }
    dirCuda[id] = dir;
    originCuda[id] = newOri;
    rayCuda[id].sdfCount = rayPointCount;
}

__global__ void GenerateRayPointsKernel(vec3f *sdfPointList, vec3f *renderPointList, int *sdfIndexList, int *renderIndexList, vec3f *viewDirList, RayInfo *rayInfoListCudaHost, vec3f *dirCuda, vec3f *originCuda, unsigned int theardCount, unsigned int baseVolumeReso, float volumeOffset, unsigned int z_offset, float interval)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= theardCount)
    {
        return;
    }
    vec3f dir = dirCuda[id];
    vec3f ori = originCuda[id];
    unsigned int sdfCount = rayInfoListCudaHost[id].sdfCount;
    unsigned int sdfOffset = rayInfoListCudaHost[id].sdfOffset;
    for (unsigned int j = 0; j < sdfCount; j += 1)
    {
        vec3f p = (ori + dir * (float)j * interval + 1.0f) * volumeOffset;
        int x_0 = floorf(p.x);
        int y_0 = floorf(p.y);
        int z_0 = floorf(p.z);
        vec3f sdfPoint = p * 2.0f - vec3f(2 * x_0 + 1.0f, 2 * y_0 + 1.0f, 2 * z_0 + 1.0f);
        int index = x_0 + y_0 * baseVolumeReso + z_0 * z_offset;
        sdfPointList[sdfOffset + j] = sdfPoint;
        sdfIndexList[sdfOffset + j] = index;
    }
    unsigned int renderCount = rayInfoListCudaHost[id].renderCount;
    unsigned int renderOffset = rayInfoListCudaHost[id].renderOffset;
    for (unsigned int j = 0; j < renderCount; j += 1)
    {
        vec3f p = (ori + dir * (((float)j + 0.5f) * interval) + 1.0f) * volumeOffset;
        int x_0 = floorf(p.x);
        int y_0 = floorf(p.y);
        int z_0 = floorf(p.z);
        vec3f renderPoint = p * 2.0f - vec3f(2 * x_0 + 1.0f, 2 * y_0 + 1.0f, 2 * z_0 + 1.0f);
        int index = x_0 + y_0 * baseVolumeReso + z_0 * z_offset;
        renderPointList[renderOffset + j] = renderPoint;
        renderIndexList[renderOffset + j] = index;
        viewDirList[renderOffset + j] = dir;
    }
}

void RayCut(PyCameraArgs &args, Tensor &mask, Tensor &rayList, Tensor &originList, Tensor &dirList, float interval)
{
    RayInfo *rayCuda = (RayInfo *)rayList.data<int>();
    vec3f *originCuda = (vec3f *)originList.data<float>();
    vec3f *dirCuda = (vec3f *)dirList.data<float>();
    RayInfo *rayCPU = new RayInfo[args.pixelCount];
    bool *maskCPU = (bool *)mask.data<bool>();

    const unsigned int blockSize = commonBlockSize;
    const unsigned int gridSize = GetGridSize(blockSize, args.pixelCount);
    RayCutKernel<<<gridSize, blockSize>>>(ToCameraArgs(args), dirCuda, originCuda, rayCuda, interval, args.pixelCount);

    cudaMemcpy(rayCPU, rayCuda, sizeof(RayInfo) * args.pixelCount, cudaMemcpyDeviceToHost);
    args.sdfCount = 0;
    args.renderCount = 0;
    for (unsigned int i = 0; i < args.pixelCount; i += 1)
    {
        if (maskCPU[i])
        {
            if (rayCPU[i].sdfCount < 2)
            {
                rayCPU[i].sdfCount = 0;
            }
            rayCPU[i].sdfOffset = args.sdfCount;
            args.sdfCount += rayCPU[i].sdfCount;
            rayCPU[i].renderOffset = args.renderCount;
            rayCPU[i].renderCount = rayCPU[i].sdfCount > 1 ? rayCPU[i].sdfCount - 1 : 0;
            args.renderCount += rayCPU[i].renderCount;
        }
    }
    cudaMemcpy(rayCuda, rayCPU, sizeof(RayInfo) * args.pixelCount, cudaMemcpyHostToDevice);
    delete[] rayCPU;
}

void GenerateRayPoints(PyCameraArgs &args, Tensor &sdfPointList, Tensor &sdfIndexList, Tensor &renderPointList, Tensor &renderIndexList, Tensor &viewDirList, Tensor &rayList, Tensor &originList, Tensor &dirList, float interval, unsigned int baseVolumeReso)
{
    RayInfo *rayCuda = (RayInfo *)rayList.data<int>();
    vec3f *originCuda = (vec3f *)originList.data<float>();
    vec3f *dirCuda = (vec3f *)dirList.data<float>();
    vec3f *sdfPointListCuda = (vec3f *)sdfPointList.data<float>();
    int *sdfIndexListCuda = (int *)sdfIndexList.data<int>();
    vec3f *renderPointListCuda = (vec3f *)renderPointList.data<float>();
    int *renderIndexListCuda = (int *)renderIndexList.data<int>();
    vec3f *viewDirListCuda = (vec3f *)viewDirList.data<float>();

    const unsigned int theardCount = rayList.size(0);
    const unsigned int blockSize = commonBlockSize;
    const unsigned int gridSize = GetGridSize(blockSize, theardCount);
    GenerateRayPointsKernel<<<gridSize, blockSize>>>(sdfPointListCuda, renderPointListCuda, sdfIndexListCuda, renderIndexListCuda, viewDirListCuda, rayCuda, dirCuda, originCuda, theardCount, baseVolumeReso, baseVolumeReso / 2.0f, baseVolumeReso * baseVolumeReso, interval);
}