#include <surface.h>

__global__ void LayerToGridForwardKernel(float *out, float *xLayer_a0Cuda, float *yLayer_a1Cuda,
                                         float *zLayer_a2Cuda, float *xLayer_a3Cuda, float *yLayer_a4Cuda,
                                         float *zLayer_a5Cuda, float *xLayer_a6Cuda, float *yLayer_a6Cuda,
                                         float *zLayer_a6Cuda, float offset_a6, const unsigned int theardCount, unsigned int reso)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= theardCount)
    {
        return;
    }
    const int x = id % reso;
    const int y = (id / reso) % reso;
    const int z = id / (reso * reso);
    id *= 7;
    out[id] = xLayer_a0Cuda[x];
    out[id + 1] = yLayer_a1Cuda[y];
    out[id + 2] = zLayer_a2Cuda[z];
    out[id + 3] = xLayer_a3Cuda[x];
    out[id + 4] = yLayer_a4Cuda[y];
    out[id + 5] = zLayer_a5Cuda[z];
    out[id + 6] = xLayer_a6Cuda[x] + yLayer_a6Cuda[y] + zLayer_a6Cuda[z] + offset_a6;
}

__global__ void LayerToGridBackwardKernel(float *out, float *xLayer_a0Cuda, float *yLayer_a1Cuda,
                                          float *zLayer_a2Cuda, float *xLayer_a3Cuda, float *yLayer_a4Cuda,
                                          float *zLayer_a5Cuda, float *xLayer_a6Cuda, float *yLayer_a6Cuda,
                                          float *zLayer_a6Cuda, float *offset_a6, const unsigned int theardCount, unsigned int reso)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= theardCount)
    {
        return;
    }
    const int x = id % reso;
    const int y = (id / reso) % reso;
    const int z = id / (reso * reso);
    id *= 7;
    float outGrad[7];
    memcpy(outGrad, out + id, sizeof(float) * 7);
    atomicAdd(xLayer_a0Cuda + x, outGrad[0]);
    atomicAdd(yLayer_a1Cuda + y, outGrad[1]);
    atomicAdd(zLayer_a2Cuda + z, outGrad[2]);
    atomicAdd(xLayer_a3Cuda + x, outGrad[3]);
    atomicAdd(yLayer_a4Cuda + y, outGrad[4]);
    atomicAdd(zLayer_a5Cuda + z, outGrad[5]);
    atomicAdd(xLayer_a6Cuda + x, outGrad[6]);
    atomicAdd(yLayer_a6Cuda + y, outGrad[6]);
    atomicAdd(zLayer_a6Cuda + z, outGrad[6]);
    atomicAdd(offset_a6, outGrad[6]);
}

__global__ void SampleSDFForwardKernel(float *out, float *sdfGrid, vec3f *pointList, int *indexList, const unsigned int theardCount, float reso)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= theardCount)
    {
        return;
    }
    int index = indexList[id];
    vec3f p = pointList[id];
    index *= 7;
    float a0 = sdfGrid[index] * p.x;
    float a1 = sdfGrid[index + 1] * p.y;
    float a2 = sdfGrid[index + 2] * p.z;
    float a3 = sdfGrid[index + 3];
    float a4 = sdfGrid[index + 4];
    float a5 = sdfGrid[index + 5];
    float a6 = sdfGrid[index + 6];
    out[id] = ((a0 + a3) * p.x + (a1 + a4) * p.y + (a2 + a5) * p.z + a6) / vec3f(2 * a0 + a3, 2 * a1 + a4, 2 * a2 + a5).Norm() / reso;
}

__global__ void SampleSDFBackwardKernel(float *out, float *sdfGrid, vec3f *pointList, int *indexList, float *sdfGradGrid, const unsigned int theardCount, float reso)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= theardCount)
    {
        return;
    }
    int index = indexList[id];
    vec3f p = pointList[id];
    index *= 7;
    float a0 = sdfGrid[index] * p.x;
    float a1 = sdfGrid[index + 1] * p.y;
    float a2 = sdfGrid[index + 2] * p.z;
    float a3 = sdfGrid[index + 3];
    float a4 = sdfGrid[index + 4];
    float a5 = sdfGrid[index + 5];
    float a6 = sdfGrid[index + 6];
    float value = (a0 + a3) * p.x + (a1 + a4) * p.y + (a2 + a5) * p.z + a6;
    float norm = 1.0f / vec3f(2 * a0 + a3, 2 * a1 + a4, 2 * a2 + a5).Norm();
    float norm3 = value * norm * norm * norm;
    float outGrad = out[id] / reso;
    atomicAdd(sdfGradGrid + index, outGrad * p.x * (p.x * norm - 2 * (2 * a0 + a3) * norm3));
    atomicAdd(sdfGradGrid + index + 1, outGrad * p.y * (p.y * norm - 2 * (2 * a1 + a4) * norm3));
    atomicAdd(sdfGradGrid + index + 2, outGrad * p.z * (p.z * norm - 2 * (2 * a2 + a5) * norm3));
    atomicAdd(sdfGradGrid + index + 3, outGrad * (p.x * norm - (2 * a0 + a3) * norm3));
    atomicAdd(sdfGradGrid + index + 4, outGrad * (p.y * norm - (2 * a1 + a4) * norm3));
    atomicAdd(sdfGradGrid + index + 5, outGrad * (p.z * norm - (2 * a2 + a5) * norm3));
    atomicAdd(sdfGradGrid + index + 6, outGrad * norm);
}

__global__ void SampleNormalForwardKernel(vec3f *out, float *sdfGrid, vec3f *pointList, int *indexList, const unsigned int theardCount)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= theardCount)
    {
        return;
    }
    int index = indexList[id];
    vec3f p = pointList[id];
    index *= 7;
    out[id] = vec3f(2 * sdfGrid[index] * p.x + sdfGrid[index + 3], 2 * sdfGrid[index + 1] * p.y + sdfGrid[index + 4], 2 * sdfGrid[index + 2] * p.z + sdfGrid[index + 5]).Normalize();
}

__global__ void SampleNormalBackwardKernel(vec3f *out, float *sdfGrid, vec3f *pointList, int *indexList, float *sdfGradGrid, const unsigned int theardCount)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= theardCount)
    {
        return;
    }
    int index = indexList[id];
    vec3f p = pointList[id];
    index *= 7;
    float a0 = 2 * sdfGrid[index] * p.x;
    float a1 = 2 * sdfGrid[index + 1] * p.y;
    float a2 = 2 * sdfGrid[index + 2] * p.z;
    float a3 = sdfGrid[index + 3];
    float a4 = sdfGrid[index + 4];
    float a5 = sdfGrid[index + 5];
    float norm = 1.0f / vec3f(a0 + a3, a1 + a4, a2 + a5).Norm();
    float norm3 = norm * norm * norm;

    vec3f outGrad = out[id];
    float gx = outGrad.x * (norm - norm3 * (a0 + a3) * (a0 + a3)) - outGrad.y * (a1 + a4) * norm3 * (a0 + a3) - outGrad.z * (a2 + a5) * norm3 * (a0 + a3);
    float gy = outGrad.y * (norm - norm3 * (a1 + a4) * (a1 + a4)) - outGrad.x * (a0 + a3) * norm3 * (a1 + a4) - outGrad.z * (a2 + a5) * norm3 * (a1 + a4);
    float gz = outGrad.z * (norm - norm3 * (a2 + a5) * (a2 + a5)) - outGrad.y * (a1 + a4) * norm3 * (a2 + a5) - outGrad.x * (a0 + a3) * norm3 * (a2 + a5);
    atomicAdd(sdfGradGrid + index, 2.0f * p.x * gx);
    atomicAdd(sdfGradGrid + index + 1, 2.0f * p.y * gy);
    atomicAdd(sdfGradGrid + index + 2, 2.0f * p.z * gz);
    atomicAdd(sdfGradGrid + index + 3, gx);
    atomicAdd(sdfGradGrid + index + 4, gy);
    atomicAdd(sdfGradGrid + index + 5, gz);
}

void LayerToGridForward(Tensor &out, Tensor &xLayer, Tensor &yLayer, Tensor &zLayer, Tensor &offset)
{
    unsigned int reso = xLayer.size(0);
    float *xLayer_a0 = xLayer.data<float>();
    float *yLayer_a1 = yLayer.data<float>();
    float *zLayer_a2 = zLayer.data<float>();
    float *offset_ = offset.data<float>();
    float *xLayer_a3 = new float[reso];
    float *yLayer_a4 = new float[reso];
    float *zLayer_a5 = new float[reso];
    float *xLayer_a6 = new float[reso];
    float *yLayer_a6 = new float[reso];
    float *zLayer_a6 = new float[reso];
    xLayer_a3[0] = offset_[0];
    yLayer_a4[0] = offset_[1];
    zLayer_a5[0] = offset_[2];
    xLayer_a6[0] = 0;
    yLayer_a6[0] = 0;
    zLayer_a6[0] = 0;
    for (unsigned int i = 1; i < reso; i += 1)
    {
        xLayer_a3[i] = 2 * xLayer_a0[i - 1] + 2 * xLayer_a0[i] + xLayer_a3[i - 1];
        yLayer_a4[i] = 2 * yLayer_a1[i - 1] + 2 * yLayer_a1[i] + yLayer_a4[i - 1];
        zLayer_a5[i] = 2 * zLayer_a2[i - 1] + 2 * zLayer_a2[i] + zLayer_a5[i - 1];
    }
    for (unsigned int i = 1; i < reso; i += 1)
    {
        xLayer_a6[i] = 3 * xLayer_a0[i - 1] + xLayer_a0[i] + 2 * xLayer_a3[i - 1] + xLayer_a6[i - 1];
        yLayer_a6[i] = 3 * yLayer_a1[i - 1] + yLayer_a1[i] + 2 * yLayer_a4[i - 1] + yLayer_a6[i - 1];
        zLayer_a6[i] = 3 * zLayer_a2[i - 1] + zLayer_a2[i] + 2 * zLayer_a5[i - 1] + zLayer_a6[i - 1];
    }
    float *xLayer_a0Cuda;
    float *yLayer_a1Cuda;
    float *zLayer_a2Cuda;
    float *xLayer_a3Cuda;
    float *yLayer_a4Cuda;
    float *zLayer_a5Cuda;
    float *xLayer_a6Cuda;
    float *yLayer_a6Cuda;
    float *zLayer_a6Cuda;
    size_t l = reso * sizeof(float);
    cudaMalloc((void **)&xLayer_a0Cuda, l);
    cudaMalloc((void **)&yLayer_a1Cuda, l);
    cudaMalloc((void **)&zLayer_a2Cuda, l);
    cudaMalloc((void **)&xLayer_a3Cuda, l);
    cudaMalloc((void **)&yLayer_a4Cuda, l);
    cudaMalloc((void **)&zLayer_a5Cuda, l);
    cudaMalloc((void **)&xLayer_a6Cuda, l);
    cudaMalloc((void **)&yLayer_a6Cuda, l);
    cudaMalloc((void **)&zLayer_a6Cuda, l);
    cudaMemcpy(xLayer_a0Cuda, xLayer_a0, l, cudaMemcpyHostToDevice);
    cudaMemcpy(yLayer_a1Cuda, yLayer_a1, l, cudaMemcpyHostToDevice);
    cudaMemcpy(zLayer_a2Cuda, zLayer_a2, l, cudaMemcpyHostToDevice);
    cudaMemcpy(xLayer_a3Cuda, xLayer_a3, l, cudaMemcpyHostToDevice);
    cudaMemcpy(yLayer_a4Cuda, yLayer_a4, l, cudaMemcpyHostToDevice);
    cudaMemcpy(zLayer_a5Cuda, zLayer_a5, l, cudaMemcpyHostToDevice);
    cudaMemcpy(xLayer_a6Cuda, xLayer_a6, l, cudaMemcpyHostToDevice);
    cudaMemcpy(yLayer_a6Cuda, yLayer_a6, l, cudaMemcpyHostToDevice);
    cudaMemcpy(zLayer_a6Cuda, zLayer_a6, l, cudaMemcpyHostToDevice);
    const unsigned int theardCount = reso * reso * reso;
    const unsigned int blockSize = commonBlockSize;
    const unsigned int gridSize = GetGridSize(blockSize, theardCount);
    LayerToGridForwardKernel<<<gridSize, blockSize>>>(out.data<float>(), xLayer_a0Cuda, yLayer_a1Cuda,
                                                      zLayer_a2Cuda, xLayer_a3Cuda, yLayer_a4Cuda,
                                                      zLayer_a5Cuda, xLayer_a6Cuda, yLayer_a6Cuda,
                                                      zLayer_a6Cuda, offset_[3], theardCount, reso);
    cudaFree(xLayer_a0Cuda);
    cudaFree(yLayer_a1Cuda);
    cudaFree(zLayer_a2Cuda);
    cudaFree(xLayer_a3Cuda);
    cudaFree(yLayer_a4Cuda);
    cudaFree(zLayer_a5Cuda);
    cudaFree(xLayer_a6Cuda);
    cudaFree(yLayer_a6Cuda);
    cudaFree(zLayer_a6Cuda);
    delete[] xLayer_a3;
    delete[] yLayer_a4;
    delete[] zLayer_a5;
    delete[] xLayer_a6;
    delete[] yLayer_a6;
    delete[] zLayer_a6;
}

void LayerToGridBackward(Tensor &outGrad, Tensor &xLayer, Tensor &yLayer, Tensor &zLayer, Tensor &offset)
{
    unsigned int reso = xLayer.size(0);
    float *xLayer_a0 = xLayer.data<float>();
    float *yLayer_a1 = yLayer.data<float>();
    float *zLayer_a2 = zLayer.data<float>();
    float *offset_ = offset.data<float>();
    float *xLayer_a3 = new float[reso];
    float *yLayer_a4 = new float[reso];
    float *zLayer_a5 = new float[reso];
    float *xLayer_a6 = new float[reso];
    float *yLayer_a6 = new float[reso];
    float *zLayer_a6 = new float[reso];
    float *xLayer_a0Cuda;
    float *yLayer_a1Cuda;
    float *zLayer_a2Cuda;
    float *xLayer_a3Cuda;
    float *yLayer_a4Cuda;
    float *zLayer_a5Cuda;
    float *xLayer_a6Cuda;
    float *yLayer_a6Cuda;
    float *zLayer_a6Cuda;
    float *offsetCuda;
    size_t l = reso * sizeof(float);
    cudaMalloc((void **)&offsetCuda, sizeof(float));
    cudaMalloc((void **)&xLayer_a0Cuda, l);
    cudaMalloc((void **)&yLayer_a1Cuda, l);
    cudaMalloc((void **)&zLayer_a2Cuda, l);
    cudaMalloc((void **)&xLayer_a3Cuda, l);
    cudaMalloc((void **)&yLayer_a4Cuda, l);
    cudaMalloc((void **)&zLayer_a5Cuda, l);
    cudaMalloc((void **)&xLayer_a6Cuda, l);
    cudaMalloc((void **)&yLayer_a6Cuda, l);
    cudaMalloc((void **)&zLayer_a6Cuda, l);
    cudaMemset(offsetCuda, 0, sizeof(float));
    cudaMemset(xLayer_a0Cuda, 0, l);
    cudaMemset(yLayer_a1Cuda, 0, l);
    cudaMemset(zLayer_a2Cuda, 0, l);
    cudaMemset(xLayer_a3Cuda, 0, l);
    cudaMemset(yLayer_a4Cuda, 0, l);
    cudaMemset(zLayer_a5Cuda, 0, l);
    cudaMemset(xLayer_a6Cuda, 0, l);
    cudaMemset(yLayer_a6Cuda, 0, l);
    cudaMemset(zLayer_a6Cuda, 0, l);
    const unsigned int theardCount = reso * reso * reso;
    const unsigned int blockSize = commonBlockSize;
    const unsigned int gridSize = GetGridSize(blockSize, theardCount);
    LayerToGridBackwardKernel<<<gridSize, blockSize>>>(outGrad.data<float>(), xLayer_a0Cuda, yLayer_a1Cuda,
                                                       zLayer_a2Cuda, xLayer_a3Cuda, yLayer_a4Cuda,
                                                       zLayer_a5Cuda, xLayer_a6Cuda, yLayer_a6Cuda,
                                                       zLayer_a6Cuda, offsetCuda, theardCount, reso);
    cudaMemcpy(&offset_[3], offsetCuda, l, cudaMemcpyDeviceToHost);
    cudaMemcpy(xLayer_a0, xLayer_a0Cuda, l, cudaMemcpyDeviceToHost);
    cudaMemcpy(yLayer_a1, yLayer_a1Cuda, l, cudaMemcpyDeviceToHost);
    cudaMemcpy(zLayer_a2, zLayer_a2Cuda, l, cudaMemcpyDeviceToHost);
    cudaMemcpy(xLayer_a3, xLayer_a3Cuda, l, cudaMemcpyDeviceToHost);
    cudaMemcpy(yLayer_a4, yLayer_a4Cuda, l, cudaMemcpyDeviceToHost);
    cudaMemcpy(zLayer_a5, zLayer_a5Cuda, l, cudaMemcpyDeviceToHost);
    cudaMemcpy(xLayer_a6, xLayer_a6Cuda, l, cudaMemcpyDeviceToHost);
    cudaMemcpy(yLayer_a6, yLayer_a6Cuda, l, cudaMemcpyDeviceToHost);
    cudaMemcpy(zLayer_a6, zLayer_a6Cuda, l, cudaMemcpyDeviceToHost);
    cudaFree(xLayer_a0Cuda);
    cudaFree(yLayer_a1Cuda);
    cudaFree(zLayer_a2Cuda);
    cudaFree(xLayer_a3Cuda);
    cudaFree(yLayer_a4Cuda);
    cudaFree(zLayer_a5Cuda);
    cudaFree(xLayer_a6Cuda);
    cudaFree(yLayer_a6Cuda);
    cudaFree(zLayer_a6Cuda);
    for (unsigned int i = reso - 1; i > 0; i -= 1)
    {
        xLayer_a0[i] += xLayer_a6[i];
        xLayer_a0[i - 1] += 3 * xLayer_a6[i];
        xLayer_a3[i - 1] += 2 * xLayer_a6[i];
        xLayer_a6[i - 1] += xLayer_a6[i];

        yLayer_a1[i] += yLayer_a6[i];
        yLayer_a1[i - 1] += 3 * yLayer_a6[i];
        yLayer_a4[i - 1] += 2 * yLayer_a6[i];
        yLayer_a6[i - 1] += yLayer_a6[i];

        zLayer_a2[i] += zLayer_a6[i];
        zLayer_a2[i - 1] += 3 * zLayer_a6[i];
        zLayer_a5[i - 1] += 2 * zLayer_a6[i];
        zLayer_a6[i - 1] += zLayer_a6[i];
    }
    for (unsigned int i = reso - 1; i > 0; i -= 1)
    {
        xLayer_a0[i] += 2 * xLayer_a3[i];
        xLayer_a0[i - 1] += 2 * xLayer_a3[i];
        xLayer_a3[i - 1] += xLayer_a3[i];

        yLayer_a1[i] += 2 * yLayer_a4[i];
        yLayer_a1[i - 1] += 2 * yLayer_a4[i];
        yLayer_a4[i - 1] += yLayer_a4[i];

        zLayer_a2[i] += 2 * zLayer_a5[i];
        zLayer_a2[i - 1] += 2 * zLayer_a5[i];
        zLayer_a5[i - 1] += zLayer_a5[i];
    }
    offset_[0] = xLayer_a3[0];
    offset_[1] = yLayer_a4[0];
    offset_[2] = zLayer_a5[0];
    delete[] xLayer_a3;
    delete[] yLayer_a4;
    delete[] zLayer_a5;
    delete[] xLayer_a6;
    delete[] yLayer_a6;
    delete[] zLayer_a6;
}

void SampleSDFForward(Tensor &out, Tensor &sdfGrid, Tensor &pointList, Tensor &indexList, float reso)
{
    const unsigned int theardCount = pointList.size(0);
    const unsigned int blockSize = commonBlockSize;
    const unsigned int gridSize = GetGridSize(blockSize, theardCount);
    SampleSDFForwardKernel<<<gridSize, blockSize>>>(out.data<float>(), sdfGrid.data<float>(), (vec3f *)pointList.data<float>(),
                                                    indexList.data<int>(), theardCount, reso);
}

void SampleSDFBackward(Tensor &out, Tensor &sdfGrid, Tensor &pointList, Tensor &indexList, Tensor &sdfGradGrid, float reso)
{
    const unsigned int theardCount = pointList.size(0);
    const unsigned int blockSize = commonBlockSize;
    const unsigned int gridSize = GetGridSize(blockSize, theardCount);
    SampleSDFBackwardKernel<<<gridSize, blockSize>>>(out.data<float>(), sdfGrid.data<float>(), (vec3f *)pointList.data<float>(),
                                                     indexList.data<int>(), sdfGradGrid.data<float>(), theardCount, reso);
}

void SampleNormalForward(Tensor &out, Tensor &sdfGrid, Tensor &pointList, Tensor &indexList)
{
    const unsigned int theardCount = pointList.size(0);
    const unsigned int blockSize = commonBlockSize;
    const unsigned int gridSize = GetGridSize(blockSize, theardCount);
    SampleNormalForwardKernel<<<gridSize, blockSize>>>((vec3f *)out.data<float>(), sdfGrid.data<float>(), (vec3f *)pointList.data<float>(), indexList.data<int>(), theardCount);
}

void SampleNormalBackward(Tensor &out, Tensor &sdfGrid, Tensor &pointList, Tensor &indexList, Tensor &sdfGradGrid)
{
    const unsigned int theardCount = pointList.size(0);
    const unsigned int blockSize = commonBlockSize;
    const unsigned int gridSize = GetGridSize(blockSize, theardCount);
    SampleNormalBackwardKernel<<<gridSize, blockSize>>>((vec3f *)out.data<float>(), sdfGrid.data<float>(), (vec3f *)pointList.data<float>(), indexList.data<int>(), sdfGradGrid.data<float>(), theardCount);
}
