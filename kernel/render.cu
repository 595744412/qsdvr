#include <render.h>

const unsigned int dataCount = 32;

__device__ __constant__ const float g_SHFactor[] =
    {
        0.28209479177387814347403972578039f,
        0.48860251190291992158638462283835f,
        0.48860251190291992158638462283835f,
        0.48860251190291992158638462283835f,
        1.0925484305920790705433857058027f,
        1.0925484305920790705433857058027f,
        0.31539156525252000603089369029571f,
        1.0925484305920790705433857058027f,
        0.54627421529603953527169285290135f};

__device__ __forceinline__ vec3f GetSH3Irradiance(vec3f v, vec3f coef[9])
{
    float x_4 = v.x * v.z;
    float zz = v.z * v.z;
    float xx = v.x * v.x;
    float x_5 = v.x * v.y;
    float x_6 = 2.0 * v.y * v.y - zz - xx;
    float x_7 = v.y * v.z;
    float x_8 = zz - xx;
    float x =
        g_SHFactor[0] * coef[0].x +
        g_SHFactor[1] * coef[1].x * v.x +
        g_SHFactor[2] * coef[2].x * v.y +
        g_SHFactor[3] * coef[3].x * v.z +
        g_SHFactor[4] * coef[4].x * x_4 +
        g_SHFactor[5] * coef[5].x * x_5 +
        g_SHFactor[6] * coef[6].x * x_6 +
        g_SHFactor[7] * coef[7].x * x_7 +
        g_SHFactor[8] * coef[8].x * x_8;

    float y =
        g_SHFactor[0] * coef[0].y +
        g_SHFactor[1] * coef[1].y * v.x +
        g_SHFactor[2] * coef[2].y * v.y +
        g_SHFactor[3] * coef[3].y * v.z +
        g_SHFactor[4] * coef[4].y * x_4 +
        g_SHFactor[5] * coef[5].y * x_5 +
        g_SHFactor[6] * coef[6].y * x_6 +
        g_SHFactor[7] * coef[7].y * x_7 +
        g_SHFactor[8] * coef[8].y * x_8;

    float z =
        g_SHFactor[0] * coef[0].z +
        g_SHFactor[1] * coef[1].z * v.x +
        g_SHFactor[2] * coef[2].z * v.y +
        g_SHFactor[3] * coef[3].z * v.z +
        g_SHFactor[4] * coef[4].z * x_4 +
        g_SHFactor[5] * coef[5].z * x_5 +
        g_SHFactor[6] * coef[6].z * x_6 +
        g_SHFactor[7] * coef[7].z * x_7 +
        g_SHFactor[8] * coef[8].z * x_8;
    return vec3f{x, y, z};
}

__device__ __forceinline__ void GetSH3IrradianceBackward(vec3f out_grad, vec3f v, vec3f coef[9], vec3f &v_grad, vec3f coef_grad[9])
{
    float x_4 = v.x * v.z;
    float zz = v.z * v.z;
    float xx = v.x * v.x;
    float x_5 = v.x * v.y;
    float x_6 = 2.0 * v.y * v.y - zz - xx;
    float x_7 = v.y * v.z;
    float x_8 = zz - xx;
    float gc[8] = {
        g_SHFactor[1] * (coef[1].x * out_grad.x + coef[1].y * out_grad.y + coef[1].z * out_grad.z),
        g_SHFactor[2] * (coef[2].x * out_grad.x + coef[2].y * out_grad.y + coef[2].z * out_grad.z),
        g_SHFactor[3] * (coef[3].x * out_grad.x + coef[3].y * out_grad.y + coef[3].z * out_grad.z),
        g_SHFactor[4] * (coef[4].x * out_grad.x + coef[4].y * out_grad.y + coef[4].z * out_grad.z),
        g_SHFactor[5] * (coef[5].x * out_grad.x + coef[5].y * out_grad.y + coef[5].z * out_grad.z),
        g_SHFactor[6] * (coef[6].x * out_grad.x + coef[6].y * out_grad.y + coef[6].z * out_grad.z),
        g_SHFactor[7] * (coef[7].x * out_grad.x + coef[7].y * out_grad.y + coef[7].z * out_grad.z),
        g_SHFactor[8] * (coef[8].x * out_grad.x + coef[8].y * out_grad.y + coef[8].z * out_grad.z)};
    v_grad.x = gc[0] + v.z * gc[3] + v.y * gc[4] - 2 * v.x * (gc[5] + gc[7]);
    v_grad.y = gc[1] + v.z * gc[6] + v.x * gc[4] + 4 * v.y * gc[5];
    v_grad.z = gc[2] + v.x * gc[3] + v.y * gc[6] + 2 * v.z * (gc[7] - gc[5]);
    coef_grad[0] = out_grad * g_SHFactor[0];
    coef_grad[1] = out_grad * (g_SHFactor[1] * v.x);
    coef_grad[2] = out_grad * (g_SHFactor[2] * v.y);
    coef_grad[3] = out_grad * (g_SHFactor[3] * v.z);
    coef_grad[4] = out_grad * (g_SHFactor[4] * x_4);
    coef_grad[5] = out_grad * (g_SHFactor[5] * x_5);
    coef_grad[6] = out_grad * (g_SHFactor[6] * x_6);
    coef_grad[7] = out_grad * (g_SHFactor[7] * x_7);
    coef_grad[8] = out_grad * (g_SHFactor[8] * x_8);
}

__device__ __forceinline__ float Interpolation1D(float a, float b, float x)
{
    return a * (1 - x) + b * x;
}

__global__ void GridInterpolationForwardKernel(float *out, float *dataGrid, vec3f *pointList, int *indexList, const unsigned int theardCount, unsigned int reso)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= theardCount)
    {
        return;
    }
    const int nid = id % dataCount;
    const int pid = id / dataCount;
    vec3f p = pointList[pid];
    int i000 = indexList[pid] * dataCount + nid;
    int i010 = i000 + reso * dataCount;
    int i100 = i000 + reso * reso * dataCount;
    int i110 = i100 + reso * dataCount;
    float a00 = Interpolation1D(dataGrid[i000], dataGrid[i000 + 1], p.x);
    float a01 = Interpolation1D(dataGrid[i010], dataGrid[i010 + 1], p.x);
    float a0 = Interpolation1D(a00, a01, p.y);
    float a10 = Interpolation1D(dataGrid[i100], dataGrid[i100 + 1], p.x);
    float a11 = Interpolation1D(dataGrid[i110], dataGrid[i110 + 1], p.x);
    float a1 = Interpolation1D(a10, a11, p.y);
    out[id] = Interpolation1D(a0, a1, p.z);
}

__global__ void ShaderForwardKernel(vec3f *out, vec3f *normalList, vec3f *viewDirList, float *dataList, const unsigned int theardCount)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= theardCount)
    {
        return;
    }
    vec3f normal = normalList[id];
    vec3f view = viewDirList[id];
    unsigned int offset = id * 32;
    vec3f specular[9];
#pragma unroll
    for (int i = 0; i < 9; i += 1)
    {
        specular[i] = {dataList[offset + i], dataList[offset + i + 1], dataList[offset + i + 2]};
    }
    vec3f diffuse(dataList[offset + 27], dataList[offset + 28], dataList[offset + 29]);
    float metallic = dataList[offset + 30];
    float ao = dataList[offset + 31];
    float vdotn = view.x * normal.x + view.y * normal.y + view.z * normal.z;
    vec3f reflect = view - normal * (2.0f * vdotn);
    vec3f color = diffuse * (1.0f - metallic) + GetSH3Irradiance(reflect, specular) * (metallic + (1.0f - metallic) * 0.04f);
    out[id] = color * ao;
}

__global__ void RayAggregateForwardKernel(vec3f *out, vec3f *rgbList, float *sdfList, RayInfo *rayList, float logisticCoef, const unsigned int theardCount)
{
    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= theardCount)
    {
        return;
    }
    RayInfo info = rayList[id];
    const unsigned int sdfOffset = info.sdfOffset;
    const unsigned int renderOffset = info.renderOffset;
    vec3f color;
    if (info.sdfCount == 0)
    {
        color = {0.0f, 0.0f, 0.0f};
    }
    else
    {

        float exp_SDF_i = expf(-logisticCoef * sdfList[sdfOffset]);
        float T = 1.0f;
        for (unsigned int i = 1; i < info.sdfCount; i += 1)
        {
            float exp_SDF_i_1 = expf(-logisticCoef * sdfList[sdfOffset + i]);
            float alpha = fmaxf((exp_SDF_i_1 - exp_SDF_i) / (1.0f + exp_SDF_i_1), 0.0f);
            color = color + rgbList[renderOffset + i] * alpha * T;
            exp_SDF_i = exp_SDF_i_1;
            T *= (1 - alpha);
        }
    }
    __syncthreads();
    out[id] = color;
}

void GridInterpolationForward(Tensor &dataGrid, Tensor &pointList, Tensor &indexList, Tensor &out, unsigned int reso)
{
    const unsigned int theardCount = dataCount * pointList.size(0);
    const unsigned int blockSize = commonBlockSize;
    const unsigned int gridSize = GetGridSize(blockSize, theardCount);
    GridInterpolationForwardKernel<<<gridSize, blockSize>>>(out.data<float>(), dataGrid.data<float>(), (vec3f *)pointList.data<float>(), indexList.data<int>(), theardCount, reso);
}

void ShaderForward(Tensor &out, Tensor &normalList, Tensor &viewDirList, Tensor &dataList)
{
    const unsigned int theardCount = normalList.size(0);
    const unsigned int blockSize = commonBlockSize;
    const unsigned int gridSize = GetGridSize(blockSize, theardCount);
    ShaderForwardKernel<<<gridSize, blockSize>>>((vec3f *)out.data<float>(), (vec3f *)normalList.data<float>(), (vec3f *)viewDirList.data<float>(), dataList.data<float>(), theardCount);
}

void RayAggregateForward(Tensor &out, Tensor &rgbList, Tensor &sdfList, Tensor &rayList, float logisticCoef)
{
    const unsigned int theardCount = rayList.size(0);
    const unsigned int blockSize = commonBlockSize;
    const unsigned int gridSize = GetGridSize(blockSize, theardCount);
    RayAggregateForwardKernel<<<gridSize, blockSize>>>((vec3f *)out.data<float>(), (vec3f *)rgbList.data<float>(), sdfList.data<float>(), (RayInfo *)rayList.data<int>(), logisticCoef, theardCount);
}
