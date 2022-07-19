#pragma once
#include <common.h>

void GridInterpolationForward(Tensor &dataGrid, Tensor &pointList, Tensor &indexList, Tensor &out, unsigned int reso);

void GridInterpolationBackward(Tensor &dataGrid, Tensor &pointList, Tensor &indexList, Tensor &out, unsigned int reso);

void ShaderForward(Tensor &out, Tensor &normalList, Tensor &viewDirList, Tensor &dataList, Tensor &specularList);

void ShaderBackward(Tensor &out, Tensor &normalList, Tensor &viewDirList, Tensor &dataList, Tensor &specularList, Tensor &normalGradList, Tensor &dataGradList);

void RayAggregateForward(Tensor &out, Tensor &rgbList, Tensor &sdfList, Tensor &rayList, Tensor &alphaList, Tensor &TList, float logisticCoef);

void RayAggregateBackward(Tensor &outgrad, Tensor &rgbList, Tensor &sdfList, Tensor &rayList, Tensor &alphaList, Tensor &TList, Tensor &rgbGradList, Tensor &sdfGradList, float logisticCoef);
