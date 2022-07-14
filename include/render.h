#pragma once
#include <common.h>

void GridInterpolationForward(Tensor &dataGrid, Tensor &pointList, Tensor &indexList, Tensor &out, unsigned int reso);

Tensor GridInterpolationBackward(Tensor, Tensor);

void ShaderForward(Tensor &out, Tensor &normalList, Tensor &viewDirList, Tensor &dataList);

void RayAggregateForward(Tensor &out, Tensor &rgbList, Tensor &sdfList, Tensor &rayList, float logisticCoef);
