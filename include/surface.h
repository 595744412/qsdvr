#pragma once
#include <common.h>

void LayerToGridForward(Tensor &out, Tensor &xLayer, Tensor &yLayer, Tensor &zLayer, Tensor &offset);

void SampleSDFForward(Tensor &out, Tensor &sdfGrid, Tensor &pointList, Tensor &indexList);

void SampleNormalForward(Tensor &out, Tensor &sdfGrid, Tensor &pointList, Tensor &indexList);