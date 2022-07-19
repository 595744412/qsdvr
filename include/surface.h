#pragma once
#include <common.h>

void LayerToGridForward(Tensor &out, Tensor &xLayer, Tensor &yLayer, Tensor &zLayer, Tensor &offset);

void LayerToGridBackward(Tensor &out, Tensor &xLayer, Tensor &yLayer, Tensor &zLayer, Tensor &offset);

void SampleSDFForward(Tensor &out, Tensor &sdfGrid, Tensor &pointList, Tensor &indexList, float reso);

void SampleSDFBackward(Tensor &out, Tensor &sdfGrid, Tensor &pointList, Tensor &indexList, Tensor &sdfGradGrid, float reso);

void SampleNormalForward(Tensor &out, Tensor &sdfGrid, Tensor &pointList, Tensor &indexList);

void SampleNormalBackward(Tensor &out, Tensor &sdfGrid, Tensor &pointList, Tensor &indexList, Tensor &sdfGradGrid);