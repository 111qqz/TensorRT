/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "opParsers.h"

using namespace nvinfer1;

namespace nvcaffeparser1
{
ILayer* parseReshape(INetworkDefinition& network, const trtcaffe::LayerParameter& msg, CaffeWeightFactory& /*weightFactory*/, BlobNameToTensor& tensors)
{
    if (!checkBlobs(msg, 1, 1))
    {
        return nullptr;
    }

    const trtcaffe::ReshapeParameter& p = msg.reshape_param();
    Dims bottomDims = tensors[msg.bottom(0)]->getDimensions();
    int axis = p.has_axis() ? p.axis() : 0;

    const ::trtcaffe::BlobShape& shape = p.shape();
    // Check that N (batch dim) is 0. TensorRT does not support reshape in batch dimension
    //  第一位必须是0,写成和输入维度相同的值也不行．
    if ((axis == 0) && (shape.dim(0) != 0))
    {
        std::cout << "Caffe Parser: Invalid reshape param. TensorRT does not support reshape in N (batch) dimension" << std::endl;
        return nullptr;
    }

    // Handle axis and dims parameters
    //  注意　nbDims是不包括batch这个维度的，默认为3
    //  这代码有问题吧
    // １．默认情况，没有axes和num_axes的参数，处理区间为[0,bottomDims.nbDims),正确
    // ２．axis为正数，没有num_axes参数的情况． 逻辑比较奇怪，但是结果是正确的．（除了整除) 
    // 3.　axis为负数，没有num_axes参数的情况．　完全不正确． TO DO
    // 4. reshape param参数不够4个的情况
    // int axStart = std::max(0, axis - 1);
    int axStart = (axis >= 0) ? std::max(0, axis - 1):
      bottomDims.nbDims + axis + 1;
    //  这里axis-1的原因是，axis是按照caffe标准包含batch的，也就是0代表batch的维度
    //  但是Dims里面却是不包括batch的．　因此axis=1其实是实际Dims中的第0维．　
    // 　而axis=0的特殊情况 已经在　  if (axis == 0 && i == 0)　判掉了．

    int axEnd = p.has_num_axes() ? std::max(0, axis - 1 + p.num_axes()) : bottomDims.nbDims;

    std::cout<<" axStart:"<< axStart << " axEnd:" << axEnd << std::endl;
    std::vector<int> reshapeDims;

    // reshapeDims.reserve(axStart);
    // // 不做reshape的维度，直接复制bottom的
    for (int i = 0; i < axStart; i++)
    {
        reshapeDims.push_back(bottomDims.d[i]);
    }

    //  检查情况２
    for (const auto & dim: reshapeDims)
    {
        std::cout<<" dim in reshapeDims:"<< dim << std::endl;
    }


    for (int i = 0; i < shape.dim_size(); i++)
    {
        // skip first 0 (batch)
        if (axis == 0 && i == 0)
        {
            continue;
        }
        if (shape.dim(i) == 0)
        {
            // If there is no bottom dimension corresponding to the current axis, then the params are invalid
            assert(static_cast<int>(reshapeDims.size()) < bottomDims.nbDims);
            std::cerr<<" reshapeDims.size():"<<reshapeDims.size()<<" bottomDims.nbDims:"<<bottomDims.nbDims<<std::endl;
            std::cerr<<" bottomDims.d[reshapeDims.size()]:"<< bottomDims.d[reshapeDims.size()] << std::endl;
            reshapeDims.push_back(bottomDims.d[reshapeDims.size()]);
        }
        else
        {
            reshapeDims.push_back(shape.dim(i));
        }
    }
    for (const auto & dim: reshapeDims)
    {
        std::cout<<"222: dim in reshapeDims:"<< dim << std::endl;
    }

    for (int i = axEnd; i < bottomDims.nbDims; i++)
    {
        reshapeDims.push_back(bottomDims.d[i]);
    }

    Dims topDims{};
    topDims.nbDims = static_cast<int>(reshapeDims.size());

    std::cout<< "bottomDims.nbDims:"<< bottomDims.nbDims << " topDims.nbDims:"<< topDims.nbDims << std::endl;
    for (int i = 0; i < topDims.nbDims; i++)
    {
        topDims.d[i] = reshapeDims[i];
    }

    // Check there is at most one -1, and handle such case
    int countMinusOne = 0;
    for (int i = 0; i < topDims.nbDims; i++)
    {
        if (topDims.d[i] == -1)
        {
            countMinusOne += 1;
            // Inferred dimension
            int64_t newDim = parserutils::volume(bottomDims) / -parserutils::volume(topDims);
            // 　都不考虑能不能整除的？？？
            topDims.d[i] = newDim;
        }
    }

    for (int i = 0; i < topDims.nbDims; i++)
    {
        std::cout<<"topDims.d[i]:" << topDims.d[i] << std::endl;
    }

    if (countMinusOne > 1)
    {
        std::cout << "Caffe Parser: Invalid reshape param. At most one axis can be inferred from other dimensions" << std::endl;
        return nullptr;
    }

    auto layer = network.addShuffle(*tensors[msg.bottom(0)]);
    layer->setReshapeDimensions(topDims);
    return layer;
}
} //namespace nvcaffeparser1