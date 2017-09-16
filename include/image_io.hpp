/*
 * image_io.hpp
 *
 *  Created on: 2017/8/29
 *      Author: ZhangHua
 */

#ifndef INCLUDE_IMAGE_IO_HPP_
#define INCLUDE_IMAGE_IO_HPP_

#include <tensor.hpp>

void generate_24bits_bmp(unsigned char* pData, int width, int height, const char* file);

clnet::Tensor* read_mnist_images(std::string file, std::string name, int alignment_size = 1);
clnet::Tensor* read_mnist_labels(std::string file, std::string name, int alignment_size = 1);

#endif /* INCLUDE_IMAGE_IO_HPP_ */
