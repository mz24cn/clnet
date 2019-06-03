/*
 * image_io.hpp
 *
 *  Created on: 2017/8/29
 *      Author: ZhangHua
 */

#ifndef INCLUDE_IMAGE_IO_HPP_
#define INCLUDE_IMAGE_IO_HPP_

#include <tensor.hpp>

bool generate_24bits_bmp(unsigned char* pData, int width, int height, const char* file);
unsigned char* read_24bits_bmp(const char *file, int* width, int* height);

clnet::Tensor* read_mnist_images(std::string file, std::string name, int alignment_size = 1);
clnet::Tensor* read_mnist_labels(std::string file, std::string name, int alignment_size = 1);

void read_cifar10_images_and_labels(std::string file, int alignment_size, int offset, int num_of_images, clnet::Tensor* images, clnet::Tensor* labels);
void read_cifar100_images_and_labels(std::string file, int alignment_size, bool use_fine_label, int num_of_images, clnet::Tensor* images, clnet::Tensor* labels);

#endif /* INCLUDE_IMAGE_IO_HPP_ */
