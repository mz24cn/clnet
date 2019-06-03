/*
 * image_io.cpp
 *
 *  Created on: 2017/8/22
 *      Author: ZhangHua
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <algorithm>

#include <image_io.hpp>
#include <device_instance.hpp>

using namespace std;
using namespace clnet;

int reverse_int(int i) {
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

//revised from https://github.com/fengbingchun/NN_Test/blob/master/demo/DatasetToImage/funset.cpp
Tensor* read_mnist_images(string file, string name, int alignment_size)
{
	ifstream ifs(file, ios::binary);
	if (!ifs)
		throw runtime_error("failed to open " + file);

	int magic_number, n_cols, n_rows, num_of_images;
	ifs.read((char*) &magic_number, sizeof(magic_number));
	magic_number = reverse_int(magic_number);
	ifs.read((char*) &num_of_images, sizeof(num_of_images));
	num_of_images = reverse_int(num_of_images);
	ifs.read((char*) &n_rows, sizeof(n_rows));
	n_rows = reverse_int(n_rows);
	ifs.read((char*) &n_cols, sizeof(n_cols));
	n_cols = reverse_int(n_cols);

	int num = (num_of_images + alignment_size - 1) / alignment_size * alignment_size;
	auto tensor = new Tensor({num, n_cols, n_rows}, {}, name);
	tensor->initialize(nullptr);
	float* p = tensor->pointer;
	unsigned char temp;
	for (int i = 0; i < num_of_images; ++i) {
		for (int r = 0; r < n_rows; ++r) {
			for (int c = 0; c < n_cols; ++c) {
				ifs.read((char*) &temp, sizeof(temp));
				*p++ = (float) temp;
			}
		}
	}
	ifs.close();
	memcpy(p, tensor->pointer, (num - num_of_images) * n_cols * n_rows * sizeof(float));
	return tensor;
}

Tensor* read_mnist_labels(string file, string name, int alignment_size)
{
	ifstream ifs(file, ios::binary);
	if (!ifs)
		throw runtime_error("failed to open " + file);

	int magic_number, num_of_images;
	ifs.read((char*) &magic_number, sizeof(magic_number));
	magic_number = reverse_int(magic_number);
	ifs.read((char*) &num_of_images, sizeof(num_of_images));
	num_of_images = reverse_int(num_of_images);

	int num = (num_of_images + alignment_size - 1) / alignment_size * alignment_size;
	auto tensor = new Tensor({num}, {}, name);
	tensor->initialize(nullptr);
	float* p = tensor->pointer;
	unsigned char temp;
	for (int i = 0; i < num_of_images; ++i) {
		ifs.read((char*) &temp, sizeof(temp));
		*p++ = (float) temp;
	}
	ifs.close();
	memcpy(p, tensor->pointer, (num - num_of_images) * sizeof(float));
	return tensor;
}

typedef long LONG;
typedef unsigned long DWORD;
typedef unsigned short WORD;
typedef unsigned char BYTE;

// 位图文件头文件定义
#pragma pack(push, 2)
typedef struct {
	WORD    bfType; //文件类型，必须是0x424D,即字符“BM”
	DWORD   bfSize; //文件大小
	WORD    bfReserved1; //保留字
	WORD    bfReserved2; //保留字
	DWORD   bfOffBits; //从文件头到实际位图数据的偏移字节数
} BMPFILEHEADER;
#pragma pack(pop)

typedef struct{
	DWORD      biSize; //信息头大小
	LONG       biWidth; //图像宽度
	LONG       biHeight; //图像高度
	WORD       biPlanes; //位平面数，必须为1
	WORD       biBitCount; //每像素位数
	DWORD      biCompression; //压缩类型
	DWORD      biSizeImage; //压缩图像大小字节数
	LONG       biXPelsPerMeter; //水平分辨率
	LONG       biYPelsPerMeter; //垂直分辨率
	DWORD      biClrUsed; //位图实际用到的色彩数
	DWORD      biClrImportant; //本位图中重要的色彩数
} BMPINFOHEADER; //位图信息头定义

bool generate_24bits_bmp(unsigned char* pData, int width, int height, const char* file) //生成Bmp图片，传递RGB值，传递图片像素大小，传递图片存储路径
{
	int size = width * height * 3; //像素数据大小
	// 位图第一部分，文件信息
	BMPFILEHEADER bfh;
	bfh.bfType = 0x4D42; //bm
	bfh.bfSize = size + sizeof(BMPFILEHEADER) + sizeof(BMPINFOHEADER);
	bfh.bfReserved1 = 0; //reserved
	bfh.bfReserved2 = 0; //reserved
	bfh.bfOffBits = sizeof(BMPFILEHEADER) + sizeof(BMPINFOHEADER);

	// 位图第二部分，数据信息
	BMPINFOHEADER bih;
	bih.biSize = sizeof(BMPINFOHEADER);
	bih.biWidth = width;
	bih.biHeight = height;
	bih.biPlanes = 1;
	bih.biBitCount = 24;
	bih.biCompression = 0;
	bih.biSizeImage = size;
	bih.biXPelsPerMeter = 0;
	bih.biYPelsPerMeter = 0;
	bih.biClrUsed = 0;
	bih.biClrImportant = 0;

	FILE* fp = fopen(file,"wb");
	if (!fp)
		return false;
	fwrite(&bfh, sizeof(BMPFILEHEADER), 1, fp);
	fwrite(&bih, sizeof(BMPINFOHEADER), 1, fp);
	fwrite(pData, 1, size, fp);
	fclose(fp);
	return true;
}

unsigned char* read_24bits_bmp(const char *file, int* width, int* height)
{
	//二进制读方式打开指定的图像文件
	FILE* fp = fopen(file, "rb");
	if (!fp)
		return nullptr;

	//跳过位图文件头结构BMPFILEHEADER
	fseek(fp, sizeof(BMPFILEHEADER), 0);
	//定义位图信息头结构变量，读取位图信息头进内存，存放在变量head中
	BMPINFOHEADER head;
	fread(&head, sizeof(BMPINFOHEADER), 1, fp);
	if (head.biBitCount != 24) //仅支持24位BMP
		return nullptr;
	//获取图像宽、高、每像素所占位数等信息
	*width = head.biWidth;
	*height = head.biHeight;
	//申请位图数据所需要的空间，读位图数据进内存
	int size = *width * *height * 3;
	unsigned char* buffer = new unsigned char[size];
	fread(buffer, 1, size, fp);
	//关闭文件
	fclose(fp);
	return buffer;
}

void read_cifar10_images_and_labels(string file, int alignment_size, int offset, int num_of_images, Tensor* images, Tensor* labels)
{
	ifstream ifs(file, ios::binary);
	if (!ifs)
		throw runtime_error("failed to open " + file);

	char* buffer = new char[(32 * 32 * 3 + 1) * num_of_images];
	ifs.read(buffer, (32 * 32 * 3 + 1) * num_of_images);
	ifs.close();

	int num = (num_of_images + alignment_size - 1) / alignment_size * alignment_size;
	float *pI = images->pointer + offset * 32 * 32 * 3, *pL = labels->pointer + offset;
	for (int i = 0; i < num_of_images; ++i) {
		*pL++ = (float) (unsigned char) buffer[(32 * 32 * 3 + 1) * i];
		for (int r = 0; r < 32; ++r) {
			for (int c = 0; c < 32; ++c) {
				int n = (32 * 32 * 3 + 1) * i + r * 32 + c;
				*pI++ = ((unsigned char) buffer[n + 1]) / 255.0f; //normalize to [0, 1]
				*pI++ = ((unsigned char) buffer[n + 1 + 32 * 32]) / 255.0f;
				*pI++ = ((unsigned char) buffer[n + 1 + 32 * 32 * 2]) / 255.0f;
			}
		}
	}
	memcpy(pI, images->pointer, (num - num_of_images) * 32 * 32 * 3 * sizeof(float));
	memcpy(pL, labels->pointer, (num - num_of_images) * sizeof(float));
	delete buffer;
}

void read_cifar100_images_and_labels(string file, int alignment_size, bool use_fine_label, int num_of_images, Tensor* images, Tensor* labels)
{
	ifstream ifs(file, ios::binary);
	if (!ifs)
		throw runtime_error("failed to open " + file);

	char* buffer = new char[(32 * 32 * 3 + 2) * num_of_images];
	ifs.read(buffer, (32 * 32 * 3 + 2) * num_of_images);
	ifs.close();

	int num = (num_of_images + alignment_size - 1) / alignment_size * alignment_size;
	float *pI = images->pointer, *pL = labels->pointer;
	for (int i = 0; i < num_of_images; ++i) {
		if (use_fine_label)
			*pL++ = (float) (unsigned char) buffer[(32 * 32 * 3 + 2) * i + 1]; //fine label
		else
			*pL++ = (float) (unsigned char) buffer[(32 * 32 * 3 + 2) * i]; //coarse label
		for (int r = 0; r < 32; ++r) {
			for (int c = 0; c < 32; ++c) {
				int n = (32 * 32 * 3 + 2) * i + r * 32 + c;
				*pI++ = ((unsigned char) buffer[n + 2]) / 255.0f;
				*pI++ = ((unsigned char) buffer[n + 2 + 32 * 32]) / 255.0f;
				*pI++ = ((unsigned char) buffer[n + 2 + 32 * 32 * 2]) / 255.0f;
			}
		}
	}
	memcpy(pI, images->pointer, (num - num_of_images) * 32 * 32 * 3 * sizeof(float));
	memcpy(pL, labels->pointer, (num - num_of_images) * sizeof(float));
	delete buffer;
}
