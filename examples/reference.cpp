/*
 * reference.cpp
 *
 *  Created on: 2017/1/31
 *  	Recovered from revision 2017/4/30 11:15PM, then simplified.
 *  	It is reserved as a performance reference. All code in one cpp file.
 *	  Author: ZhangHua
 */

#define CL_HPP_TARGET_OPENCL_VERSION 120

#include <ctime>
#include <vector>
#include <queue>
#include <memory>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <random>
#include <array>
#include <omp.h>

#if defined(__GNUC__) && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#if CL_HPP_TARGET_OPENCL_VERSION < 200
#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"
#else
#define CL_HPP_ENABLE_EXCEPTIONS
#include "cl2.hpp"
#endif

#define MILLIS(time) (clock() - time) * 1000 / CLOCKS_PER_SEC
using std::chrono::system_clock;

enum TensorType {weight, bias, no_bias, data, label, gradient};
struct _Tensor  {
	std::vector<int> dimensions;
	TensorType type;
	std::string alias;
	int size;
	float* data;

	void allocate()
	{
		if (dimensions.empty()) {
			size = 0;
			data = nullptr;
		}
		else {
			size = 1;
			for (int dim : dimensions)
				size *= dim;
			data = new float[size];
			size *= sizeof(float);
		}
	}
};

static cl_device_type targetDeviceType = CL_DEVICE_TYPE_GPU;
static std::vector<cl::Device>* computeDevices = NULL;
static size_t totalComputeUnits;
static std::vector<int> targetDeviceIDs{0};

static const int K = 2, N = 128, HIDDEN = 4096;
#if CL_HPP_TARGET_OPENCL_VERSION < 200
#define KernelFunctor make_kernel
#endif
typedef cl::KernelFunctor<
		cl::Buffer&, //out
		cl::Buffer&, //in
		cl::Buffer&, //weight
		cl::Buffer&, //bias
		cl::LocalSpaceArg, //tmp
		int, int //dim_hidden, dim_in
		> kernelFullyConnected;
typedef cl::KernelFunctor<
		cl::Buffer&, //weight_grad
		cl::Buffer&, //bias_grad
		cl::Buffer&, //in
		cl::Buffer&, //out
		cl::Buffer&, //label
		cl::LocalSpaceArg, //tmp
		int, int, int //dim_out, dim_in, batch_size
		> kernelLossFunction;
typedef cl::KernelFunctor<
		cl::Buffer&, //weight_grad
		cl::Buffer&, //bias_grad
		cl::Buffer&, //in
		cl::Buffer&, //out
		cl::Buffer&, //weight
		cl::Buffer&, //nabla
		cl::LocalSpaceArg, //tmp
		int, int, int, int //dim_out, dim_in, dim_weight_next_out, batch_size
		> kernelBackPropagate;
typedef cl::KernelFunctor<
		cl::Buffer&, //params
		cl::Buffer&, //params_grad
		float, //learning_rate
		float //weight_decay
		> kernelStochasticGradientDescent;

static const char KERNEL_FILE[] = "D:\\CPP\\OpenCLNet\\src\\kernels.cl";
#ifdef __MINGW64__
// Remove '-cl-std=CL2.0' when CL_HPP_TARGET_OPENCL_VERSION=200 slightly speed up
static const char* cl_build_options = "-ID:\\CPP\\OpenCLNet\\include -D_WINDOWS -cl-std=CL2.0";
#else
static const char* cl_build_options = "-I/root/data";
#endif

//std::mutex notification_mutex, queue_mutex;
//std::unique_lock<std::mutex> notification_lock(notification_mutex);
//std::condition_variable notification;
//std::queue<int> parameters_queue, gradients_queue;

static std::vector<_Tensor> parameters;
//size_t parameters_timestamp;
static int /*UPDATER_DEVICE = 1, */N_PARAM = -1;

static int max_iters = 10001;
static float learning_rate = 0.00001;
static float weight_decay = 0;//0.000012;

//CL_CALLBACK void gradients_event_callback(cl_event, cl_int, void * user_data)
//{
//	DeviceInstance* instance = reinterpret_cast<DeviceInstance*>(user_data);
//	if (instance->gradients_state-- == 1) {
//		queue_mutex.lock();
//		gradients_queue.push(instance->ID);
//		queue_mutex.unlock();
//		notification.notify_all();
//	}
//}
//
//CL_CALLBACK void parameters_event_callback(cl_event, cl_int, void * user_data)
//{
//	DeviceInstance* instance = reinterpret_cast<DeviceInstance*>(user_data);
//	if (instance->parameters_state-- == 1) {
//		queue_mutex.lock();
//		parameters_queue.push(instance->ID);
//		queue_mutex.unlock();
//		notification.notify_all();
//	}
//}

void printBuffer(cl::CommandQueue& queue, const cl::Buffer& buffer, int rows, int columns, const char* name = "")
{
	float* data = new float[rows * columns];
	queue.enqueueReadBuffer(buffer, CL_TRUE, 0, rows * columns * sizeof(float), data);
	std::cout << name << "[" << rows << "x" << columns << "]:" << std::endl;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < columns; j++)
			std::cout << data[i * columns + j] << ",";
		std::cout << std::endl;
	}
}

void OutputAccuracy(float* pred, float* target) {
	float sum = 0;
	for (int i = 0; i < N; ++i) {
		float delta = pred[i] - target[i];
		sum += delta * delta;
	}
	std::cout << " Error: " << sum / 2 / N  << std::endl;
}

//void update_device_parameters(DeviceInstance& instance)
//{
//	if (instance.parameters_state > 0)
//		return;
//#pragma omp parallel for
//	for (int i = 0; i < N_PARAM; i++)
//		memcpy(instance.parameters[i], parameters[i].data, parameters[i].size);
//	instance.parameters_timestamp = parameters_timestamp;
//	instance.parameters_state = N_PARAM;
//}

//void stochastic_gradient_decent_update()
//{
//    std::string kernel_code{R"CLC(
//kernel void stochastic_gradient_descent_update_params(
//		global float* params,
//		global const float* params_grad,
//		float learning_rate,
//		float weight_decay)
//{
//	size_t GID = get_global_id(0);
//	params[GID] -= learning_rate * (params_grad[GID] + weight_decay * params[GID]);
//}
//    )CLC"};
//	auto& device = (*computeDevices)[UPDATER_DEVICE];
//	cl::Context context(device);
//	cl::Program program(context, kernel_code);
//	try {
//		program.build(cl_build_options);
//		std::cout << "Optimizer running on " << UPDATER_DEVICE << ":" << device.getInfo<CL_DEVICE_NAME>() << std::endl;
//	}
//	catch (cl::Error& e) {
//		std::cout << "Error in " << e.what() << " (" << e.err() << "): " << clErrorCodeDescriptions[-e.err()] << std::endl;
//		std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
//	}
//#if CL_HPP_TARGET_OPENCL_VERSION < 200
//			cl::CommandQueue queue(context, device);
//#else
//			cl::CommandQueue queue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
//#endif
//
//	kernelStochasticGradientDescent stochastic_gradient_descent_update_params(program, "stochastic_gradient_descent_update_params");
//	std::vector<cl::Buffer> parameters_buffer, gradients_buffer;
//	for (auto& parameter : parameters) {
//		parameters_buffer.push_back(cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, parameter.size, parameter.data));
//		gradients_buffer.push_back(cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, parameter.size));
//	}
//	cl::Event event;
//	std::vector<cl::Event> updates;
//
//	while (UPDATER_DEVICE >= 0) {
//		notification.wait(notification_lock, [] { return !gradients_queue.empty() || UPDATER_DEVICE < 0; });
//
//		while (!gradients_queue.empty()) {
//			queue_mutex.lock();
//			int ID = gradients_queue.front();
//			gradients_queue.pop();
//			queue_mutex.unlock();
//
//			auto& instance = DeviceInstance::ALL[ID];
//			for (int i = 0; i < N_PARAM; i++) {
//				queue.enqueueWriteBuffer(gradients_buffer[i], CL_FALSE, 0, parameters[i].size, instance.gradients[i], NULL, &event);
//				updates.push_back(stochastic_gradient_descent_update_params(
//						cl::EnqueueArgs(queue, event, cl::NDRange(parameters[i].size / sizeof(float))),
//						parameters_buffer[i], gradients_buffer[i], learning_rate * N, weight_decay));
//			}
//			for (auto& ev : updates)
//				ev.wait();
//			instance.gradients_state = N_PARAM;
//		}
//		if (!updates.empty()) {
//			updates.clear();
//			for (int i = 0; i < N_PARAM; i++) {
//				queue.enqueueReadBuffer(parameters_buffer[i], CL_FALSE, 0, parameters[i].size, parameters[i].data, NULL, &event);
//				updates.push_back(event);
//			}
//			for (auto& ev : updates)
//				ev.wait();
//			parameters_timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(system_clock::now().time_since_epoch()).count();
//			for (auto& iter : DeviceInstance::ALL)
//				update_device_parameters(iter.second);
//			updates.clear();
//		}
//
//		while (!parameters_queue.empty()) {
//			queue_mutex.lock();
//			int ID = parameters_queue.front();
//			parameters_queue.pop();
//			queue_mutex.unlock();
//
//			auto& instance = DeviceInstance::ALL[ID];
//			if (parameters_timestamp <= instance.parameters_timestamp)
//				continue;
//			update_device_parameters(instance);
//		}
//	}
//}

void initialize_Xavier_normal_distribution(float mu = 0, float sigma = 1.0f)
{
	std::default_random_engine generator;
	for (auto& parameter : parameters) {
		std::normal_distribution<float> distribution(mu, sigma / parameter.dimensions[1]);
		if (parameter.type == bias)
			memset(parameter.data, 0, parameter.size);
		else if (parameter.type == weight)
			for (float *p = parameter.data, *end = p + parameter.size / sizeof(float); p < end; p++)
				*p = (float) distribution(generator);
	}
}

////Reference: http://stackoverflow.com/questions/24465533/implementing-boostbarrier-in-c11
//class thread_barrier
//{
//private:
//	std::mutex mutex;
//	std::condition_variable cv;
//	std::size_t count, times, threads_num;
//public:
//	explicit thread_barrier(std::size_t threads) : count{threads}, times(0), threads_num(threads) { }
//	void wait() {
//		auto current = times;
//		std::unique_lock<std::mutex> lock{mutex};
//		if (--count == 0) {
//			times++;
//			count = threads_num;
//			cv.notify_all();
//		}
//		else
//			cv.wait(lock, [this, current] { return current != times; });
//	}
//};

const std::string kernel_code = R"CLC(
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics: enable
#ifdef cl_khr_int64_base_atomics
	#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
	#define atom_type ulong
#else
	#define atom_type uint
#endif
#ifdef cl_amd_printf
//	#pragma OPENCL EXTENSION cl_amd_printf: enable
#endif
#ifndef work_group_barrier
	#define work_group_barrier barrier
#endif
#ifndef NULL
	#define NULL 0
#endif

inline float sigmoid(float z)
{
	return 1.0f / (1.0f + exp(-z));
}
inline float sigmoid_gradient(float a)
{
	return a * (1.0f - a);
}

inline float relu(float z)
{
	return z > 0? z : 0;
}
inline float relu_gradient(float a)
{
	return a > 0? 1 : 0;
}

//tanh is predefined
inline float tanh_gradient(float a)
{
	return 1.0f - a * a;
}

inline float softrelu(float z)
{
	return log(1.0f + exp(z));
}
inline float softrelu_gradient(float a)
{
	return 1.0f - exp(-a);
}

inline float linear_regression(float y, float label)
{
	float delta = y - label;
	return delta * delta;
}
inline float linear_regression_gradient(float y, float label)
{
	return y - label;
}

//softmax is calculated by pow(z) / sum of pow
inline float softmax_output_gradient(float a_j, float a_i, bool i_equal_j)
{
	return i_equal_j? a_j * (1.0 - a_j) : - a_j * a_i;
}

//Parallel: (sizeof(data))
kernel void activation_sigmoid(global float* data)
{
	int GID = get_global_id(0);
	data[GID] = sigmoid(data[GID]);
}

//Parallel: (batch_size * dim_hidden)
kernel void feed_forward_fully_connected_sigmoid(global float* out, global const float* in, global const float* weight, global const float* bias, 
		local float* tmp, const int dim_hidden, const int dim_in)
{
	int GID = get_global_id(0);
	int n = GID / dim_hidden;
	int hidden = GID % dim_hidden;
	int weight_offset = hidden * dim_in;
	int in_offset = n * dim_in;
	float z = bias[hidden];

#pragma unroll
	for (int i = 0; i < dim_in; i++)
		z += weight[weight_offset + i] * in[in_offset + i];
	out[GID] = sigmoid(z);
}

//Parallel: (batch_size * dim_hidden * get_local_size(0))
//Note: choose local NDRange size near (2 * dim_in) when enqueue ASAP
kernel void feed_forward_fully_connected_softrelu(global float* out, global const float* in, global const float* weight, global const float* bias, 
		local float* tmp, const int dim_hidden, const int dim_in)
{
	int GID = get_global_id(0);
// parallel for reduction: should be power of 2
// Use max parallel get_local_size(0) because OpenCL only synchorizes in a work group 
	int parallel = get_local_size(0);
	int n = GID / parallel / dim_hidden;
	int hidden = (GID / parallel) % dim_hidden;
	int weight_offset = hidden * dim_in;
	int in_offset = n * dim_in;
	float z = bias[hidden];

// parallel addition, trade space for time
	int pos = GID % parallel;
	float sum = 0;
// support any value for dim_in. Inefficient for dim_in < parallel / 2
	for (int index = pos; index < dim_in; index += parallel)
		sum += weight[weight_offset + index] * in[in_offset + index];

	tmp[pos] = sum;
	work_group_barrier(CLK_LOCAL_MEM_FENCE);
	for (int stride = parallel / 2; stride > 0; stride = stride / 2) {
		if (pos < stride)
			tmp[pos] += tmp[pos + stride];
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (pos == 0)
		out[GID / parallel] = softrelu(z + tmp[0]);
}

//Parallel: weight_Out_dims x weight_In_dims
kernel void back_propagate_cascade_fully_connected_sigmoid_gradient(global float* weight_grad, global float* bias_grad, global const float* in,
		global const float* out, global const float* weight_next, global const float* nabla_next/*bias_grad*/,
		local float* tmp, const int dim_out, const int dim_in/*k*/, const int dim_weight_next_out, const int batch_size)
{
	int GID = get_global_id(0);
	int j = GID / dim_in, k = GID % dim_in;
	if (k == 0)
		bias_grad[j] = 0;
	weight_grad[GID] = 0;

	for (int n = 0; n < batch_size; n++) {
		float a_j = out[n * dim_out + j];

		float sum = 0;
		for (int i = 0; i < dim_weight_next_out; i++)
			sum += weight_next[i * dim_in + j] * nabla_next[i];

		float nabla = sum * sigmoid_gradient(a_j);
		if (k == 0)
			bias_grad[j] += nabla;
		weight_grad[GID] += in[n * dim_in + k] * nabla;
	}
	if (k == 0)
		bias_grad[j] /= batch_size;
	weight_grad[GID] /= batch_size;
}

//Parallel: (weight_Out_dims * weight_In_dims * get_local_size(0))
kernel void back_propagate_linear_regression_loss_softrelu_gradient(global float* weight_grad, global float* bias_grad, global const float* in,
		global const float* out, global const float* label, local float* tmp/*size is 2 * get_local_size(0)*/,
		const int dim_out/*num_hidden*/, const int dim_in/*k*/, const int batch_size)
{
	int GID = get_global_id(0);
	int parallel = get_local_size(0);
	int j = GID / parallel / dim_in;
	int k = (GID / parallel) % dim_in;
	int pos = GID % parallel;
	
	float nabla_sum = 0, weight_grad_sum = 0;
	for (int n = pos; n < batch_size; n += parallel) {
		float a_j = out[n * dim_out + j];
		float nabla = linear_regression_gradient(a_j, label[n * dim_out + j]) * softrelu_gradient(a_j);
		nabla_sum += nabla;
		weight_grad_sum += in[n * dim_in + k] * nabla;
	}

	local float* tmp_w = tmp + parallel;
	tmp[pos] = nabla_sum;
	tmp_w[pos] = weight_grad_sum;
	work_group_barrier(CLK_LOCAL_MEM_FENCE);
	for (int stride = parallel / 2; stride > 0; stride = stride / 2) {
		if (pos < stride) {
			tmp[pos] += tmp[pos + stride];
			tmp_w[pos] += tmp_w[pos + stride];
		}
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (pos == 0) {
		if (k == 0)
			bias_grad[j] = tmp[0] / batch_size;
		weight_grad[GID / parallel] = tmp_w[0] / batch_size;
	}
}

//Parallel: params->dims
kernel void update_parameters_by_stochastic_gradient_descent(global float* params, global const float* params_grad,
		float learning_rate, float weight_decay)
{
	int GID = get_global_id(0);
	params[GID] -= learning_rate * (params_grad[GID] + weight_decay * params[GID]);
}
)CLC";

void train_on_device(int device)
{
	targetDeviceIDs[0] = device;
	N_PARAM = 4;
	std::array<TensorType, 4> _types = {weight, bias, weight, bias};
	int _params_size[4][2] = {{HIDDEN, K}, {HIDDEN, 1}, {1, HIDDEN}, {1, 1}};
	for (int i = 0; i < N_PARAM; i++) {
		_Tensor tensor = {{_params_size[i][0], _params_size[i][1]}, _types[i]};
		tensor.allocate();
		parameters.push_back(tensor);
	}
	initialize_Xavier_normal_distribution(0, 2.34f);

	const int DEFAULT_GROUP_SIZE = 256; //should be bigger than N * 2
	cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * DEFAULT_GROUP_SIZE);
//	std::thread updater(stochastic_gradient_decent_update);

//	for (int ID : targetDeviceIDs)
//		DeviceInstance::ALL[ID] = ID;
	int dev_no, deviceNum = targetDeviceIDs.size();
//	thread_barrier barrier(deviceNum);
#pragma omp parallel for
	for (dev_no = 0; dev_no < deviceNum; dev_no++) {
//		auto& instance = DeviceInstance::ALL[targetDeviceIDs[device]];
		auto& device = (*computeDevices)[targetDeviceIDs[dev_no]];
		auto name = device.getInfo<CL_DEVICE_NAME>();

//        std::string test_code{R"CLC(
//
//kernel void kernelTest(global float* dest, global float* src)
//{
//	if (src != NULL)
//		dest[get_global_id(0)] = 1;
//	else
//		dest[get_global_id(0)] = 2;
//}
//        )CLC"};
		cl::Context context(device);
		cl::Program program(context, kernel_code/* + test_code*/);
		try {
			clock_t time = MILLIS(0);
			program.build(cl_build_options);
			time = MILLIS(time);
			std::cout << name << ":\n\tbuild time: " << time << "ms." << std::endl;

#if CL_HPP_TARGET_OPENCL_VERSION < 200
			cl::CommandQueue queue(context, device);
#else
			cl::CommandQueue queue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
#endif

			kernelFullyConnected fully_connected_activation_sigmoid(program, "feed_forward_fully_connected_sigmoid");
			kernelFullyConnected fully_connected_activation_softrelu(program, "feed_forward_fully_connected_softrelu");
			kernelLossFunction linear_regression_gradient_softrelu(program, "back_propagate_linear_regression_loss_softrelu_gradient");
			kernelBackPropagate back_propagate_gradient_sigmoid(program, "back_propagate_cascade_fully_connected_sigmoid_gradient");
			kernelStochasticGradientDescent stochastic_gradient_descent_update_params(program, "update_parameters_by_stochastic_gradient_descent");

			float* x = new float[N * K];
			float* y = new float[N];
			cl::Buffer data(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(float) * N * K, x);
			cl::Buffer label(context, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, sizeof(float) * N, y);
			cl::Buffer l0_output(context, CL_MEM_READ_WRITE, sizeof(float) * N * HIDDEN);
			cl::Buffer l1_output(context, CL_MEM_READ_WRITE, sizeof(float) * N * 1);

			cl::Buffer l0_weight(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, (size_t) parameters[0].size, parameters[0].data);
			cl::Buffer l0_bias(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, (size_t) parameters[1].size, parameters[1].data);
			cl::Buffer l1_weight(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, (size_t) parameters[2].size, parameters[2].data);
			cl::Buffer l1_bias(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, (size_t) parameters[3].size, parameters[3].data);
//printBuffer(queue, l0_weight, HIDDEN, K, "l0_weight");

			cl::Buffer l0_weight_grad(context, CL_MEM_READ_WRITE, sizeof(float) * HIDDEN * K);
			cl::Buffer l0_bias_grad(context, CL_MEM_READ_WRITE, sizeof(float) * HIDDEN);
			cl::Buffer l1_weight_grad(context, CL_MEM_READ_WRITE, sizeof(float) * HIDDEN);
			cl::Buffer l1_bias_grad(context, CL_MEM_READ_WRITE, sizeof(float) * 1);

//			cl::Kernel kernelTest(program, "kernelTest");
//			cl::Buffer dest(context, CL_MEM_READ_WRITE, sizeof(float) * 10);
//			kernelTest.setArg(0, dest);
//			kernelTest.setArg(1, nullptr);
//			queue.enqueueNDRangeKernel(kernelTest, cl::NullRange, cl::NDRange(10), cl::NullRange, nullptr, nullptr);
//			float dest_buf[10];
//			queue.enqueueReadBuffer(dest, CL_TRUE, 0, sizeof(float) * 10, dest_buf);

//			std::array<cl::Buffer*, 4> params = {&l0_weight, &l0_bias, &l1_weight, &l1_bias};
//			std::array<cl::Buffer*, 4> params_grad = {&l0_weight_grad, &l0_bias_grad, &l1_weight_grad, &l1_bias_grad};
//			for (int i = 0; i < N_PARAM; i++) {
//				instance.parameters_buffer.push_back(*params[i]);
//				instance.gradients_buffer.push_back(*params_grad[i]);
//				instance.parameters.push_back(new float[parameters[i].size / sizeof(float)]);
////				instance.parameters.push_back(parameters[i].data);
//				instance.gradients.push_back(new float[parameters[i].size / sizeof(float)]);
//				if (deviceNum > 1)
//					instance.gradients_state = N_PARAM;
//			}

			cl::Event event;
			std::vector<cl::Event> updates;

			// set arguments for kernel, and execute it.
//			initializer_normal_distribution(cl::EnqueueArgs(queue, cl::NDRange(HIDDEN * K)), l0_weight, 1000.0f);
//			initializer_normal_distribution(cl::EnqueueArgs(queue, cl::NDRange(HIDDEN)), l0_bias, 1000.0f);
//			initializer_normal_distribution(cl::EnqueueArgs(queue, cl::NDRange(HIDDEN)), l1_weight, 1000.0f);
//			initializer_normal_distribution(cl::EnqueueArgs(queue, cl::NDRange(1)), l1_bias, 1000.0f);
//			kernelInitializer delta(program, "delta");
//			printBuffer(queue, l0_weight, HIDDEN, K, "l0_weight");
//			delta(cl::EnqueueArgs(queue, cl::NDRange(1)), l0_weight, HIDDEN * K -1);
//			printBuffer(queue, l0_weight, HIDDEN, K, "l0_weight");
//			printBuffer(queue, l1_weight, 1, HIDDEN, "l1_weight");
//			delta(cl::EnqueueArgs(queue, cl::NDRange(1)), l1_weight, HIDDEN - 1);
//			printBuffer(queue, l1_weight, 1, HIDDEN, "l1_weight");

//			barrier.wait();
			time = MILLIS(0);
			for (int iter = 0; iter < max_iters; ++iter) {
				for (int i = 0; i < N; i++) {
					for (int j = 0; j < K; j++) {
						float value;
						while ((value = std::rand()) == 0);
						x[i * K + j] = (1 + 3.0f * value / RAND_MAX);
					}
					y[i] = x[i * K] / x[i * K + 1];
				}

//				if (!updates.empty())
//					event.wait(); //enqueueWriteBuffer seems not to wait for kernel enqueue event, we have to call wait() here.
				queue.enqueueWriteBuffer(data, CL_FALSE, 0, sizeof(float) * N * K, x, NULL, &event);
				updates.push_back(event);
				queue.enqueueWriteBuffer(label, CL_FALSE, 0, sizeof(float) * N, y, NULL, &event);
				updates.push_back(event);
//printBuffer(queue, data, N, K, "data");
//printBuffer(queue, label, N, 1, "label");

				event = fully_connected_activation_sigmoid(
						cl::EnqueueArgs(queue, updates, cl::NDRange(N * HIDDEN)),
						l0_output, data, l0_weight, l0_bias, tmpMem, HIDDEN, K);
//printBuffer(queue, l0_weight, HIDDEN, K, "l0_weight");
//printBuffer(queue, l0_bias, HIDDEN, 1, "l0_bias");
//printBuffer(queue, l0_output, N, HIDDEN, "l0_output");
				event = fully_connected_activation_softrelu(
						cl::EnqueueArgs(queue, event, cl::NDRange(N * DEFAULT_GROUP_SIZE), cl::NDRange(DEFAULT_GROUP_SIZE)),
						l1_output, l0_output, l1_weight, l1_bias, tmpMem, 1, HIDDEN);
//printBuffer(queue, l1_weight, 1, HIDDEN, "l1_weight");
//printBuffer(queue, l1_bias, 1, 1, "l1_bias");
//printBuffer(queue, l1_output, N, 1, "l1_output");

				if (iter % 2000 == 0) {
					event.wait();
					float output[N];
					queue.enqueueReadBuffer(l1_output, CL_TRUE, 0, sizeof(float) * N * 1, output);
					std::cout << "Epoch[" << iter << "]";
					OutputAccuracy(output, y);
				}

				// back propagate
				event = linear_regression_gradient_softrelu(
						cl::EnqueueArgs(queue, event, cl::NDRange(1 * HIDDEN * N), cl::NDRange(N)),
						l1_weight_grad, l1_bias_grad, l0_output, l1_output, label,
						tmpMem, 1, HIDDEN, N);
//printBuffer(queue, l1_weight_grad, 1, HIDDEN, "l1_weight_grad");
//printBuffer(queue, l1_bias_grad, 1, 1, "l1_bias_grad");
				event = back_propagate_gradient_sigmoid(
						cl::EnqueueArgs(queue, event, cl::NDRange(HIDDEN * K/* * 128), cl::NDRange(128*/)),
						l0_weight_grad, l0_bias_grad, data, l0_output, l1_weight, l1_bias_grad,
						tmpMem, HIDDEN, K, 1, N);
//printBuffer(queue, l0_weight_grad, HIDDEN, K, "l0_weight_grad");
//printBuffer(queue, l0_bias_grad, HIDDEN, 1, "l0_bias_grad");

//				fit(	cl::EnqueueArgs(queue, cl::NDRange(1)),
//						data, label,
//						l0_weight, l0_bias, l1_weight, l1_bias,
//						l0_weight_grad, l0_bias_grad, l1_weight_grad, l1_bias_grad, l0_output, l1_output,
//						K, HIDDEN, 1, N, learning_rate, weight_decay);
				// update the parameters
//				if (instance.gradients_state == N_PARAM)
//					for (int i = 0; i < N_PARAM; i++) { //report gradients
//						queue.enqueueReadBuffer(instance.gradients_buffer[i], CL_FALSE, 0, parameters[i].size, instance.gradients[i], NULL, &event);
//						event.setCallback(CL_COMPLETE, gradients_event_callback, &instance);
//					}
//
//				updates.clear();
//				if (instance.parameters_state == N_PARAM)
//					for (int i = 0; i < N_PARAM; i++) { //load parameters
//						queue.enqueueWriteBuffer(instance.parameters_buffer[i], CL_FALSE, 0, parameters[i].size, instance.parameters[i], NULL, &event);
//						event.setCallback(CL_COMPLETE, parameters_event_callback, &instance);
//						updates.push_back(event);
//					}
//				else
//					for (int i = 0; i < N_PARAM; i++)
//						updates.push_back(stochastic_gradient_descent_update_params(
//								cl::EnqueueArgs(queue, event, cl::NDRange(parameters[i].size / sizeof(float))),
//								instance.parameters_buffer[i], instance.gradients_buffer[i], learning_rate * N, weight_decay));

				updates.push_back(stochastic_gradient_descent_update_params(
						cl::EnqueueArgs(queue, event, cl::NDRange(HIDDEN * K)),
						l0_weight, l0_weight_grad, learning_rate * N, weight_decay));
				updates.push_back(stochastic_gradient_descent_update_params(
						cl::EnqueueArgs(queue, event, cl::NDRange(HIDDEN)),
						l0_bias, l0_bias_grad, learning_rate * N, weight_decay));
				updates.push_back(stochastic_gradient_descent_update_params(
						cl::EnqueueArgs(queue, event, cl::NDRange(1 * HIDDEN)),
						l1_weight, l1_weight_grad, learning_rate * N, weight_decay));
				updates.push_back(stochastic_gradient_descent_update_params(
						cl::EnqueueArgs(queue, event, cl::NDRange(1)),
						l1_bias, l1_bias_grad, learning_rate * N, weight_decay));
//				printBuffer(queue, l0_weight, HIDDEN, K, "l0_weight");
//				printBuffer(queue, l0_bias, HIDDEN, 1, "l0_bias");
//				printBuffer(queue, l1_weight, 1, HIDDEN, "l1_weight");
//				printBuffer(queue, l1_bias, 1, 1, "l1_bias");
				for (auto ev : updates)
					ev.wait();
				updates.clear();
			}
			for (auto ev : updates)
				ev.wait();
			float output[N];
			queue.enqueueReadBuffer(l1_output, CL_TRUE, 0, sizeof(float) * N * 1, output);
			OutputAccuracy(output, y);
			time = MILLIS(time);
			std::cout << "time: " << time << std::endl;
//			printBuffer(queue, l0_weight, HIDDEN, K, "l0_weight");
//			printBuffer(queue, l0_bias, HIDDEN, 1, "l0_bias");
//			printBuffer(queue, l1_weight, 1, HIDDEN, "l1_weight");
//			printBuffer(queue, l1_bias, 1, 1, "l1_bias");

//			barrier.wait();
//			if (UPDATER_DEVICE == instance.ID || deviceNum <= 1) {
//				UPDATER_DEVICE = -1;
//				notification.notify_all();
//				updater.join();
//			}
//			barrier.wait();
			delete[] x;
			delete[] y;
//			for (int i = 0; i < N_PARAM; i++) {
//				delete instance.parameters[i];
//				delete instance.gradients[i];
//			}
		}
		catch (cl::Error& e) {
			std::cout << "Error in " << e.what() << " (" << e.err() << ")" << std::endl;
			std::cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		}
	}
}

void predict()
{

}

bool showInfo(std::ostream& out)
{
	out << "OpenCL Platforms:" << std::endl;
	std::vector<cl::Platform> platforms;
	try {
		cl::Platform::get(&platforms);
	}
	catch (cl::Error&) {
		out << "No OpenCL platforms found.\n" << std::endl;
		return false;
	}
	totalComputeUnits = 0;
	if (computeDevices != NULL)
		delete computeDevices;
	computeDevices = new std::vector<cl::Device>;
	for (auto& platform : platforms) {
		out << "Name:" << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
		out << "Version:" << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;
		out << "Vendor:" << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
		out << "Profile:" << platform.getInfo<CL_PLATFORM_PROFILE>() << std::endl;
		out << "Extensions:" << platform.getInfo<CL_PLATFORM_EXTENSIONS>() << std::endl;

		out << "Devices:" << std::endl;
		std::vector<cl::Device> devices;
		try {
			platform.getDevices(targetDeviceType, &devices);
		}
		catch (cl::Error&) {
			out << "No OpenCL GPU Devices found.\n" << std::endl;
			continue;
		}
		for (auto& device : devices) {
			std::string name = device.getInfo<CL_DEVICE_NAME>();
			computeDevices->push_back(device);
			totalComputeUnits += device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
			auto deviceType = device.getInfo<CL_DEVICE_TYPE>();
			auto sizeGroup = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
			auto sizesItem = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

			out << "\tName:" << name << std::endl;
			out << "\tType:";
			switch (deviceType) {
			case CL_DEVICE_TYPE_CPU: out << "CPU"; break;
			case CL_DEVICE_TYPE_GPU: out << "GPU"; break;
			default: out << "OTHER"; break;
			}
			out << std::endl;
			out << "\tVersion:" << device.getInfo<CL_DEVICE_VERSION>() << '/' << device.getInfo<CL_DRIVER_VERSION>() << std::endl;
			out << "\tGlobal/Local Memory:" << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << '/' << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
			out << "\tMax ComputeUnits/WorkGroupSize/WorkItemDims:" << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << '/' << sizeGroup << '/' << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>() << std::endl;
			out << "\tMax WorkItemSizes:";
			bool _first = true;
			for (auto size : sizesItem) {
				if (_first) _first = false; else	out << ':';
				out << size;
			}
			out << std::endl;
			out << "\tBuiltInKernels:";
			try {
				out << device.getInfo<CL_DEVICE_BUILT_IN_KERNELS>();
			} catch (cl::Error& e) {
				out << "Error in " << e.what() << " (" << e.err() << ") " << std::endl;
			}
			out << std::endl;
			out << "\tExtensions:" << device.getInfo<CL_DEVICE_EXTENSIONS>() << std::endl;
			out << std::endl;
		}
	}
	return totalComputeUnits > 0;
}
