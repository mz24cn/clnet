/*
 * tensor.hpp
 *
 *  Created on: 2017/4/17
 *      Author: ZhangHua
 */

#ifndef INCLUDE_TENSOR_HPP_
#define INCLUDE_TENSOR_HPP_

#include <atomic>
#include <vector>
#include <string>
#include <set>
#include <thread>
#include <random>
#include <functional>
#include <climits>
#include <mutex>

#if defined(__GNUC__) && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif
#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

namespace clnet {
typedef long long int64;
struct DeviceInstance;

struct Tensor {
	std::vector<int64> dimensions;
	std::vector<Tensor*> inputs;
	std::string alias;
	std::vector<Tensor*> peers;
	int64 size; //physical size in memory (bytes). cannot suppose size is equal to volumne * sizeof(float).
	float* pointer;
	Tensor* gradient;
	int64 volume; //for convenience. It always equals to the product of vector dimensions' each element

	Tensor(std::vector<int64> dims = {}, std::vector<Tensor*> ins = {}, std::string name = std::string(), std::vector<Tensor*> outs = {});
	virtual std::string generate_source_code(DeviceInstance& I) { return std::string(); }
	virtual void run(DeviceInstance& I) {}
	virtual Tensor* generate_gradient(Tensor* next = nullptr) { return nullptr; }
	virtual ~Tensor();

	//generally launch() should NOT be overloaded. It is not declared as final just because of considering for flexibility.
	virtual void launch(std::set<Tensor*>* executed, void* data, void (*functor)(Tensor*, void*) = [](Tensor* tensor, void* data) { tensor->run(*static_cast<DeviceInstance*>(data)); });
	virtual void initialize(DeviceInstance* I = nullptr);

	Tensor& operator * (Tensor& other);
	Tensor& operator + (Tensor& other);
	Tensor& operator += (Tensor& other);
	Tensor& transpose();

	void dependent_on(Tensor* precusor);
	void shape_with(std::vector<int64> sizes);
	void upload(DeviceInstance& I, const std::vector<cl::Event>* preconditions = nullptr);
	void download(DeviceInstance& I, const std::vector<cl::Event>* preconditions = nullptr);

	static std::vector<Tensor*> ALL;
};

Tensor& IterativeOptimizer(std::vector<Tensor*> ins, std::vector<Tensor*> outs, size_t epoch = INT_MAX);
Tensor& StochasticGradientDescentUpdater(Tensor& graph, float eta, float decay = 0, std::string name = "SGD");
Tensor& StochasticGradientDescentUpdater(std::vector<Tensor*> parameters, float eta, float decay = 0, std::string name = "SGD");
Tensor& XavierNormalDistributionInitializer(Tensor& updater, float mu, float sigma);
Tensor& XavierNormalDistributionInitializer(std::vector<Tensor*> parameters, float mu, float sigma);
Tensor& Weight(std::vector<int64> dims = {}, std::string name = "weight", Tensor* input = nullptr);
Tensor& Bias(std::vector<int64> dims = {}, std::string name = "bias", Tensor* input = nullptr);
Tensor& Data(std::vector<int64> dims = {}, Tensor* input = nullptr, std::string name = "data");
Tensor* Gradient(Tensor* target, Tensor* input = nullptr);

Tensor& sigmoid(Tensor& z, std::string name = "");
Tensor& softrelu(Tensor& z, std::string name = "");
Tensor& relu(Tensor& z, std::string name = "");
Tensor& tanh(Tensor& z, std::string name = "");
//Tensor& softmax(Tensor& z, std::string name = "");
Tensor& DropOut(Tensor& data, float probability_dropout, std::string name = "dropout");

Tensor& LinearRegressionLoss(Tensor& y, Tensor& label);
Tensor& SoftmaxLoss(Tensor& y, Tensor& label);

Tensor& FullyConnectedLayer(Tensor& x, int num_hidden, std::string activation_function, std::string name = "FC");
Tensor& FullyConnectedLayer(Tensor& x, Tensor& weight, Tensor* bias, std::string activation_function, std::string name = "");
Tensor& LSTMCell(Tensor& cell, Tensor& hidden, Tensor* gates_data, Tensor& weight_h, Tensor& weight_x, Tensor& x, Tensor* bias, Tensor& lstm, std::string name = "lstmCell");
Tensor& LSTM(Tensor& input, int num_layer, int num_hidden, float dropout = 0, std::string name = "lstm");
Tensor& Embedding(Tensor& input, Tensor& vector_weight, std::string name = "embedding");

Tensor& ConvolutionKernel(Tensor& input/*NHWC*/, int filter_count, int kernel_size, int stride_size = 1, std::string activation_function = "relu", bool use_padding = false, std::string name = "convolution");
Tensor& ConvolutionKernel(Tensor& input/*NHWC*/, Tensor& weight, Tensor* bias, std::vector<int> stride_sizes = {1, 1}, std::string activation_function = "relu", bool use_padding = false, std::string name = "");
Tensor& Pooling(Tensor& input, std::vector<int> pooling_sizes = {2, 2}, std::vector<int> stride_sizes = {}, std::string pooling_type = "max", bool use_padding = false, std::string name = "");
Tensor& Reshape(Tensor& input, std::vector<int64> target_shape, std::string name = "reshape");

std::string type_name(Tensor* tensor);
Tensor* locate_tensor(std::string name);
void save_tensor(Tensor* tensor, std::ostream& os, DeviceInstance& I);
std::vector<Tensor*> load_tensors(std::istream& is, DeviceInstance& I);
void save_tensor_as_csv(Tensor* tensor, const std::string& file, DeviceInstance* I = nullptr, bool use_header = true);
Tensor* load_tensor_from_csv(const std::string& file, Tensor* tensor = nullptr, DeviceInstance* I = nullptr);
void generate_all_gradients(Tensor* graph);

#define CLNET_FEED_FORWARD_FUSION 1
#define CLNET_BACK_PROPAGATE_FUSION 2
#define CLNET_PREDICT_ONLY 4
#define CLNET_STEP_INTO_MODE 8
#define CLNET_OPENCL_SHOW_SOURCE 16
#define CLNET_IMAGE_CHANNEL_FIRST 32 //currently it is not supported
#define CLNET_VALUE_MISMATCH_WARN 64
#define CLNET_RUN_ON_SINGLE_DEVICE 128
#define CLNET_RUN_ON_DISTRIBUTION 256
extern size_t CLNET_TENSOR_GLOBALS;

// clnet::type **************************************************************************
namespace type {
struct Structured {
	virtual Tensor* body() { return nullptr; }
	virtual std::vector<Tensor*> auxiliaries() { return {}; }
};

struct MiniBatch : Tensor {
	int batch_size, total_batches;
	bool use_shuffle;

	MiniBatch(int batch_size, int total_samples = 0, bool use_shuffle = true);
	void set_total_samples(int N);
	virtual void initialize(DeviceInstance* I) override;
	virtual bool has_next(DeviceInstance& I);
	virtual void reset(DeviceInstance& I);
};

struct IterativeOptimizer : Tensor, type::Structured {
	size_t max_epochs;

	size_t& current_epoch(DeviceInstance& I);
	size_t milliseconds_since_last(DeviceInstance& I);
	virtual void run(DeviceInstance& I) override;
	virtual Tensor* body() override;
	virtual std::vector<Tensor*> auxiliaries() override;
};

struct Updater : Tensor, type::Structured {
	void synchronize_device_parameters(DeviceInstance& I);
	void global_updater_thread(DeviceInstance& I);

	virtual void run_globally(DeviceInstance& I, DeviceInstance& source) {}
	virtual std::vector<Tensor*> auxiliaries() override;
};

struct StochasticGradientDescentUpdater : Updater {
	float learning_rate, weight_decay;

//	StochasticGradientDescentUpdater(std::vector<Tensor*> grads, float eta, float decay);
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
	virtual void run_globally(DeviceInstance& I, DeviceInstance& source) override;
};

struct XavierNormalDistributionInitializer : Tensor, type::Structured {
	float mu, sigma;
	bool initialized = false;
	std::mutex initialization;

	virtual void run(DeviceInstance& I) override;
	virtual void run_globally(DeviceInstance& I);
	virtual std::vector<Tensor*> auxiliaries() override;
};

struct Weight : Tensor {
	virtual Tensor* generate_gradient(Tensor* generator = nullptr) override;
};

struct Bias : Tensor {
	virtual Tensor* generate_gradient(Tensor* generator = nullptr) override;
};

struct Data : Tensor {
	virtual void run(DeviceInstance& I) override;
};

struct Output : Tensor {
	Output(std::vector<int64> dims = {}, std::vector<Tensor*> ins = {}, std::string name = std::string());
	virtual Tensor* generate_gradient(Tensor* generator = nullptr) override;
};

struct Activation : Tensor {
	std::string function;
};

struct BinaryOperator : Tensor {
	std::string function;
	virtual Tensor* generate_gradient(Tensor* generator = nullptr) override;
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
};

struct Transpose : Tensor {
	virtual Tensor* generate_gradient(Tensor* generator = nullptr) override;
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
};

struct DropOut : Tensor {
	float probability_keep;

	virtual Tensor* generate_gradient(Tensor* out_gradient = nullptr) override;
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
	virtual void refresh_random_numbers(DeviceInstance& I, const std::vector<cl::Event>& preconditions);

	static std::default_random_engine generator;
};

struct FullyConnectedLayer : Tensor {
	std::string activation;

	virtual Tensor* generate_gradient(Tensor* out_gradient = nullptr) override;
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
};

struct LSTMCell : Tensor {
	virtual Tensor* generate_gradient(Tensor* out_gradient = nullptr) override;
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
};

struct LSTMInitializer : Tensor, type::Structured {
	LSTMInitializer(std::string name) : Tensor({}, {}, name) {}
	virtual void run(DeviceInstance& I) override;
	virtual std::vector<Tensor*> auxiliaries() override;
};

struct LSTM : Tensor, type::Structured {
	virtual Tensor* generate_gradient(Tensor* out_gradient = nullptr) override;
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
	virtual Tensor* body() override;
	virtual std::vector<Tensor*> auxiliaries() override;
};

struct Embedding : Tensor {
	virtual Tensor* generate_gradient(Tensor* out_gradient = nullptr) override;
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
};

struct ConvolutionKernel : Tensor {
	std::string activation;
	int stride_size[2];

	virtual Tensor* generate_gradient(Tensor* out_gradient = nullptr) override;
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
};

struct Pooling : Tensor {
	std::string type;
	int pooling_size[2], stride_size[2];

	virtual Tensor* generate_gradient(Tensor* out_gradient = nullptr) override;
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
};

struct Reshape : Tensor {
	virtual void initialize(DeviceInstance* I) override;
	virtual Tensor* generate_gradient(Tensor* out_gradient = nullptr) override;
	virtual void run(DeviceInstance& I) override;
	virtual ~Reshape();
};

}

//for convenience
typedef Tensor& T;

class InstantTensor : public Tensor, public type::Structured {
	std::function<std::string (InstantTensor* self, DeviceInstance&)> source_code_function;
	std::function<void (InstantTensor* self, DeviceInstance&)> run_function;
	std::function<Tensor* (InstantTensor* self, Tensor*)> gradient_function;
	std::function<Tensor* (InstantTensor* self)> body_function;
	std::function<std::vector<Tensor*> (InstantTensor* self)> auxiliaries_function;

public:
	InstantTensor(std::string name, std::vector<Tensor*> ins = {}, std::vector<Tensor*> outs = {},
			std::function<void (InstantTensor* self, DeviceInstance&)> run_func = [](InstantTensor* self, DeviceInstance&) {},
			std::function<std::string (InstantTensor* self, DeviceInstance&)> code_func = [](InstantTensor* self, DeviceInstance&) -> std::string { return std::string(); },
			std::function<Tensor* (InstantTensor* self, Tensor*)> gradient_func = [](InstantTensor* self, Tensor*) -> Tensor* { return nullptr; })
			: Tensor({}, ins, name, outs), source_code_function(code_func), run_function(run_func), gradient_function(gradient_func),
			body_function([](InstantTensor* self) -> Tensor* { return self->type::Structured::body(); }), auxiliaries_function([](InstantTensor* self) -> std::vector<Tensor*> { return self->type::Structured::auxiliaries(); }) {}
	InstantTensor(std::string name, std::vector<Tensor*> ins = {},
			std::function<void (InstantTensor* self, DeviceInstance&)> run_func = [](InstantTensor* self, DeviceInstance&) {}, std::vector<Tensor*> outs = {},
			std::function<Tensor* (InstantTensor* self)> body_func = [](InstantTensor* self) -> Tensor* { return self->type::Structured::body(); },
			std::function<std::vector<Tensor*> (InstantTensor* self)> auxiliaries_func = [](InstantTensor* self) -> std::vector<Tensor*> { return self->type::Structured::auxiliaries(); },
			std::function<std::string (InstantTensor* self, DeviceInstance&)> code_func = [](InstantTensor* self, DeviceInstance&) -> std::string { return std::string(); },
			std::function<Tensor* (InstantTensor* self, Tensor*)> gradient_func = [](InstantTensor* self, Tensor*) -> Tensor* { return nullptr; })
			: Tensor({}, ins, name, outs), source_code_function(code_func), run_function(run_func), gradient_function(gradient_func), body_function(body_func), auxiliaries_function(auxiliaries_func) {}

	virtual std::string generate_source_code(DeviceInstance& I) { return source_code_function(this, I); }
	virtual void run(DeviceInstance& I) { run_function(this, I); }
	virtual Tensor* generate_gradient(Tensor* next = nullptr) { return gradient_function(this, next); }
	virtual Tensor* body() { return body_function(this); }
	virtual std::vector<Tensor*> auxiliaries() { return auxiliaries_function(this); }
};

// clnet::back **************************************************************************
namespace back {
struct Gradient : Tensor {
	bool attached = false;
	virtual Tensor* merge_gradient(Tensor* input);
};

struct Loss : Tensor {
	std::string function;

	virtual float L(DeviceInstance& I);
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual Tensor* generate_gradient(Tensor* next = nullptr) override;
	virtual void run(DeviceInstance& I) override;
};

struct kernel_gradient_loss_fully_connected : Tensor {
	std::string loss, activation;

	kernel_gradient_loss_fully_connected(Tensor& weight_grad, Tensor& bias_grad, Tensor& in, Tensor& out, Tensor& label, std::string loss, std::string activation, std::string name = "kernel_gradient_loss_fully_connected");
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
};

struct BinaryOperator : Tensor {
	std::string function;

	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
};

struct kernel_gradient_cascade_fully_connected : Tensor {
	std::string activation;

	kernel_gradient_cascade_fully_connected(Tensor& weight_grad, Tensor& bias_grad, Tensor& in, Tensor& out, Tensor& weight_next, Tensor& nabla_next, std::string activation, std::string name = "kernel_gradient_cascade_fully_connected");
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
};

struct Embedding : Tensor {
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
};

struct LSTMCell : Tensor {
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
};

struct LSTM : Tensor, type::Structured {
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
	virtual Tensor* body() override;
	virtual std::vector<Tensor*> auxiliaries() override;
};

struct FullyConnectedLayer : Gradient {
	std::string activation;

	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
};

struct DropOut : Tensor {
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
};

struct ConvolutionKernel : Tensor {
	struct BackForInput : Tensor {
		virtual std::string generate_source_code(DeviceInstance& I) override;
	};
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
};

struct Pooling : Tensor {
	virtual std::string generate_source_code(DeviceInstance& I) override;
	virtual void run(DeviceInstance& I) override;
};

struct Reshape : Tensor {
	virtual void initialize(DeviceInstance* I) override;
	virtual void run(DeviceInstance& I) override;
	virtual ~Reshape();
};

}

}

#endif /* INCLUDE_TENSOR_HPP_ */
