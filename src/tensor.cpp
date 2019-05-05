/*
 * tensor.cpp
 *
 *  Created on: 2017/5/2
 *      Author: ZhangHua
 */

#include <iostream>
#include <atomic>
#include <stack>
#include <queue>
#include <random>
#include <condition_variable>
#include <mutex>
#include <memory.h>

#include <tensor.hpp>

using namespace std;

namespace clnet {
size_t CLNET_TENSOR_GLOBALS = CLNET_FEED_FORWARD_FUSION | CLNET_BACK_PROPAGATE_FUSION;
vector<Tensor*> Tensor::ALL;
std::default_random_engine type::DropOut::generator;

Tensor::Tensor(vector<int64> dims, vector<Tensor*> ins, string name, vector<Tensor*> outs)
{
	inputs = ins;
	alias = name;
	peers = outs;
	size = 0;
	pointer = nullptr;
	gradient = nullptr;
	shape_with(dims);
	ALL.push_back(this);
}

Tensor::~Tensor()
{
	if (pointer != nullptr)
		delete pointer;
}

Tensor* type::Parameter::generate_gradient(Tensor* generator)
{
	auto back = new back::Gradient;
	back->shape_with(dimensions);
	back->alias = "gradient(" + alias + ")";
	if (generator != nullptr)
		back->inputs.push_back(generator);
	return back;
}

Tensor& IterativeOptimizer(vector<Tensor*> ins, vector<Tensor*> outs, size_t epoch)
{
	auto tensor = new type::IterativeOptimizer;
	tensor->shape_with({ static_cast<int64>(sizeof(size_t) * 2 / sizeof(float)) });
	tensor->alias = "IterativeOptimizer";
	tensor->max_epochs = epoch;
	tensor->inputs = ins;
	tensor->peers = outs;
	for (size_t i = 1; i < outs.size(); i++)
		outs[i]->peers.push_back(tensor);
	return *tensor;
}

Tensor* type::IterativeOptimizer::body()
{
	return peers[0];
}

vector<Tensor*> type::IterativeOptimizer::auxiliaries()
{
	vector<Tensor*> aux;
	for (size_t i = 2; i < peers.size(); i++)
		aux.push_back(peers[i]);
	return aux;
}

type::StochasticGradientDescentUpdater::StochasticGradientDescentUpdater(vector<Tensor*> gradients, vector<Tensor*> all_parameters, float eta, float decay, float momentum, std::string name)
		: learning_rate(eta), weight_decay(decay), momentum(momentum)
{
	alias = name;
	inputs = gradients;
	peers = all_parameters;
	if (momentum != 0)
		for (auto param : all_parameters)
			dependent_on(new type::Output(param->dimensions, {}, name + "_velocity_" + param->alias));
}

void generate_all_gradients(Tensor* graph)
{
	set<Tensor*> visited;
	queue<Tensor*> queue;
	queue.push(graph);
	while (!queue.empty()) { //prior-root traversal
		auto current = queue.front();
		queue.pop();
		if (visited.count(current) != 0)
			continue;
		visited.insert(current);

		for (auto input : current->inputs) {
			if (input == nullptr)
				continue;
			clnet::Gradient(input, clnet::Gradient(current));
			queue.push(input);
		}
	}
}

Tensor& StochasticGradientDescentUpdater(Tensor& graph, float eta, float decay, float momentum, std::string name)
{
	auto gradient = clnet::Gradient(&graph); //gradient output from loss
	generate_all_gradients(gradient);

	vector<Tensor*> gradients, all_parameters;
	for (auto param : Tensor::ALL) {
		if (dynamic_cast<type::Parameter*>(param) == nullptr)
			continue;
		if (param->gradient == nullptr) {
			if (CLNET_TENSOR_GLOBALS & CLNET_VALUE_MISMATCH_WARN)
				cout << "warning: Gradient of " << param->alias << " not found." << endl;
			continue;
		}
		all_parameters.push_back(param);
	}

	set<Tensor*> visited;
	graph.launch(&visited, &gradients, [](Tensor* param, void* data) { //may part of gradients w.r.t all_parameters
		if (dynamic_cast<type::Parameter*>(param) != nullptr)
			static_cast<vector<Tensor*>*>(data)->push_back(param->gradient);
	});

	return *new type::StochasticGradientDescentUpdater(gradients, all_parameters, eta, decay, momentum, name);
}

Tensor& GeneralInitializer(vector<Tensor*> parameters, float mu, float sigma)
{
	auto tensor = new type::GeneralInitializer;
	tensor->alias = "GeneralInitializer";
	tensor->mu = mu;
	tensor->sigma = sigma;
	tensor->peers = parameters;
	return *tensor;
}

vector<Tensor*> type::GeneralInitializer::auxiliaries()
{
	return peers;
}

void Tensor::dependent_on(Tensor* precusor)
{
	inputs.push_back(precusor);
	if (precusor != nullptr)
		precusor->peers.push_back(this);
}

void Tensor::shape_with(vector<int64> sizes)
{
	dimensions = sizes;
	volume = sizes.empty()? 0 : 1;
	for (int dim : sizes)
		volume *= dim;
}

Tensor& Parameter(vector<int64> dims, string name, Tensor* input, std::string type_hint)
{
	type::Parameter* tensor;
	if (type_hint.find("weight") != string::npos)
		tensor = new type::Weight;
	else if (type_hint.find("bias") != string::npos)
			tensor = new type::Bias;
	else
			tensor = new type::Parameter;
	tensor->shape_with(dims);
	tensor->alias = name;
	tensor->type_hint = type_hint;
	if (input != nullptr)
		tensor->dependent_on(input);
	return *tensor;
}

Tensor& Weight(vector<int64> dims, string name, Tensor* input)
{
	return Parameter(dims, name, input, "weight");
}

Tensor& Bias(vector<int64> dims, string name, Tensor* input)
{
	return Parameter(dims, name, input, "bias");
}

type::Output::Output(vector<int64> dims, vector<Tensor*> ins, string name)
{
	shape_with(dims);
	alias = name;
	for (auto in : ins)
		dependent_on(in);
}

Tensor* Gradient(Tensor* target, Tensor* back_operator)
{
	if (target == nullptr)
		return nullptr;
	auto gradient = target->gradient;
	if (gradient != nullptr) {
		if (back_operator != nullptr && gradient->gradient != back_operator) {
			if (gradient->gradient != nullptr) {
				back_operator->inputs.push_back(gradient->gradient);
				bool inCycle = (CLNET_TENSOR_GLOBALS & CLNET_IN_CYCLE) != 0;
				gradient->gradient->inputs.push_back(inCycle? gradient->gradient : nullptr);
			}
			gradient->gradient = back_operator;

			bool found = false;
			set<Tensor*> visited;
			back_operator->launch(&visited, gradient, [&found](Tensor* param, void* data) {
				found |= static_cast<Tensor*>(data) == param;
			});
			if (!found) //remove redundent dependency or circular dependency
				gradient->inputs.push_back(back_operator);
		}
		return gradient;
	}

	gradient = target->generate_gradient(back_operator);
	if (gradient == nullptr)
		return nullptr;
	gradient->gradient = back_operator;
	target->gradient = gradient;
	return gradient;
}

Tensor& Data(vector<int64> dims, Tensor* input, string name)
{
	auto tensor = new type::Data;
	tensor->shape_with(dims);
	tensor->alias = name;
	tensor->dependent_on(input);
	return *tensor;
}

Tensor& sigmoid(Tensor& z, std::string name)
{
	auto tensor = new type::Activation;
	tensor->function = "sigmoid";
	tensor->alias = name + (name.empty()? "" : "=") + tensor->function + "(" + z.alias + ")";
	tensor->dependent_on(&z);
	auto result = new type::Output(z.dimensions, {tensor}, name.empty()? tensor->function + "(" + z.alias + ")" : name);
	return *result;
}

Tensor& ReLU(Tensor& z, std::string type, std::string name)
{
	auto tensor = new type::Activation;
	tensor->function = type + "relu";
	tensor->alias = name + (name.empty()? "" : "=") + type + "ReLU(" + z.alias + ")";
	tensor->dependent_on(&z);
	auto result = new type::Output(z.dimensions, {tensor}, name.empty()? tensor->function + "(" + z.alias + ")" : name);
	return *result;
}

Tensor& tanh(Tensor& z, std::string name)
{
	auto tensor = new type::Activation;
	tensor->function = "tanh";
	tensor->alias = name + (name.empty()? "" : "=") + tensor->function + "(" + z.alias + ")";
	tensor->dependent_on(&z);
	auto result = new type::Output(z.dimensions, {tensor}, name.empty()? tensor->function + "(" + z.alias + ")" : name);
	return *result;
}

Tensor& Activation(Tensor& z, std::string type)
{
	if (type == "sigmoid")
		return sigmoid(z);
	else if (type == "relu")
		return ReLU(z);
	else if (type == "leakyrelu")
		return ReLU(z, "leaky");
	else if (type == "softrelu")
		return ReLU(z, "soft");
	else if (type == "tanh")
		return tanh(z);
	else
		throw new runtime_error("unknown activation type: " + type);
}

Tensor& Tensor::operator * (Tensor& other)
{
	auto tensor = new type::BinaryOperator;
	tensor->function = "multiply";
	tensor->alias = alias + "*" + other.alias;
	tensor->dependent_on(this);
	tensor->dependent_on(&other);
	auto result = new type::Output({dimensions[0], other.dimensions[0]}, {tensor}, "(" + alias + "*" + other.alias + ")");
	return *result;
}

Tensor& Tensor::operator + (Tensor& other)
{
	auto tensor = new type::BinaryOperator;
	tensor->function = "add";
	tensor->alias = alias + "+" + other.alias;
	tensor->dependent_on(this);
	tensor->dependent_on(&other);
	auto result = new type::Output(dimensions, {tensor}, "(" + alias + "+" + other.alias + ")");
	return *result;
}

Tensor& Tensor::operator += (Tensor& other)
{
	auto tensor = new type::BinaryOperator;
	tensor->function = "plus";
	tensor->alias = alias + "+=" + other.alias;
	tensor->dependent_on(this);
	tensor->dependent_on(&other);
	dependent_on(tensor);
	return *this;
}

Tensor& Tensor::operator = (Tensor& other)
{
	inputs = other.inputs;
	for (auto tensor : other.inputs)
		for (size_t i = 0; i < tensor->peers.size(); i++)
			if (tensor->peers[i] == &other)
				tensor->peers[i] = this;
	if (volume < other.volume)
		shape_with(other.dimensions);
	return *this;
}

//Tensor& Tensor::transpose()
//{
//	auto tensor = new type::Output;
//	tensor->function = "plus";
//	tensor->alias = alias + "+=" + other.alias;
//	tensor->dependent_on(this);
//	tensor->dependent_on(&other);
//	dependent_on(tensor);
//	return *this;
//}

Tensor* type::Output::generate_gradient(Tensor* generator)
{
	auto back = new back::Gradient;
	back->shape_with(dimensions);
	back->alias = "gradient(" + alias + ")";
	if (generator != nullptr)
		back->inputs.push_back(generator);
	return back;
}

Tensor* type::BinaryOperator::generate_gradient(Tensor* generator)
{
	auto back = new back::BinaryOperator;
	back->function = function;
	back->alias = "back:" + alias;
	back->inputs.push_back(generator);
	for (auto tensor : inputs)
		back->peers.push_back(clnet::Gradient(tensor, back));
	if (function == "multipy") {
		back->inputs.push_back(inputs[0]);
		back->inputs.push_back(inputs[1]);
	}
	return back;
}

Tensor& LinearRegressionLoss(Tensor& y, Tensor& label)
{
	auto back = new back::Loss;
	back->function = "linear_regression";
	back->alias = back->function + "(" + y.alias + "," + label.alias + ")";
	back->dependent_on(&y);
	back->dependent_on(&label);
	auto out_gradient = clnet::Gradient(&y, back);
	back->peers.push_back(out_gradient);
	return *back;
}

Tensor& CrossEntropyLoss(Tensor& y, Tensor& label)
{
	auto back = new back::Loss;
	back->function = "negative_log_likelihood";
	string softmax = "softmax(" + y.alias + ")";
	back->alias = back->function + "(" + softmax + "," + label.alias + ")";
	back->dependent_on(&y);
	back->dependent_on(&label);
	auto out_gradient = clnet::Gradient(&y, back);
	back->peers.push_back(out_gradient);
	new type::Output(y.dimensions, {back}, softmax);
	return *back;
}

Tensor& BatchNormalizedLayer(Tensor& input, float epsilon, float momentum, std::string name)
{
	auto tensor = new type::BatchNormalizedLayer;
	auto prefix = name.empty()? name : name + "_";
	Tensor& gamma = Weight({input.dimensions.back()}, prefix + "gamma");
	Tensor& beta = Bias({input.dimensions.back()}, prefix + "beta");
	string formula = gamma.alias + "*normalize(" + input.alias + ")+" + beta.alias;
	if (!name.empty())
		formula = name + "=" + formula;
	tensor->alias = formula;
	tensor->epsilon = epsilon;
	tensor->momentum = momentum;
	tensor->dependent_on(&input);
	tensor->dependent_on(&gamma);
	tensor->dependent_on(&beta);
	auto result = new type::Output(input.dimensions, {tensor}, name.empty()? tensor->alias : name);
	new type::Output(input.dimensions, {tensor}, prefix + "deviation"); //peers[1]
	new type::Output({input.dimensions.back()}, {tensor}, prefix + "std_dev"); //peers[2]
	Parameter({input.dimensions.back()}, prefix + "moving_mean", tensor);
	Parameter({input.dimensions.back()}, prefix + "moving_variance", tensor, "unit");
	return *result;
}

Tensor* type::BatchNormalizedLayer::generate_gradient(Tensor* out_gradient)
{
	auto back = new back::BatchNormalizedLayer;
	back->alias = "back:" + alias;
	back->inputs.push_back(out_gradient);
	back->peers.push_back(clnet::Gradient(inputs[1], back)); //gamma_grad
	back->peers.push_back(clnet::Gradient(inputs[2], back)); //beta_grad
	back->peers.push_back(inputs[1]); //peers[2]: gamma
	back->peers.push_back(inputs[0]); //peers[3]: in
	back->peers.push_back(peers[0]); //peers[4]: out
	auto in_gradient = clnet::Gradient(inputs[0], back);
	back->peers.push_back(in_gradient); //peers[5]: in_gradient
	back->peers.push_back(peers[1]); //peers[6]: deviation
	back->peers.push_back(peers[2]); //peers[7]: std_dev
	back->peers.push_back(clnet::Gradient(peers[1], back)); //peers[8]: deviation_grad
	return back;
}

Tensor& DropOut(Tensor& data, float probability_dropout, std::string name)
{
	auto tensor = new type::DropOut;
	auto mask = new Tensor(data.dimensions, {}, name + "_mask");
	tensor->probability_keep = 1.0f - probability_dropout;
	tensor->alias = name;
	tensor->dependent_on(&data);
	tensor->dependent_on(mask);
	data.dependent_on(tensor);
	return data;
}

Tensor& FullyConnectedLayer(Tensor& x, Tensor& weight, Tensor* bias, string activation_function, string name)
{
	auto tensor = new type::FullyConnectedLayer;
	tensor->activation = activation_function;
	string formula = weight.alias + "*" + x.alias;
	if (bias != nullptr)
		formula += "+" + bias->alias;
	if (!activation_function.empty())
		formula = activation_function + "(" + formula + ")";
	if (!name.empty())
		formula = name + "=" + formula;
	tensor->alias = formula;
	tensor->dependent_on(&x);
	tensor->dependent_on(&weight);
	tensor->dependent_on(bias);
	auto result = new type::Output({}, {tensor}, name.empty()? tensor->alias : name);
	if (!x.dimensions.empty())
		result->shape_with({ x.volume / x.dimensions.back(), weight.dimensions.back() });
	return *result;
}

Tensor& FullyConnectedLayer(Tensor& x, int num_hidden, std::string activation_function, std::string name)
{
	int dim_in = x.dimensions.back();
	Tensor& weight = Weight({dim_in, num_hidden}, name + "_weight");
	Tensor& bias = Bias({num_hidden}, name + "_bias");
	return FullyConnectedLayer(x, weight, &bias, activation_function, name);
}

Tensor* back::Loss::generate_gradient(Tensor* next)
{
	return peers[0];
}

Tensor* type::FullyConnectedLayer::generate_gradient(Tensor* out_gradient)
{
	auto back = new back::FullyConnectedLayer;
	back->activation = activation;
	back->alias = "back:" + alias;
	back->inputs.push_back(out_gradient);
	back->peers.push_back(clnet::Gradient(inputs[1], back)); //weight_grad
	back->peers.push_back(clnet::Gradient(inputs[2], back)); //bias_grad
	back->peers.push_back(inputs[1]); //peers[2]: weight
	back->peers.push_back(inputs[0]); //peers[3]: in
	back->peers.push_back(peers[0]); //peers[4]: out
	auto in_gradient = clnet::Gradient(inputs[0], back);
	back->peers.push_back(in_gradient); //peers[5]: in_gradient
	return back;
}

Tensor& LSTMCell(Tensor& cell, Tensor& hidden, Tensor* gates_data, Tensor& weight_h, Tensor& weight_x, Tensor& x, Tensor* bias, Tensor& cell_no, string name)
{
	Tensor& FC_hidden = FullyConnectedLayer(hidden, weight_h, nullptr/*&Bias(bias.dimensions, bias.alias + "_h")*/, "", name + "_FC_hidden");
	Tensor& FC_input = FullyConnectedLayer(x, weight_x, bias, "", name + "_FC_input");
	FC_hidden += FC_input; //use "+=" instead of "+" to reduce memory footprint
//	Tensor& z = FC_hidden + FC_input;

	auto tensor = new type::LSTMCell;
	tensor->alias = name;
	tensor->dependent_on(&FC_hidden);
//	tensor->inputs.push_back(&z);
	tensor->peers = {&cell, &hidden, gates_data, &cell_no, &weight_h, &weight_x, bias, &x};
	cell.inputs.push_back(tensor);
	hidden.inputs.push_back(tensor);
	return *tensor;
}

Tensor* type::LSTMCell::generate_gradient(Tensor* out_gradient)
{
	auto back = new back::LSTMCell;
	back->alias = "back:" + alias;
	back->inputs.push_back(out_gradient); //inputs[0]: h_grad
	auto in_gradient = clnet::Gradient(inputs[0], back); //z_grad: {batch_size, 4*dim_hidden}
	in_gradient->dependent_on(back); //peers[0]: in_gradient
	back->peers.push_back(peers[2]); //peers[1]: gates_data
	back->peers.push_back(peers[3]); //peers[2]: cell_no
	back->peers.push_back(peers[1]); //peers[3]: previous hidden
	back->peers.push_back(clnet::Gradient(peers[0], back)); //peers[4]: cell_state_grad

	//fully connected fusion version
//	back->peers.push_back(clnet::Gradient(peers[4])); //peers[0]: weight_h_grad, don't use tensor as generator to avoid captured by Updater
//	back->peers.push_back(clnet::Gradient(peers[5])); //peers[1]: weight_x_grad
//	back->peers.push_back(clnet::Gradient(peers[6])); //peers[2]: bias_grad
//	back->peers.push_back(peers[3]); //peers[3]: cell_no
//	back->peers.push_back(peers[2]); //peers[4]: gates_data
//	back->peers.push_back(peers[7]); //peers[5]: x
//	back->peers.push_back(peers[4]); //peers[6]: weight_h
//	back->peers.push_back(clnet::Gradient(peers[7], back)); //peers[7]: x_grad
//	back->peers.push_back(peers[5]); //peers[8]: weight_x
	return back;
}

Tensor& LSTM(Tensor& input, int num_layer, int num_hidden, float dropout, string name)
{
	auto tensor = new type::LSTM;
	tensor->alias = "LSTM(" + input.alias + ")";
	int batch_size = input.dimensions[0];
	int sequence_length_max = input.dimensions[1];
	tensor->dependent_on(&input); //inputs[0]: input
	bool predictOnly = CLNET_TENSOR_GLOBALS & CLNET_PREDICT_ONLY;
	Tensor* next = predictOnly? &input : new type::Output({batch_size, input.dimensions[2]}, {}, name + "_input_timestep"); //one step from input
	tensor->peers.push_back(next); //peers[0]: input_timestep
	auto cell_no = new Tensor({3}, {}, name + "_runtime_cell_no"); //store data for cell_no, don't use name prefix 'cell' otherwise conflicting with Tensor& cell

	auto initializer = new type::LSTMInitializer(name + "_initializer");
	tensor->inputs.push_back(initializer); //inputs[1]: initializer
	auto gates_data = predictOnly? nullptr : new Tensor({sequence_length_max, num_layer, 7 * batch_size * num_hidden}, {}, name + "_gates_data");
	for (int l = 0; l < num_layer; l++) {
		const auto& layer = to_string(l);
		Tensor& weight_h = Weight({num_hidden, 4 * num_hidden}, name + "_weight_h" + layer);
		Tensor& weight_x = Weight({next->dimensions.back(), 4 * num_hidden}, name + "_weight_x" + layer);
		Tensor& bias = Bias({4 * num_hidden}, name + "_bias" + layer);
		auto cell_state = new type::Output({batch_size, num_hidden}, {}, name + "_cell_state" + layer);
		auto hidden = new type::Output({batch_size, num_hidden}, {}, name + "_hidden" + layer);
		initializer->peers.push_back(cell_state);
		initializer->peers.push_back(hidden);
		LSTMCell(*cell_state, *hidden, gates_data, weight_h, weight_x, *next, &bias, *cell_no, name + "_cell" + layer);
		next = &DropOut(*hidden, dropout, name + "_dropout" + layer);
	}

	if (predictOnly) {
		next->peers.push_back(initializer); //peers[2]: put here for return. needed for initializing hidden
		return *next;
	}
	tensor->peers.push_back(next); //peers[1]: output
	tensor->peers.push_back(cell_no); //peers[2]: cell_no
	return *new type::Output({batch_size, sequence_length_max, num_hidden}, {tensor}, name); //peers[3]: collect all outpus
}

Tensor* type::LSTM::generate_gradient(Tensor* out_gradient)
{
	auto back = new back::LSTM;
	back->alias = "back:" + alias;
	back->inputs.push_back(inputs[0]); //inputs[0]: x
	back->inputs.push_back(out_gradient); //inputs[1]: out_grad
	CLNET_TENSOR_GLOBALS |= CLNET_IN_CYCLE;
	generate_all_gradients(peers[1]); //peers[1]: out
	CLNET_TENSOR_GLOBALS ^= CLNET_IN_CYCLE;
	// prepare weight/bias gradients and/or build back-propagation-through-time dependency
	auto intput_timestamp_gradient = clnet::Gradient(peers[0]);
	set<Tensor*> visited;
	peers[1]->launch(&visited, intput_timestamp_gradient, [](Tensor* param, void* data) {
		if (dynamic_cast<type::Parameter*>(param) != nullptr)
			static_cast<Tensor*>(data)->dependent_on(param->gradient);
	});
	back->peers.push_back(clnet::Gradient(inputs[0], back)); //peers[0]: x_grad
	back->peers.push_back(intput_timestamp_gradient); //peers[1]: gradient of intput_timestamp
	back->peers.push_back(clnet::Gradient(peers[1])); //peers[2]: hidden_grad
	back->peers.push_back(peers[2]); //peers[3]: runtime_cell_no
	back->peers.push_back(peers[0]); //peers[4]: intput_timestamp

	auto initializer = new type::LSTMInitializer(alias + "_gradient_initializer");
	for (auto tensor : inputs[1]->peers) {
		auto gradient = clnet::Gradient(tensor);
		if (gradient != nullptr)
			initializer->peers.push_back(gradient);
	}
	back->inputs.push_back(initializer);
	return back;
}

vector<Tensor*> type::Updater::auxiliaries()
{
	return peers;
}

Tensor* type::LSTM::body()
{
	return peers[1];
}

vector<Tensor*> type::LSTM::auxiliaries()
{
	return {peers[2]};
}

Tensor* back::LSTM::body()
{
	return peers[1];
}

vector<Tensor*> back::LSTM::auxiliaries()
{
	return {peers[0], peers[3]};
}

Tensor& Embedding(Tensor& input, Tensor& vector_weight, string name)
{
	auto tensor = new type::Embedding;
	tensor->alias = "Embedding(" + input.alias + ")";
	auto result = new type::Output({input.dimensions[0], input.dimensions[1], vector_weight.dimensions[1]}, {tensor}, name);
	tensor->dependent_on(&input);
	tensor->dependent_on(&vector_weight);
	return *result;
}

Tensor* type::Embedding::generate_gradient(Tensor* out_gradient)
{
	auto back = new back::Embedding; //doesn't propagate gradient to input
	back->alias = "back:" + alias;
	back->inputs.push_back(inputs[0]); //input
	back->inputs.push_back(out_gradient);
	back->peers.push_back(clnet::Gradient(inputs[1], back)); //vector_weight_grad
	return back;
}

Tensor* type::DropOut::generate_gradient(Tensor* out_gradient)
{
	auto back = new back::DropOut;
	back->alias = "back:" + alias;
	out_gradient->inputs.push_back(back);
	back->inputs.push_back(out_gradient);
	back->peers.push_back(this);
	return back;
}

Tensor& ConvolutionLayer(Tensor& input/*NHWC*/, int filter_count, int kernel_size, int stride_size, string activation_function, bool use_padding, bool use_bias, string name)
{
	Tensor& weight = Weight({filter_count, kernel_size, kernel_size, input.dimensions[3]}, name + "_weight");

	return ConvolutionLayer(input, weight, use_bias? &Bias({filter_count}, name + "_bias") : nullptr, {stride_size, stride_size}, activation_function, use_padding, name);
}

Tensor& ConvolutionLayer(Tensor& input/*NHWC*/, Tensor& weight, Tensor* bias, vector<int> stride_sizes, string activation_function, bool use_padding, string name)
{
	auto tensor = new type::ConvolutionLayer;
	tensor->activation = activation_function;
	string formula;
	if (!name.empty())
		formula = name + "=" + formula;
	formula += "Convolution:" + to_string(weight.dimensions[1]) + "x" + to_string(weight.dimensions[1]) + "(" + input.alias;
	if (!activation_function.empty())
		formula += "," + activation_function;
	formula += ")";
	tensor->alias = formula;
	tensor->stride_size[0] = stride_sizes.size() > 0? stride_sizes[0] : 1;
	tensor->stride_size[1] = stride_sizes.size() > 1? stride_sizes[1] : tensor->stride_size[0];
	int output_height, output_width;
	if (use_padding) {
		output_height = (input.dimensions[1] + tensor->stride_size[0] - 1) / tensor->stride_size[0];
		output_width = (input.dimensions[2] + tensor->stride_size[1] - 1) / tensor->stride_size[1];
	}
	else {
		output_height = (input.dimensions[1] - weight.dimensions[1] + tensor->stride_size[0]) / tensor->stride_size[0];
		output_width = (input.dimensions[2] - weight.dimensions[2] + tensor->stride_size[1]) / tensor->stride_size[1];
	}
	vector<int64> dims = {input.dimensions[0], output_height, output_width, weight.dimensions[0]};

	auto result = new type::Output(dims, {tensor}, name);
	tensor->dependent_on(&input);
	tensor->dependent_on(&weight);
	tensor->dependent_on(bias);
	return *result;
}

Tensor* type::ConvolutionLayer::generate_gradient(Tensor* out_gradient)
{
	auto back = new back::ConvolutionLayer;
	back->alias = "back:" + alias;
	back->inputs.push_back(out_gradient);
	back->peers.push_back(this); //peers[0]
	back->peers.push_back(clnet::Gradient(inputs[0], back)); //peers[1]: in_grad
	back->peers.push_back(clnet::Gradient(inputs[1], back)); //peers[2]: weight_grad
	back->peers.push_back(clnet::Gradient(inputs[2], back)); //peers[3]: bias_grad
	back->peers.push_back(inputs[0]); //peers[4]: input
	back->peers.push_back(peers[0]); //peers[5]: output
	back->peers.push_back(inputs[1]); //peers[6]: weight
	return back;
}

Tensor& Pooling(Tensor& input, vector<int> pooling_sizes, vector<int> stride_sizes, string pooling_type, bool use_padding, string name)
{
	auto tensor = new type::Pooling;
	tensor->type = pooling_type;
	string formula;
	if (!name.empty())
		formula = name + "=" + formula;
	formula += "Pooling(" + input.alias + "," + pooling_type + ")";
	tensor->alias = formula;
	tensor->pooling_size[0] = pooling_sizes[0];
	tensor->pooling_size[1] = pooling_sizes.size() > 1? pooling_sizes[1] : pooling_sizes[0];
	tensor->stride_size[0] = stride_sizes.size() > 0? stride_sizes[0] : pooling_sizes[0];
	tensor->stride_size[1] = stride_sizes.size() > 1? stride_sizes[1] : tensor->stride_size[0];
	int output_height, output_width;
	if (use_padding) {
		output_height = (input.dimensions[1] + tensor->stride_size[0] - 1) / tensor->stride_size[0];
		output_width = (input.dimensions[2] + tensor->stride_size[1] - 1) / tensor->stride_size[1];
	}
	else {
		output_height = (input.dimensions[1] - tensor->pooling_size[0] + tensor->stride_size[0]) / tensor->stride_size[0];
		output_width = (input.dimensions[2] - tensor->pooling_size[1] + tensor->stride_size[1]) / tensor->stride_size[1];
	}
	vector<int64> dims = {input.dimensions[0]/*batch*/, output_height, output_width, input.dimensions[3]/*filter*/};

	auto result = new type::Output(dims, {tensor}, name);
	tensor->dependent_on(&input);
	if (pooling_type == "max")
		tensor->peers.push_back(new Tensor(dims, {}, name + "_max_index")); //peers[1]
	return *result;
}

Tensor* type::Pooling::generate_gradient(Tensor* out_gradient)
{
	auto back = new back::Pooling;
	back->alias = "back:" + alias;
	back->inputs.push_back(out_gradient);
	back->peers.push_back(this); //peers[0]
	back->peers.push_back(clnet::Gradient(inputs[0], back)); //peers[1]:in_grad
	if (type == "max")
		back->peers.push_back(peers[1]); //peers[2]
	return back;
}

Tensor& Reshape(Tensor& input, vector<int64> target_shape, string name)
{
	auto tensor = new type::Reshape;
	tensor->shape_with(target_shape);
	tensor->alias = name;
	tensor->dependent_on(&input);
	return *tensor;
}

Tensor* type::Reshape::generate_gradient(Tensor* out_gradient)
{
	auto back = new back::Reshape;
	back->shape_with(dimensions);
	back->alias = "gradient(" + alias + ")";
	back->inputs.push_back(out_gradient);
	auto in_gradient = clnet::Gradient(inputs[0], back);
	back->peers.push_back(in_gradient);
	if (in_gradient == nullptr)
		return nullptr;
	return back;
}

Tensor* type::Activation::generate_gradient(Tensor* out_gradient)
{
	auto back = new back::Activation;
	back->function = function;
	back->alias = "gradient(" + alias + ")";
	back->inputs.push_back(out_gradient);
	back->peers.push_back(peers[0]); //peers[0]: out
	auto in_gradient = clnet::Gradient(inputs[0], back);
	back->peers.push_back(in_gradient); //peers[1]: in_gradient
	return back;
}

Tensor& Softmax(Tensor& z, std::string name)
{
	auto tensor = new type::Softmax;
	tensor->alias = name + (name.empty()? "" : "=") + "Softmax(" + z.alias + ")";
	tensor->dependent_on(&z);
	auto result = new type::Output(z.dimensions, {tensor}, name.empty()? "Softmax(" + z.alias + ")" : name);
	return *result;
}

Tensor& Concatenate(vector<Tensor*> parts, int axis, std::string name)
{
	auto tensor = new type::Concatenate;
	string formula = "Concatenate(" + parts[0]->alias;
	for (size_t i = 1; i < parts.size(); i++)
		formula.append(",").append(parts[i]->alias);
	formula.append(")");

	if (axis < 0)
		axis += parts[0]->dimensions.size();
	vector<int64> dims(parts[0]->dimensions);
	for (size_t i = 1; i < parts.size(); i++)
		dims[axis] += parts[0]->dimensions[axis];
	auto result = new type::Output(dims, {tensor}, name.empty()? formula : name);

	if (!name.empty())
		formula = name + "=" + formula;
	tensor->alias = formula;
	tensor->axis = axis;
	tensor->inputs = parts;
	return *result;
}

Tensor* type::Concatenate::generate_gradient(Tensor* out_gradient)
{
	auto back = new type::Split;
	back->alias = "back:" + alias;
	back->axis = axis;
	back->inputs.push_back(out_gradient);
	for (auto input : inputs)
		back->peers.push_back(clnet::Gradient(input, back)); //in_grad
	return back;
}

vector<Tensor*> Split(Tensor& input, int number, int axis, string name)
{
	int size = (int) input.dimensions.back();
	int length = size / number;
	vector<int64> dims;
	for (int i = 0; i < number; i++)
		dims.push_back(length);
	dims.back() += size % number;
	return Split(input, dims, axis, name);
}

vector<Tensor*> Split(Tensor& input, vector<int64> lengths, int axis, std::string name)
{
	auto tensor = new type::Split;
	string formula = "Split(" + input.alias + ",axis:" + to_string(axis) + "," + to_string(lengths.size()) + ")";
	if (!name.empty())
		formula = name + "=" + formula;
	tensor->alias = formula;
	tensor->inputs.push_back(&input);
	if (axis < 0)
		axis += input.dimensions.size();
	tensor->axis = axis;
	vector<int64> dims(input.dimensions);
	for (size_t i = 0; i < lengths.size(); i++) {
		dims[axis] = lengths[i];
		new type::Output(dims, {tensor}, name + "(" + to_string(i) + ")");
	}
	return tensor->peers;
}

Tensor* type::Split::generate_gradient(Tensor* out_gradient)
{
	auto back = new type::Concatenate;
	back->alias = "back:" + alias;
	back->axis = axis;
	for (auto output : peers)
		back->inputs.push_back(clnet::Gradient(output)); //out_grad
	peers.push_back(clnet::Gradient(inputs[0], back));
	return back;
}

Tensor& Collector(Tensor& input, int64 size, string name)
{
	auto tensor = new type::Collector;
	tensor->alias = "Collector(" + input.alias + ")";
	tensor->inputs.push_back(&input);
	tensor->shape_with({sizeof(int64) / sizeof(float)});
	vector<int64> dims(input.dimensions);
	dims.front() = size;
	auto result = new type::Output(dims, {tensor}, name.empty()? tensor->alias : name + "=" + tensor->alias);
	return *result;
}

}
