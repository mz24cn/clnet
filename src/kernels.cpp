/*
 * kernels.cpp
 *
 *  Created on: 2017/2/20
 *      Author: ZhangHua
 */

#include <vector>
#include <memory>
#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <sstream>
#include <set>
#include <omp.h>

#include <tensor.hpp>
#include <device_instance.hpp>

using namespace std;

namespace clnet {
const char KERNEL_FILE[] = "kernels.cl";
const string KERNEL_DEF = "kernel void ";
const vector<cl::Event> no_preconditions;

unordered_map<string, string> kernels_source;
unordered_map<Tensor*, size_t> kernels_cost;

extern string cl_build_options;
extern condition_variable breakpoint;
extern unique_lock<mutex> breakpoint_lock;
Tensor *_current, *_breakpoint = nullptr;
size_t microseconds = 0, breakpoint_hit_times = 1;

template <typename T> bool read_file_content(const string file, basic_string<T>& content)
{
	basic_ifstream<T> ifs(file, ios::binary);
	if (ifs) {
		basic_ostringstream<T> os;
		os << ifs.rdbuf();
		content.append(os.str());
		return true;
	}
	return false;
}

template bool read_file_content<wchar_t>(const string file, basic_string<wchar_t>& content);

#define replace_once(str, key, value) str = str.replace(str.find(key), sizeof(key) - 1, value)

void replace_all(string& content, const string key, const string replace)
{
	string::size_type pos = content.find(key), length = key.size(), span = replace.size();
	while(pos != string::npos) {
		content.replace(pos, length, replace);
		pos = content.find(key, pos + span);
	}
}

void load_kernel_source(const string file)
{
	string source;
	if (!read_file_content<char>(OpenCL.location + file, source)
		&& !read_file_content<char>(OpenCL.location + "kernels/" + file, source)
		&& !read_file_content<char>(OpenCL.location + "src/" + file, source)) //custom for visual studio x64
		throw runtime_error("kernel file " + file + " not found.");
	auto pos = source.find(KERNEL_DEF);
	auto end = source.rfind("}\n", pos) + 2;
	string name = "_header";
	string code = source.substr(0, end);
	kernels_source[name] = code;

	while (pos != string::npos) {
		string::size_type begin = pos, name_begin = begin + KERNEL_DEF.size();
		pos = source.find(KERNEL_DEF, name_begin); //next kernel begins
		if (pos == string::npos)
			end = source.length();
		else
			end = source.rfind("}\n", pos) + 2;
		auto name_end = source.find("(", name_begin);
		name = source.substr(name_begin, name_end - name_begin);
		code = source.substr(begin, end - begin);
		kernels_source[name] = code;
	}
}

string generate_kernel_sources(DeviceInstance& I, const cl::Device& device, unordered_map<Tensor*, string>& tensor_kernels)
{
	load_kernel_source(KERNEL_FILE);

	string name = "_header";
	string source;
	string sources = kernels_source[name];
	set<string> kernels; //Note: use c++0x in debug mode. MinGW/GCC has a bug when using set<string> and unordered_map.operator[] simultaneously under c++1y.
	for (auto tensor : Tensor::ALL) {
		source = tensor->generate_source_code(I);

		if (source.empty())
			continue;
		int begin = source.find(KERNEL_DEF) + KERNEL_DEF.size();
		int end = source.find("(");
		name = source.substr(begin, end - begin);
		if (kernels.count(name) == 0) {
			kernels.insert(name);
			sources.append("\n").append(source);
		}
//		logger << "****** " << name << endl << code << endl;
		tensor_kernels[tensor] = name;
	}
	return sources;
}

void Tensor::launch(std::set<Tensor*>* executed, void* data, void (*functor)(Tensor*, void*))
{
	if (executed->count(this) == 0) {
		executed->insert(this);
		for (auto tensor : inputs)
			if (tensor != nullptr)
				tensor->launch(executed, data, functor);

		_current = this;
		if (CLNET_TENSOR_GLOBALS & CLNET_STEP_INTO_MODE) {
			_breakpoint = this;
			breakpoint_hit_times = 1;
		}
		if (this == _breakpoint && --breakpoint_hit_times == 0) {
			int device_id = static_cast<DeviceInstance*>(data)->ID;
			logger << "[debugger] device " << device_id << " break on " << alias << ": " << type_name(this) << endl;
			breakpoint.wait(breakpoint_lock); //No considering spurious wake-up
			logger << "[debugger] device " << device_id << " continue to run." << endl;
		}
		if (microseconds > 0)
			microseconds = MICROS();
		functor(this, data);
		if (microseconds > 0) {
			wait_for_all_kernels_finished(*static_cast<DeviceInstance*>(data));
			auto duration = MICROS(microseconds);
			kernels_cost[this] += duration;
//			logger << type_name(this) << " " << alias << ": " << duration << "¦Ìs" << endl;
		}
	}
}

void Tensor::upload(DeviceInstance& I, const vector<cl::Event>* preconditions)
{
	if (size <= 0) //size=0: CL_INVALID_VALUE for clEnqueueWriteBuffer
		return;
	I.queue.enqueueReadBuffer(I.buffers[this], preconditions == nullptr, 0, size, I.pointers[this], preconditions, &I.events[this]);
}

void Tensor::download(DeviceInstance& I, const vector<cl::Event>* preconditions)
{
	if (size <= 0) //size=0: CL_INVALID_VALUE for clEnqueueWriteBuffer
		return;
	I.queue.enqueueWriteBuffer(I.buffers[this], preconditions == nullptr, 0, size, I.pointers[this], preconditions, &I.events[this]);
}

inline int find_proper_local_size(int required, int work_group_size)
{
	if (required < work_group_size) {
		int parallel = 1;
		while (parallel < required)
			parallel <<= 1;
		return parallel;
	}
	else
		return work_group_size;
}

inline cl::Kernel& prepare_for_running_kernel(Tensor* tensor, DeviceInstance& I)
{
	I.precondition_events.clear();
	for (auto input : tensor->inputs)
		if (input != nullptr && input->volume > 0) //exclude "pure kernel" tensor
			I.precondition_events.push_back(I.events[input]);
	return I.kernels[tensor];
}

#define FULLY_CONNECTED_STD_DIM_IN_UPLIMIT 64
string type::FullyConnectedLayer::generate_source_code(DeviceInstance& I)
{
	int dim_in = inputs[1]->dimensions[1];
	string code; //make a copy
	if (dim_in < FULLY_CONNECTED_STD_DIM_IN_UPLIMIT) {
		//Parallel: (batch_size * dim_hidden)
		code = kernels_source["feed_forward_fully_connected_sigmoid"];
		if (dim_in > 0) {
			auto dim_in_str = to_string(dim_in);
			replace_once(code, "feed_forward_fully_connected_sigmoid", "feed_forward_fully_connected_sigmoid_" + dim_in_str);
			replace_once(code, "#pragma unroll", "#pragma unroll dim_in");
			replace_once(code, "int dim_in", "int _unused");
			replace_all(code, "dim_in", dim_in_str);
		}
		if (activation != "sigmoid")
			replace_all(code, "sigmoid", activation);
	}
	else {
		//Parallel: (batch_size * dim_hidden * get_local_size(0))
		//Note: choose local NDRange size near (2 * dim_in) when enqueue ASAP
		code = kernels_source["feed_forward_fully_connected_softrelu"];
		if (activation != "softrelu")
			replace_all(code, "softrelu", activation);
	}

	return code;
}

void type::FullyConnectedLayer::run(DeviceInstance& I)
{
	auto& kernel = prepare_for_running_kernel(this, I);
	int N = inputs[0]->volume / inputs[0]->dimensions.back();
	int HIDDEN = inputs[1]->dimensions[0];
	int dim_in = inputs[1]->dimensions[1];
	kernel.setArg(0, I.buffers[peers[0]]);
	kernel.setArg(1, I.buffers[inputs[0]]);
	kernel.setArg(2, I.buffers[inputs[1]]);
	if (inputs[2] != nullptr)
		kernel.setArg(3, I.buffers[inputs[2]]);
	else
		kernel.setArg(3, nullptr);

	int parallel = find_proper_local_size(dim_in, I.work_group_size);
	cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * parallel);
	kernel.setArg(4, tmpMem);
	kernel.setArg(5, HIDDEN);
	kernel.setArg(6, dim_in);
	if (dim_in < FULLY_CONNECTED_STD_DIM_IN_UPLIMIT) {
		cl::NDRange global(N * HIDDEN);
		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[peers[0]]);
	}
	else {
		cl::NDRange local(parallel);
		cl::NDRange global(N * HIDDEN * parallel);
		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, &I.precondition_events, &I.events[peers[0]]);
	}
}

string type::DropOut::generate_source_code(DeviceInstance& I)
{
	string code = kernels_source["feed_forward_dropout"];
	return code;
}

void type::DropOut::run(DeviceInstance& I)
{
	if (probability_keep == 1)
		return;
	auto data = inputs[0], mask = inputs[1];
	if (I.buffers.count(this) == 0) { //trick: runtime initializing
		refresh_random_numbers(I, no_preconditions);
		I.buffers[this] = I.buffers[data];
	}
	auto& kernel = prepare_for_running_kernel(this, I);
	int batch_size = data->dimensions[0];
	int num_hidden = data->volume / batch_size;
	auto& mask_buffer = I.buffers[mask];
	kernel.setArg(0, I.buffers[data]);
	kernel.setArg(1, mask_buffer);

	cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * I.work_group_size);
	kernel.setArg(2, tmpMem);
	kernel.setArg(3, num_hidden);
	kernel.setArg(4, probability_keep);
	kernel.setArg(5, batch_size);

	cl::NDRange global(num_hidden);
	I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[data]);
//	I.precondition_events.clear();
//	I.precondition_events.push_back(I.events[data]);
//	I.events[mask].wait(); //should wait for last download to be finished
//	refresh_random_numbers(I, I.precondition_events);
}

void type::DropOut::refresh_random_numbers(DeviceInstance& I, const vector<cl::Event>& preconditions)
{
	bernoulli_distribution distribution(probability_keep);
	auto mask = inputs[1];
	int N = mask->volume;
	for (float *p = I.pointers[mask], *end = p + N; p < end; p++)
		*p = distribution(generator)? 1.0f : 0;
	mask->download(I, &preconditions);
}

string back::DropOut::generate_source_code(DeviceInstance& I)
{
	string code = kernels_source["back_propagate_dropout"];
	return code;
}

void back::DropOut::run(DeviceInstance& I)
{
	type::DropOut* dropout = static_cast<type::DropOut*>(peers[0]);
	if (dropout->probability_keep == 1)
		return;
	auto data = dropout->inputs[0], mask = dropout->inputs[1];
	auto& kernel = prepare_for_running_kernel(this, I);
	int batch_size = data->dimensions[0];
	int num_hidden = data->volume / batch_size;
	auto& mask_buffer = I.buffers[mask];
	kernel.setArg(0, I.buffers[data]);
	kernel.setArg(1, mask_buffer);

	cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * I.work_group_size);
	kernel.setArg(2, tmpMem);
	kernel.setArg(3, num_hidden);
	kernel.setArg(4, dropout->probability_keep);
	kernel.setArg(5, batch_size);
	kernel.setArg(6, 0); //max_norm

	cl::NDRange global(num_hidden);
	I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[data]);
	I.precondition_events.clear();
	I.precondition_events.push_back(I.events[data]);
	I.events[mask].wait(); //should wait for last download to be finished
	dropout->refresh_random_numbers(I, I.precondition_events);
}

string back::FullyConnectedLayer::generate_source_code(DeviceInstance& I)
{
	string code = kernels_source["back_propagate_fully_connected_softrelu_gradient"];
	if (activation != "softrelu")
		replace_all(code, "softrelu", activation);

	return code;
}

void back::FullyConnectedLayer::run(DeviceInstance& I)
{
	auto& kernel = prepare_for_running_kernel(this, I);
	int N = peers[3]->volume / peers[3]->dimensions.back();
	int dim_in = peers[2]->dimensions[1];
	int dim_out = peers[2]->dimensions[0];
	auto in_gradient = peers[5];
	if (in_gradient != nullptr)
		kernel.setArg(0, I.buffers[in_gradient]);
	else
		kernel.setArg(0, nullptr);
	kernel.setArg(1, I.buffers[peers[0]]);
	auto bias = peers[1];
	if (bias != nullptr)
		kernel.setArg(2, I.buffers[bias]);
	else
		kernel.setArg(2, nullptr);
	kernel.setArg(3, I.buffers[peers[2]]);
	kernel.setArg(4, I.buffers[peers[3]]);
	kernel.setArg(5, I.buffers[peers[4]]);
	kernel.setArg(6, I.buffers[inputs[0]]);

	int global_size = (N > dim_out? N : dim_out) * dim_in;
	kernel.setArg(7, dim_out);
	kernel.setArg(8, dim_in);
	kernel.setArg(9, N);
	kernel.setArg(10, attached? 1 : 0); //merge_gradient mode

	cl::Event& event = I.events[peers[0]];
	cl::NDRange global(global_size);
	I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &event);
	if (in_gradient != nullptr)
		I.events[in_gradient] = event;
	if (bias != nullptr)
		I.events[bias] = event;
}

string back::kernel_gradient_cascade_fully_connected::generate_source_code(DeviceInstance& I)
{
	string code = kernels_source["back_propagate_cascade_fully_connected_sigmoid_gradient"];
	if (activation != "sigmoid")
		replace_all(code, "sigmoid", activation);

	return code;
}

void back::kernel_gradient_cascade_fully_connected::run(DeviceInstance& I)
{
	auto& kernel = prepare_for_running_kernel(this, I);
	int batch_size = inputs[0]->dimensions[0];
	int dim_in = inputs[0]->dimensions[1];
	int dim_out = inputs[1]->dimensions[1];
	int dim_weight_next_out = inputs[2]->dimensions[0];
	kernel.setArg(0, I.buffers[peers[0]]);
	if (peers[1] != nullptr)
		kernel.setArg(1, I.buffers[peers[1]]);
	else
		kernel.setArg(1, nullptr);
	kernel.setArg(2, I.buffers[inputs[0]]);
	kernel.setArg(3, I.buffers[inputs[1]]);
	kernel.setArg(4, I.buffers[inputs[2]]);
	kernel.setArg(5, I.buffers[inputs[3]]);

	cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * I.work_group_size);
	kernel.setArg(6, tmpMem);
	kernel.setArg(7, dim_out);
	kernel.setArg(8, dim_in);
	kernel.setArg(9, dim_weight_next_out);
	kernel.setArg(10, batch_size);

	cl::NDRange global(inputs[0]->dimensions[0] * inputs[0]->dimensions[1]);
	I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[peers[0]]);
	if (peers[1] != nullptr)
		I.events[peers[1]] = I.events[peers[0]];
}

string back::kernel_gradient_loss_fully_connected::generate_source_code(DeviceInstance& I)
{
	string name = "back_propagate_" + loss + "_loss_softrelu_gradient";
	string code = kernels_source[name];
	if (activation != "softrelu")
		replace_all(code, "softrelu", activation);

	return code;
}

void back::kernel_gradient_loss_fully_connected::run(DeviceInstance& I)
{
	auto& kernel = prepare_for_running_kernel(this, I);
	int batch_size = inputs[0]->dimensions[0];
	int dim_in = inputs[0]->dimensions[1];
	int dim_out = inputs[1]->dimensions[1];
	kernel.setArg(0, I.buffers[peers[0]]);
	if (peers[1] != nullptr)
		kernel.setArg(1, I.buffers[peers[1]]);
	else
		kernel.setArg(1, nullptr);
	kernel.setArg(2, I.buffers[inputs[0]]);
	kernel.setArg(3, I.buffers[inputs[1]]);
	kernel.setArg(4, I.buffers[inputs[2]]);

	cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * I.work_group_size);
	kernel.setArg(5, tmpMem);
	kernel.setArg(6, dim_out);
	kernel.setArg(7, dim_in);
	kernel.setArg(8, batch_size);

	if (I.work_group_size < batch_size * 2)
		throw cl::Error(USER_GROUP_SIZE_NOT_BIG_ENOUGH);
	cl::NDRange local(batch_size);
	cl::NDRange global(dim_out * dim_in * batch_size);
	I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, &I.precondition_events, &I.events[peers[0]]);
	if (peers[1] != nullptr)
		I.events[peers[1]] = I.events[peers[0]];
}

size_t& type::IterativeOptimizer::current_epoch(DeviceInstance& I)
{
	return reinterpret_cast<size_t*>(I.pointers[this])[0];
}

size_t type::IterativeOptimizer::milliseconds_since_last(DeviceInstance& I)
{
	size_t& millis = reinterpret_cast<size_t*>(I.pointers[this])[1];
	size_t last = millis;
	millis = MILLIS(0);
	return millis - last;
}

void type::IterativeOptimizer::run(DeviceInstance& I)
{
	set<Tensor*> visited;
	auto graph = body(); //peers[0]
	auto others = auxiliaries();
	size_t& epoch = current_epoch(I);
	milliseconds_since_last(I);
	MiniBatch* batcher = dynamic_cast<MiniBatch*>(peers[1]);
	for (; epoch < max_epoch; epoch++) {
		if (batcher == nullptr) {
			visited.clear();
			graph->launch(&visited, &I);
		}
		else
			while (batcher->has_next(I)) {
				visited.clear();
				graph->launch(&visited, &I);
				wait_for_all_kernels_finished(I); //avoid piling up too many events. It's a must for AMD devices.
			}

		for (auto aux : others)
			aux->launch(&visited, &I);
		if (batcher != nullptr)
			batcher->reset(I);

		wait_for_all_kernels_finished(I);
	}
}

string type::StochasticGradientDescentUpdater::generate_source_code(DeviceInstance& I)
{
	return kernels_source["update_parameters_by_stochastic_gradient_descent"];
}

void type::StochasticGradientDescentUpdater::run(DeviceInstance& I)
{
	if (I.gradients_state == static_cast<int>(peers.size()))
		for (auto tensor : peers) { //report gradients
			auto grad = tensor->gradient;
			I.queue.enqueueReadBuffer(I.buffers[grad], CL_FALSE, 0, grad->size, I.pointers[grad], NULL, &I.events[grad]);
			I.events[grad].setCallback(CL_COMPLETE, gradients_event_callback, &I);
		}

	if (I.parameters_state == static_cast<int>(peers.size()))
		for (auto param : peers) { //load parameters
			I.queue.enqueueWriteBuffer(I.buffers[param], CL_FALSE, 0, param->size, I.pointers[param], NULL, &I.events[param]);
			I.events[param].setCallback(CL_COMPLETE, parameters_event_callback, &I);
		}
	else {
		auto& kernel = I.kernels[this];

		for (int i = 0, N = peers.size(); i < N ; i++) {
			I.precondition_events.clear();
			I.precondition_events.push_back(I.events[peers[i]->gradient]);
			kernel.setArg(0, I.buffers[peers[i]]);
			kernel.setArg(1, I.buffers[peers[i]->gradient]);
			kernel.setArg(2, learning_rate);
			kernel.setArg(3, weight_decay);
			cl::NDRange global(peers[i]->volume);
			I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[peers[i]]);
		}
	}
}

void type::StochasticGradientDescentUpdater::run_globally(DeviceInstance& I, DeviceInstance& source, Tensor& global_data)
{
	auto& kernel = I.kernels[this];
	cl::Event event;
	vector<cl::Event> preconditions, updates;
	for (int i = 0, N = peers.size(); i < N; i++) {
		auto tensor = peers[i]->gradient;
		auto gradient = peers[i];
		auto& gradient_buffer = I.buffers[global_data.inputs[i]];
		auto& parameter_buffer = I.buffers[global_data.peers[i]];
		I.queue.enqueueWriteBuffer(gradient_buffer, CL_FALSE, 0, gradient->size, source.pointers[gradient], NULL, &event);
		preconditions.push_back(event);

		kernel.setArg(0, parameter_buffer);
		kernel.setArg(1, gradient_buffer);
		kernel.setArg(2, learning_rate);
		kernel.setArg(3, weight_decay);
		cl::NDRange global(tensor->volume);
		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &preconditions, &event);
		updates.push_back(event);
		preconditions.clear();
	}
	for (auto& ev : updates)
		ev.wait();
}

void type::Data::run(DeviceInstance& I)
{
	download(I, &no_preconditions);
}

void type::XavierNormalDistributionInitializer::run(DeviceInstance& I)
{
	default_random_engine generator;
	for (auto tensor : peers) {
		if (dynamic_cast<Weight*>(tensor) == nullptr && dynamic_cast<Bias*>(tensor) == nullptr)
			continue;

		int fan_in = tensor->dimensions.back();
		normal_distribution<float> distribution(mu, sqrt(sigma / fan_in));
		for (float *p = I.pointers[tensor], *end = p + tensor->volume; p < end; p++)
			*p = (float) distribution(generator);
		tensor->download(I, &no_preconditions);
	}
}

string type::LSTMCell::generate_source_code(DeviceInstance& I)
{
	return kernels_source["feed_forward_LSTM_cell"];
}

void type::LSTMCell::run(DeviceInstance& I)
{
	auto& kernel = prepare_for_running_kernel(this, I);
	auto cell_no = I.pointers[peers[3]];
	int batch_size = peers[1]->dimensions[0];
	int dim_hidden = peers[1]->dimensions[1];
	kernel.setArg(0, I.buffers[peers[0]]); //cell
	kernel.setArg(1, I.buffers[peers[1]]); //hidden
	auto gates_data = peers[2];
	if(gates_data != nullptr)
		kernel.setArg(2, I.buffers[gates_data]);
	else
		kernel.setArg(2, nullptr); //prediction need not gates_data
	kernel.setArg(3, I.buffers[inputs[0]]);

	cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * I.work_group_size);
	kernel.setArg(4, tmpMem);
	kernel.setArg(5, dim_hidden);
	kernel.setArg(6, static_cast<int>(cell_no[0]));

	cl::NDRange global(batch_size * dim_hidden);
	I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[peers[0]]);
	I.events[peers[1]] = I.events[peers[0]];
	cell_no[0] += cell_no[1];
}

string type::LSTM::generate_source_code(DeviceInstance& I)
{
	return kernels_source["feed_forward_LSTM_recurrent"];
}

void type::LSTMInitializer::run(DeviceInstance& I)
{
	for (auto tensor : peers)
		tensor->download(I, &no_preconditions); //clear to zero
}

std::vector<Tensor*> type::LSTMInitializer::auxiliaries()
{
	return peers;
}

void type::LSTM::run(DeviceInstance& I)
{
	auto input = inputs[0], input_timestep = peers[0], output_timestep = body(); //peers[1]
	int length = input->dimensions[1]; //batch as major index

	auto cell_no = I.pointers[peers[2]];
	cell_no[0] = 0;
	cell_no[1] = 1;

	auto& kernel = prepare_for_running_kernel(this, I);
	int dim_input = input->dimensions.back();
	int dim_hidden = output_timestep->dimensions.back();
	kernel.setArg(0, I.buffers[input_timestep]);
	kernel.setArg(1, nullptr);
	kernel.setArg(2, I.buffers[input]);
	kernel.setArg(3, I.buffers[output_timestep]);
	kernel.setArg(4, 0);
	kernel.setArg(5, length);
	kernel.setArg(6, dim_input);
	kernel.setArg(7, dim_hidden);
	cl::NDRange global(input->dimensions[0] * dim_hidden);
	I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[input_timestep]);
	set<Tensor*> visited;
	output_timestep->launch(&visited, &I);

	if (peers.size() > 3) //collect all timestep outputs
		kernel.setArg(1, I.buffers[peers[3]]);
	for (int timestep = 1; timestep < length; timestep++) {
		I.precondition_events.clear();
		I.precondition_events.push_back(I.events[output_timestep]);
		kernel.setArg(4, timestep);
		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[input_timestep]);
		visited.clear();
		output_timestep->launch(&visited, &I);
	}
	if (peers.size() > 3) {
		I.precondition_events.clear();
		I.precondition_events.push_back(I.events[output_timestep]);
		kernel.setArg(4, -length); //negative value means the timestep at the end
		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[peers[3]]);
	}
}

string back::LSTM::generate_source_code(DeviceInstance& I)
{
	return kernels_source["back_propagate_LSTM_recurrent"];
}

void back::LSTM::run(DeviceInstance& I)
{
	auto input = inputs[0];
	auto input_timestep_gradient = body(); //peers[1]
	auto output_timestep_gradient = peers[2];
	int batch_size = input->dimensions[0]; //batch as major index
	int length = input->dimensions[1];
	int dim_input = input_timestep_gradient->dimensions.back();
	int dim_hidden = output_timestep_gradient->dimensions.back();

	auto cell_no = I.pointers[peers[3]];
	cell_no[1] = -1;

	auto& kernel = prepare_for_running_kernel(this, I);
	kernel.setArg(0, I.buffers[output_timestep_gradient]); //hidden_grad
	kernel.setArg(1, I.buffers[peers[0]]); //x_grad
	kernel.setArg(2, I.buffers[peers[4]]); //x_timestep
	kernel.setArg(3, I.buffers[inputs[1]]); //out_grad
	kernel.setArg(4, I.buffers[input_timestep_gradient]); //x_timestep_grad
	kernel.setArg(5, I.buffers[input]); //x
	kernel.setArg(6, -length); //negative value means the timestep at the end
	kernel.setArg(7, length);
	kernel.setArg(8, dim_input);
	kernel.setArg(9, dim_hidden);
	cl::NDRange global(batch_size * dim_hidden);
	I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[output_timestep_gradient]);
	set<Tensor*> visited;
	input_timestep_gradient->launch(&visited, &I);

	for (int timestep = length - 1; timestep > 0; timestep--) {
		I.precondition_events.clear();
		I.precondition_events.push_back(I.events[input_timestep_gradient]);
		kernel.setArg(6, timestep);
		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[output_timestep_gradient]);

		visited.clear();
		input_timestep_gradient->launch(&visited, &I);
	}
	I.precondition_events.clear();
	I.precondition_events.push_back(I.events[input_timestep_gradient]);
	kernel.setArg(3, nullptr); //out_grad
	kernel.setArg(6, 0);
	I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[peers[0]]);
}

string type::Embedding::generate_source_code(DeviceInstance& I)
{
	return kernels_source["feed_forward_embedding"];
}

void type::Embedding::run(DeviceInstance& I)
{
	auto& kernel = prepare_for_running_kernel(this, I);
	int dim_in = inputs[0]->volume;
	int vector_length = inputs[1]->dimensions[1];
	kernel.setArg(0, I.buffers[peers[0]]);
	kernel.setArg(1, I.buffers[inputs[0]]);
	kernel.setArg(2, I.buffers[inputs[1]]);

	int parallel = find_proper_local_size(dim_in, I.work_group_size);
	cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * parallel);
	kernel.setArg(3, tmpMem);
	kernel.setArg(4, dim_in);
	kernel.setArg(5, vector_length);

	cl::NDRange global(parallel * vector_length);
	cl::NDRange local(parallel);
	I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, &I.precondition_events, &I.events[peers[0]]);
}

string back::Embedding::generate_source_code(DeviceInstance& I)
{
	return kernels_source["back_propagate_embedding"];
}

void back::Embedding::run(DeviceInstance& I)
{
	auto& kernel = prepare_for_running_kernel(this, I);
	int dim_in = inputs[0]->volume;
	int vector_num = peers[0]->dimensions[0];
	int vector_length = peers[0]->dimensions[1];
	kernel.setArg(0, I.buffers[peers[0]]);
	kernel.setArg(1, I.buffers[inputs[0]]);
	kernel.setArg(2, I.buffers[inputs[1]]);

	int parallel = find_proper_local_size(dim_in, I.work_group_size);
	cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * parallel);
	kernel.setArg(3, tmpMem);
	kernel.setArg(4, dim_in);
	kernel.setArg(5, vector_length);
	kernel.setArg(6, vector_num);

	cl::NDRange global(vector_length);
	I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[peers[0]]);
}

string back::LSTMCell::generate_source_code(DeviceInstance& I)
{
	return kernels_source["back_propagate_LSTM_cell_gates"]; //another version: back_propagate_fully_connected_LSTM_cell
}

void back::LSTMCell::run(DeviceInstance& I)
{
	auto& kernel = prepare_for_running_kernel(this, I);
	auto cell_no = I.pointers[peers[2]];
	int batch_size = peers[0]->dimensions[0];
	int dim_hidden = inputs[0]->dimensions.back();
	kernel.setArg(0, I.buffers[peers[0]]); //z_grad
	kernel.setArg(1, I.buffers[peers[3]]); //h_prev
	kernel.setArg(2, I.buffers[peers[4]]); //cell_state_grad
	kernel.setArg(3, I.buffers[inputs[0]]); //h_grad
	kernel.setArg(4, I.buffers[peers[1]]); //gates_data

	cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * I.work_group_size);
	kernel.setArg(5, tmpMem);
	kernel.setArg(6, dim_hidden);
	kernel.setArg(7, batch_size);
	cell_no[0] += cell_no[1];
	kernel.setArg(8, static_cast<int>(cell_no[0]));

	cl::NDRange global(batch_size * dim_hidden);
	I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[peers[0]]);

	// fully connected fusion version TODO
//	auto& kernel = prepare_for_running_kernel(this, I);
//	auto cell_no = I.pointers[peers[2]];
//	int batch_size = dimensions[0];
//	int dim_hidden = dimensions[1];
//	int dim_input = peers[5]->dimensions[1];
//	kernel.setArg(0, I.buffers[peers[0]]); //weight_h_grad
//	kernel.setArg(1, I.buffers[peers[1]]); //weight_x_grad
//	kernel.setArg(2, I.buffers[peers[2]]); //bias_grad
//	kernel.setArg(3, I.buffers[inputs[0]]); //h_grad
//	kernel.setArg(4, I.buffers[peers[7]]); //x_grad
//	kernel.setArg(5, I.buffers[peers[4]]); //gates_data
//	kernel.setArg(6, I.buffers[peers[5]]); //x
//	kernel.setArg(7, I.buffers[peers[6]]); //weight_h
//	kernel.setArg(8, I.buffers[peers[8]]); //weight_x
//
//	int parallel = find_proper_local_size(dim_hidden < dim_input? dim_hidden : dim_input, I.work_group_size);
//	cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * parallel);
//	kernel.setArg(9, tmpMem);
//	kernel.setArg(10, dim_input);
//	kernel.setArg(11, dim_hidden);
//	kernel.setArg(12, batch_size);
//	kernel.setArg(13, static_cast<int>(cell_no[0]));
//
//	cl::NDRange global(dim_hidden * parallel);
//	cl::NDRange local(parallel);
//	I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, &I.precondition_events, &I.events[inputs[0]]);
//	I.events[peers[0]] = I.events[peers[1]] = I.events[peers[2]] = I.events[peers[7]] = I.events[inputs[0]];
//	cell_no[0] += cell_no[1];
}

string back::Loss::generate_source_code(DeviceInstance& I)
{
	return kernels_source[function + "_loss"];
}

void back::Loss::run(DeviceInstance& I)
{
	if (function == "negative_log_likelihood") {
		auto& kernel = prepare_for_running_kernel(this, I);
		int batch_size = peers[0]->dimensions[0];
		int dim_in = peers[0]->dimensions[1];
		kernel.setArg(0, I.buffers[peers[0]]); //out_grad
		kernel.setArg(1, I.buffers[peers[1]]); //out
		kernel.setArg(2, I.buffers[inputs[0]]); //in
		kernel.setArg(3, I.buffers[inputs[1]]); //label

		int parallel = find_proper_local_size(dim_in, I.work_group_size);
		cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * parallel);
		kernel.setArg(4, tmpMem);
		kernel.setArg(5, dim_in);
		kernel.setArg(6, batch_size);

		cl::NDRange global(batch_size * parallel);
		cl::NDRange local(parallel);
		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, &I.precondition_events, &I.events[peers[0]]);
		I.events[peers[1]] = I.events[peers[0]];
	}
	else if (function == "linear_regression") {
		auto& kernel = prepare_for_running_kernel(this, I);
		int parallel = peers[0]->volume;
		kernel.setArg(0, I.buffers[peers[0]]); //out_grad
		kernel.setArg(1, I.buffers[inputs[0]]); //y
		kernel.setArg(2, I.buffers[inputs[1]]); //label

		cl::NDRange global(parallel);
		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[peers[0]]);
	}
}

string type::BinaryOperator::generate_source_code(DeviceInstance& I)
{
	return kernels_source["parallel_" + function];
}

void type::BinaryOperator::run(DeviceInstance& I)
{
	auto& kernel = prepare_for_running_kernel(this, I);
	if (function == "plus") {
		int parallel = inputs[0]->volume;
		kernel.setArg(0, I.buffers[inputs[0]]); //a
		kernel.setArg(1, I.buffers[inputs[1]]); //b

		cl::NDRange global(parallel);
		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[inputs[0]]);
	}
	else if (function == "add") {
		kernel.setArg(0, I.buffers[peers[0]]); //z
		kernel.setArg(1, I.buffers[inputs[0]]); //a
		kernel.setArg(2, I.buffers[inputs[1]]); //b

		cl::NDRange global(volume);
		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[peers[0]]);
	}
	else if (function == "multiply") {
		int N = inputs[1]->dimensions[0];
		int HIDDEN = inputs[0]->dimensions[0];
		int dim_in = inputs[0]->dimensions[1];
		kernel.setArg(0, I.buffers[peers[0]]);
		kernel.setArg(1, I.buffers[inputs[0]]);
		kernel.setArg(2, I.buffers[inputs[1]]);

		int parallel = find_proper_local_size(dim_in, I.work_group_size);
		cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * parallel);
		kernel.setArg(3, tmpMem);
		kernel.setArg(4, HIDDEN);
		kernel.setArg(5, dim_in);
		if (dim_in < 16) {
			cl::NDRange global(N * HIDDEN);
			I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[peers[0]]);
		}
		else {
			cl::NDRange local(parallel);
			cl::NDRange global(N * HIDDEN * parallel);
			I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, &I.precondition_events, &I.events[peers[0]]);
		}
	}
}

std::string back::BinaryOperator::generate_source_code(DeviceInstance& I)
{
	return kernels_source["parallel_" + function];
}

void back::BinaryOperator::run(DeviceInstance& I)
{
	auto& kernel = prepare_for_running_kernel(this, I);
	if (function == "plus") {
		I.queue.enqueueCopyBuffer(I.buffers[inputs[0]], I.buffers[peers[1]], 0, 0, inputs[0]->size, &I.precondition_events, &I.events[peers[1]]);
	}
	else if (function == "add") {
		I.queue.enqueueCopyBuffer(I.buffers[inputs[0]], I.buffers[peers[0]], 0, 0, inputs[0]->size, &I.precondition_events, &I.events[peers[0]]);
		I.queue.enqueueCopyBuffer(I.buffers[inputs[0]], I.buffers[peers[1]], 0, 0, inputs[0]->size, &I.precondition_events, &I.events[peers[1]]);
	}
	else if (function == "multiply") {
		int dim_in = peers[0]->dimensions[1];
		int dim_hidden = peers[0]->dimensions[0];
		auto out_gradient = inputs[0];
		kernel.setArg(0, I.buffers[peers[0]]);
		kernel.setArg(1, I.buffers[out_gradient]);
		kernel.setArg(2, I.buffers[inputs[1]]);

		int parallel = find_proper_local_size(dim_hidden / 2, I.work_group_size); //dim_out/2: parallel needs < dim_out
		cl::LocalSpaceArg tmpMem = cl::Local(sizeof(float) * parallel);
		kernel.setArg(3, tmpMem);
		kernel.setArg(4, dim_hidden);
		kernel.setArg(5, dim_in);

		cl::NDRange local(parallel);
		cl::NDRange global(dim_hidden * dim_in);
		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, &I.precondition_events, &I.events[peers[0]]);

		kernel.setArg(0, I.buffers[peers[1]]);
		kernel.setArg(2, I.buffers[inputs[2]]);
		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, &I.precondition_events, &I.events[peers[1]]);
	}
}

float back::Loss::L(DeviceInstance& I) //for training progress evaluation scenario, NOT high speed critical scenario
{
	auto label = inputs[1];
	if (dynamic_cast<type::Data*>(label) == nullptr)
		label->upload(I);
	if (function == "negative_log_likelihood") {
		peers[1]->upload(I);
		float* predict = I.pointers[peers[1]];
		float* target = I.pointers[label];
		int N = peers[1]->dimensions[0];
		int num_classes = peers[1]->dimensions.back();
		float sum = 0;
		for (int i = 0; i < N; ++i) {
			int index = static_cast<int>(target[i]);
			float v = predict[i * num_classes + index];
			sum += -log(v + 1e-38f);
		}
		return sum;
	}
	else if (function == "linear_regression") {
		inputs[0]->upload(I);
		float* predict = I.pointers[inputs[0]];
		float* target = I.pointers[label];
		int N = inputs[1]->dimensions[0];
		float sum = 0;
		for (int i = 0; i < N; ++i) {
			float delta = predict[i] - target[i];
			sum += delta * delta;
		}
		return sum / 2 / N;
	}
	return 0;
}

std::string type::ConvolutionKernel::generate_source_code(DeviceInstance& I)
{
	cl::Platform platform(I.device.getInfo<CL_DEVICE_PLATFORM>());
	if (platform.getInfo<CL_PLATFORM_NAME>().find("NVIDIA") == string::npos)
		cl_build_options += " -DCONVOLUTION_VECTOR"; //screen float16 issue for NVIDIA driver

	string code = kernels_source["feed_forward_convolution_activation_relu"];
	if (activation != "relu")
		replace_all(code, "relu", activation);

	return code;
}

void type::ConvolutionKernel::run(DeviceInstance& I)
{
	auto& kernel = prepare_for_running_kernel(this, I);
	auto input = inputs[0], output = peers[0], weight = inputs[1];
	int in_height = input->dimensions[1], in_width = input->dimensions[2], in_depth = input->dimensions[3];
	int kernel_height = weight->dimensions[1], kernel_width = weight->dimensions[2];
	int batch_size = input->dimensions[0];
	kernel.setArg(0, I.buffers[output]);
	kernel.setArg(1, I.buffers[weight]);
	kernel.setArg(2, I.buffers[inputs[2]]);
	kernel.setArg(3, I.buffers[input]);
	kernel.setArg(4, in_height);
	kernel.setArg(5, in_width);
	kernel.setArg(6, in_depth);
	kernel.setArg(7, kernel_height);
	kernel.setArg(8, kernel_width);
	kernel.setArg(9, stride_size[0]); //stride_height
	kernel.setArg(10, stride_size[1]); //stride_width
	int padding_height = (output->dimensions[1] * stride_size[0] - in_height + weight->dimensions[1]) / 2;
	int padding_weight = (output->dimensions[2] * stride_size[1] - in_width + weight->dimensions[2]) / 2;
	kernel.setArg(11, padding_height);
	kernel.setArg(12, padding_weight);
	kernel.setArg(13, batch_size);

	cl::NDRange global(batch_size * output->dimensions[3], output->dimensions[1], output->dimensions[2]);
	I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[output]);
}

std::string back::ConvolutionKernel::generate_source_code(DeviceInstance& I)
{
	string code = kernels_source["back_propagate_convolution_relu_gradient"];
	auto& activation = static_cast<type::ConvolutionKernel*>(peers[0])->activation;
	if (activation != "relu")
		replace_all(code, "relu", activation);

	return code;
}

std::string back::ConvolutionKernel::BackForInput::generate_source_code(DeviceInstance& I)
{
	string code = kernels_source["back_propagate_convolution_relu_gradient_for_input"];
	auto& activation = static_cast<type::ConvolutionKernel*>(peers[0])->activation;
	if (activation != "relu")
		replace_all(code, "relu", activation);

	return code;
}

void back::ConvolutionKernel::run(DeviceInstance& I)
{
	auto in_gradient = peers[1], weight_gradient = peers[2], bias_gradient = peers[3], input = peers[4], output = peers[5], out_gradient = inputs[0];
	auto tensor = static_cast<type::ConvolutionKernel*>(peers[0]);
	int padding_height = (out_gradient->dimensions[1] * tensor->stride_size[0] - input->dimensions[1] + weight_gradient->dimensions[1]) / 2;
	int padding_weight = (out_gradient->dimensions[2] * tensor->stride_size[1] - input->dimensions[2] + weight_gradient->dimensions[2]) / 2;
	int in_height = input->dimensions[1], in_width = input->dimensions[2], in_depth = input->dimensions[3];
	int out_height = output->dimensions[1], out_width = output->dimensions[2], out_depth = output->dimensions[3];
	int weight_height = weight_gradient->dimensions[1], weight_width = weight_gradient->dimensions[2];
	int batch_size = input->dimensions[0];
	{
	auto& kernel = prepare_for_running_kernel(this, I);
	kernel.setArg(0, I.buffers[weight_gradient]);
	kernel.setArg(1, I.buffers[bias_gradient]);
	kernel.setArg(2, I.buffers[input]);
	kernel.setArg(3, I.buffers[output]);
	kernel.setArg(4, I.buffers[out_gradient]);
	kernel.setArg(5, in_height);
	kernel.setArg(6, in_width);
	kernel.setArg(7, in_depth);
	kernel.setArg(8, out_height);
	kernel.setArg(9, out_width);
	kernel.setArg(10, out_depth);
	kernel.setArg(11, tensor->stride_size[0]); //stride_height
	kernel.setArg(12, tensor->stride_size[1]); //stride_width
	kernel.setArg(13, padding_height);
	kernel.setArg(14, padding_weight);
	kernel.setArg(15, batch_size);

	cl::NDRange global(out_depth * in_depth, weight_height, weight_width);
	I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[weight_gradient]);
	if (bias_gradient != NULL)
		I.events[bias_gradient] = I.events[weight_gradient];
	}
	if (in_gradient == nullptr)
		return;
	{
	auto& kernel = prepare_for_running_kernel(peers[6], I);
	kernel.setArg(0, I.buffers[in_gradient]);
	kernel.setArg(1, I.buffers[peers[7]]); //weight
	kernel.setArg(2, I.buffers[output]);
	kernel.setArg(3, I.buffers[out_gradient]);
	kernel.setArg(4, weight_height);
	kernel.setArg(5, weight_width);
	kernel.setArg(6, in_depth);
	kernel.setArg(7, out_height);
	kernel.setArg(8, out_width);
	kernel.setArg(9, out_depth);
	kernel.setArg(10, tensor->stride_size[0]); //stride_height
	kernel.setArg(11, tensor->stride_size[1]); //stride_width
	kernel.setArg(12, padding_height);
	kernel.setArg(13, padding_weight);
	kernel.setArg(14, batch_size);

	cl::NDRange global(batch_size * in_depth, in_height, in_width);
	I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[in_gradient]);
	}
}

std::string type::Pooling::generate_source_code(DeviceInstance& I)
{
	string code = kernels_source["feed_forward_" + type + "_pooling"];
	return code;
}

void type::Pooling::run(DeviceInstance& I)
{
	auto& kernel = prepare_for_running_kernel(this, I);
	auto input = inputs[0], output = peers[0];
	int in_height = input->dimensions[1], in_width = input->dimensions[2], in_depth = input->dimensions[3];
	int batch_size = input->dimensions[0];
	kernel.setArg(0, I.buffers[output]);
	kernel.setArg(1, I.buffers[input]);
	kernel.setArg(2, in_height);
	kernel.setArg(3, in_width);
	kernel.setArg(4, in_depth);
	kernel.setArg(5, pooling_size[0]); //pool_height
	kernel.setArg(6, pooling_size[1]); //pool_width
	kernel.setArg(7, stride_size[0]); //stride_height
	kernel.setArg(8, stride_size[1]); //stride_width
	int padding_height = (output->dimensions[1] * stride_size[0] - in_height + pooling_size[0]) / 2;
	int padding_weight = (output->dimensions[2] * stride_size[1] - in_width + pooling_size[1]) / 2;
	kernel.setArg(9, padding_height);
	kernel.setArg(10, padding_weight);
	kernel.setArg(11, batch_size);
	if (type == "max")
		kernel.setArg(12, I.buffers[peers[1]]); //max_index

	cl::NDRange global(batch_size * output->dimensions[3], output->dimensions[1], output->dimensions[2]);
	I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[output]);
}

std::string back::Pooling::generate_source_code(DeviceInstance& I)
{
	string code = kernels_source["back_propagate_" + static_cast<type::Pooling*>(peers[0])->type + "_pooling"];
	return code;
}

void back::Pooling::run(DeviceInstance& I)
{
	auto& kernel = prepare_for_running_kernel(this, I);
	auto in_gradient = peers[1], out_gradient = inputs[0];
	auto tensor = static_cast<type::Pooling*>(peers[0]);
	int out_height = out_gradient->dimensions[1], out_width = out_gradient->dimensions[2], out_depth = out_gradient->dimensions[3];
	int batch_size = in_gradient->dimensions[0];
	kernel.setArg(0, I.buffers[in_gradient]);
	kernel.setArg(1, I.buffers[out_gradient]);
	kernel.setArg(2, out_height);
	kernel.setArg(3, out_width);
	kernel.setArg(4, out_depth);
	kernel.setArg(5, tensor->pooling_size[0]); //pool_height
	kernel.setArg(6, tensor->pooling_size[1]); //pool_width
	kernel.setArg(7, tensor->stride_size[0]); //stride_height
	kernel.setArg(8, tensor->stride_size[1]); //stride_width
	int padding_height = (out_gradient->dimensions[1] * tensor->stride_size[0] - in_gradient->dimensions[1] + tensor->pooling_size[0]) / 2;
	int padding_weight = (out_gradient->dimensions[2] * tensor->stride_size[1] - in_gradient->dimensions[2] + tensor->pooling_size[1]) / 2;
	kernel.setArg(9, padding_height); //padding_height
	kernel.setArg(10, padding_weight); //padding_width
	kernel.setArg(11, batch_size);
	if (tensor->type == "max")
		kernel.setArg(12, I.buffers[peers[2]]); //max_index

	cl::NDRange global(batch_size * in_gradient->dimensions[3], in_gradient->dimensions[1], in_gradient->dimensions[2]);
	I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[in_gradient]);
}

void type::Reshape::run(DeviceInstance& I)
{
	I.events[this] = I.events[inputs[0]];
}

void back::Reshape::run(DeviceInstance& I)
{
	I.events[peers[0]] = I.events[this];
}

}
