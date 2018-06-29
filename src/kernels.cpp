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

extern int debugger_device_id;
extern condition_variable breakpoint;
unordered_map<Tensor*, cl::Buffer> temp_global_buffer;
mutex breakpoint_mutex;
Tensor *_current, *_breakpoint = nullptr;
int64 microseconds = 0, breakpoint_hit_times = -1; //paused on all devices initially

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
		size_t pos = 0;
		while ((pos = source.find(KERNEL_DEF, pos)) != string::npos) {
			int begin = pos + KERNEL_DEF.size();
			int end = source.find("(", begin);
			name = source.substr(begin, end - begin);
			if (kernels.count(name) == 0) {
				kernels.insert(name);
				size_t pos2 = source.find(KERNEL_DEF, pos + KERNEL_DEF.size());
				if (pos2 != string::npos)
					pos2 = source.rfind("}\n", pos2) + 2;
				else
					pos2 = source.size();
				sources.append("\n").append(source.substr(pos, pos2 - pos));
			}
//			logger << "****** " << name << endl << code << endl;
			if (!tensor_kernels[tensor].empty())
				tensor_kernels[tensor].append(" ");
			tensor_kernels[tensor].append(name);
			pos++;
		}
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
			breakpoint_hit_times = -1;
		}
		if (this == _breakpoint) {
			auto I = static_cast<DeviceInstance*>(data);
			int device_id = I->ID;
			if (breakpoint_hit_times < 0 || (debugger_device_id == device_id && --breakpoint_hit_times == 0)) {
				logger << "[debugger] device " << device_id << " break on " << alias << ": " << type_name(this) << endl;
				unique_lock<mutex> breakpoint_lock(breakpoint_mutex);
				breakpoint.wait(breakpoint_lock); //No considering spurious wake-up
				logger << "[debugger] device " << device_id << " continue to run." << endl;
			}
		}
		if (microseconds > 0)
			microseconds = MICROS();
		functor(this, data);
		if (microseconds > 0) {
			wait_for_all_kernels_finished(*static_cast<DeviceInstance*>(data));
			auto duration = MICROS(microseconds);
			kernels_cost[this] += duration;
//			logger << type_name(this) << " " << alias << ": " << duration << "��s" << endl;
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

int find_proper_local_size(int required, int work_group_size)
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

cl::Kernel& prepare_for_running_kernel(Tensor* tensor, DeviceInstance& I, int number)
{
	I.precondition_events.clear();
	for (auto input : tensor->inputs)
		if (input != nullptr && input->volume > 0) //exclude "pure kernel" tensor
			I.precondition_events.push_back(I.events[input]);
	return I.kernels[tensor][number];
}

#define FULLY_CONNECTED_STD_DIM_IN_UPLIMIT 512
string type::FullyConnectedLayer::generate_source_code(DeviceInstance& I)
{
	int dim_in = inputs[1]->dimensions.front(); //inputs[1]: weight
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
		code = kernels_source["feed_forward_fully_connected_sigmoid"];
		if (activation != "sigmoid")
			replace_all(code, "sigmoid", activation);
		//Parallel: (batch_size * dim_hidden * get_local_size(0))
		//Note: choose local NDRange size near (2 * dim_in) when enqueue ASAP
//		code = kernels_source["feed_forward_fully_connected_softrelu"];//TODO
//		if (activation != "softrelu")
//			replace_all(code, "softrelu", activation);
	}

	return code;
}

void type::FullyConnectedLayer::run(DeviceInstance& I)
{
	auto& kernel = prepare_for_running_kernel(this, I);
	int N = inputs[0]->volume / inputs[0]->dimensions.back();
	int HIDDEN = inputs[1]->dimensions.back();
	int dim_in = inputs[1]->dimensions.front();
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
//	if (dim_in < FULLY_CONNECTED_STD_DIM_IN_UPLIMIT) {//TODO
		cl::NDRange global(N * HIDDEN);
		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[peers[0]]);
//	}
//	else {
//		cl::NDRange local(parallel);
//		cl::NDRange global(N * HIDDEN * parallel);
//		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local, &I.precondition_events, &I.events[peers[0]]);
//	}
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

#define FULLY_CONNECTED_STD_DIM_IN_UPLIMIT_BP 4096
string back::FullyConnectedLayer::generate_source_code(DeviceInstance& I)
{
	auto weight = peers[2]; //peers[2]: weight
	int dim_in = weight->dimensions.front();
	string code;
	if (dim_in < FULLY_CONNECTED_STD_DIM_IN_UPLIMIT_BP)
		code = kernels_source["back_propagate_fully_connected_softrelu_gradient"];
	else
		code = kernels_source["back_propagate_fully_connected_softrelu_gradient_for_bias"] + "\n" + kernels_source["back_propagate_fully_connected_softrelu_gradient_for_weight"] + "\n" + kernels_source["back_propagate_fully_connected_softrelu_gradient_for_data"];
	if (activation != "softrelu")
		replace_all(code, "softrelu", activation);

	return code;
}

void back::FullyConnectedLayer::run(DeviceInstance& I)
{
	auto& kernel = prepare_for_running_kernel(this, I);
	auto weight = peers[2]; //peers[2]: weight
	int N = peers[3]->volume / peers[3]->dimensions.back(); //peers[3]: in
	int dim_in = weight->dimensions.front();
	int dim_out = weight->dimensions.back();
	auto in_gradient = peers[5];
	auto weight_gradient = peers[0];
	auto bias_gradient = peers[1];

	if (dim_in < FULLY_CONNECTED_STD_DIM_IN_UPLIMIT_BP) {
		if (in_gradient != nullptr)
			kernel.setArg(0, I.buffers[in_gradient]);
		else
			kernel.setArg(0, nullptr);
		kernel.setArg(1, I.buffers[peers[0]]);
		if (bias_gradient != nullptr)
			kernel.setArg(2, I.buffers[bias_gradient]);
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

		cl::Event& event = I.events[weight_gradient];
		cl::NDRange global(global_size);
		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &event);
		if (in_gradient != nullptr)
			I.events[in_gradient] = event;
		if (bias_gradient != nullptr)
			I.events[bias_gradient] = event;
	}
	else {
		if (temp_global_buffer.count(this) == 0) {
			auto context = I.queue.getInfo<CL_QUEUE_CONTEXT>();
			temp_global_buffer[this] = cl::Buffer(context, CL_MEM_READ_WRITE, N * dim_out * sizeof(float));
		}
		kernel.setArg(0, temp_global_buffer[this]);
		if (bias_gradient != nullptr)
			kernel.setArg(1, I.buffers[bias_gradient]);
		else
			kernel.setArg(1, nullptr);
		kernel.setArg(2, I.buffers[peers[4]]); //out
		kernel.setArg(3, I.buffers[inputs[0]]); //out_gradient
		kernel.setArg(4, dim_out);
		kernel.setArg(5, dim_in);
		kernel.setArg(6, N);

		cl::Event& event = I.events[this];
		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(dim_out), cl::NullRange, &I.precondition_events, &event);
		if (bias_gradient != nullptr)
			I.events[bias_gradient] = event;

		auto& kernel1 = I.kernels[this][1];
		kernel1.setArg(0, I.buffers[weight_gradient]);
		kernel1.setArg(1, temp_global_buffer[this]);
		kernel1.setArg(2, I.buffers[peers[3]]); //in
		kernel1.setArg(3, dim_out);
		kernel1.setArg(4, dim_in);
		kernel1.setArg(5, N);
		I.precondition_events.clear();
		I.precondition_events.push_back(event);
		I.queue.enqueueNDRangeKernel(kernel1, cl::NullRange, cl::NDRange(dim_out, dim_in), cl::NullRange, &I.precondition_events, &I.events[weight_gradient]);

		if (in_gradient != nullptr) {
			auto& kernel2 = prepare_for_running_kernel(this, I, 2);
			if (in_gradient != nullptr)
				kernel2.setArg(0, I.buffers[in_gradient]);
			else
				kernel2.setArg(0, nullptr);
			kernel2.setArg(1, I.buffers[weight]);
			kernel2.setArg(2, I.buffers[peers[4]]); //out
			kernel2.setArg(3, I.buffers[inputs[0]]); //out_gradient
			kernel2.setArg(4, dim_out);
			kernel2.setArg(5, dim_in);
			kernel2.setArg(6, N);
			kernel2.setArg(7, attached? 1 : 0); //merge_gradient mode
			I.queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(N * dim_in), cl::NullRange, &I.precondition_events, &I.events[in_gradient]);
		}
	}
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
	for (; epoch < max_epochs; epoch++) {
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
		for (auto param : peers) { //report gradients
			auto grad = param->gradient;
			I.precondition_events.clear();
			I.precondition_events.push_back(I.events[grad]);
			I.queue.enqueueReadBuffer(I.buffers[grad], CL_FALSE, 0, grad->size, I.pointers[grad], &I.precondition_events, &I.events[grad]);
			I.events[grad].setCallback(CL_COMPLETE, gradients_event_callback, &I);
		}

	if (I.parameters_state == static_cast<int>(peers.size())) {
		for (auto param : peers) { //load parameters
			I.precondition_events.clear();
			I.precondition_events.push_back(I.events[param->gradient]);
			I.queue.enqueueWriteBuffer(I.buffers[param], CL_FALSE, 0, param->size, I.pointers[param], &I.precondition_events, &I.events[param]);
			I.queue.enqueueFillBuffer<float>(I.buffers[param->gradient], 0, 0, param->size, &I.precondition_events, &I.events[param->gradient]);
			I.events[param].setCallback(CL_COMPLETE, parameters_event_callback, &I);
		}
	}
	else {
		auto& kernel = I.kernels[this].front();

		for (int i = 0, N = peers.size(); i < N ; i++) {
			auto parameter = peers[i];
			I.precondition_events.clear();
			I.precondition_events.push_back(I.events[parameter->gradient]);
			kernel.setArg(0, I.buffers[parameter]);
			kernel.setArg(1, I.buffers[parameter->gradient]);
			kernel.setArg(2, learning_rate);
			kernel.setArg(3, weight_decay);
			cl::NDRange global(parameter->volume);
			I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[parameter]);
		}
	}
}

void type::StochasticGradientDescentUpdater::run_globally(DeviceInstance& I, DeviceInstance& source)
{
	auto& kernel = I.kernels[this].front();
	cl::Event event;
	vector<cl::Event> preconditions, updates;
	for (int i = 0, N = peers.size(); i < N; i++) {
		auto parameter = peers[i];
		auto gradient = parameter->gradient;
		auto& gradient_buffer = I.buffers[gradient];
		auto& parameter_buffer = I.buffers[parameter];
		I.queue.enqueueWriteBuffer(gradient_buffer, CL_FALSE, 0, gradient->size, source.pointers[gradient], NULL, &event);
		preconditions.push_back(event);

		kernel.setArg(0, parameter_buffer);
		kernel.setArg(1, gradient_buffer);
		kernel.setArg(2, learning_rate);
		kernel.setArg(3, weight_decay);
		cl::NDRange global(parameter->volume);
		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &preconditions, &event);
		updates.push_back(event);
		preconditions.clear();
	}
	for (auto& ev : updates)
		ev.wait();
	source.gradients_state = static_cast<int>(peers.size());

	updates.clear();
	for (int i = 0, N = static_cast<int>(peers.size()); i < N; i++) {
		auto parameter = peers[i];
		I.queue.enqueueReadBuffer(I.buffers[parameter], CL_FALSE, 0, parameter->size, parameter->pointer, NULL, &event); //put into tensor own data pointer
		updates.push_back(event);
	}
	for (auto& ev : updates)
		ev.wait();
}

void type::Data::run(DeviceInstance& I)
{
	download(I, &no_preconditions);
}

void type::XavierNormalDistributionInitializer::run_globally(DeviceInstance& I)
{
	default_random_engine generator;
	for (auto tensor : peers) {
		int fan_in = tensor->dimensions.front(); //TODO
		normal_distribution<float> distribution(mu, sqrt(sigma / fan_in));
		if (dynamic_cast<Weight*>(tensor) != nullptr) { //TODO
			for (int64 hidden = 0; hidden < tensor->dimensions.back(); hidden++)
				for (int64 k = 0; k < tensor->dimensions.front(); k++)
					tensor->pointer[k * tensor->dimensions.back() + hidden] = (float) distribution(generator);
		}
		else if (dynamic_cast<Bias*>(tensor) == nullptr)
			for (float *p = tensor->pointer, *end = p + tensor->volume; p < end; p++)
				*p = (float) distribution(generator);
	}
}

void type::XavierNormalDistributionInitializer::run(DeviceInstance& I)
{
	initialization.lock();
	if (!initialized) {
		run_globally(I);
		initialized = true;
	}
	initialization.unlock();
	for (auto tensor : peers) {
		memcpy(I.pointers[tensor], tensor->pointer, tensor->size);
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
