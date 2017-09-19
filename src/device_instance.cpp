/*
 * device_instance.cpp
 *
 *  Created on: 2017/5/13
 *      Author: ZhangHua
 */

#include <condition_variable>
#include <chrono>
#include <queue>
#include <iostream>
#include <thread>
#include <omp.h>
#include <algorithm>

#include <tensor.hpp>
#include <device_instance.hpp>

using namespace std;
using namespace clnet::type;

namespace clnet {
extern Tensor* _current;

OpenCL_ OpenCL;
unordered_map<int, DeviceInstance> DeviceInstance::ALL;
unordered_map<string, string> key_values;

#if CL_HPP_TARGET_OPENCL_VERSION < 200
string cl_build_options;
#else
string cl_build_options = "-cl-std=CL2.0";
#endif

thread *global_updater;
condition_variable notification;
queue<int> parameters_queue, gradients_queue;
size_t parameters_timestamp;
mutex notification_mutex, queue_mutex, allocate_mutex;
unique_lock<mutex> notification_lock(notification_mutex);

void reload_kernels(const cl::Device& device, const cl::Context& context, DeviceInstance& I)
{
	unordered_map<Tensor*, string> tensor_kernels;
	string source = generate_kernel_sources(I, device, tensor_kernels);

	cl::Program program(context, source);
	try {
		if (CLNET_TENSOR_GLOBALS & CLNET_OPENCL_SHOW_SOURCE)
			cout << "****** source code ******\n" << source << endl;
		program.build(cl_build_options.c_str());
	}
	catch (cl::Error& e) {
		cout << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
		throw e;
	}

	for (auto iter : tensor_kernels) {
		auto tensor = iter.first;
		const char* name = iter.second.c_str();
		I.kernels[tensor] = cl::Kernel(program, name);
	}
}

void DeviceInstance::initialize()
{
	int size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
	if ((size & (size - 1)) == 0)
		work_group_size = size;
	else { //not power of 2
		int power2 = 1;
		while (power2 < size)
			power2 <<= 1;
		work_group_size = power2 / 2;
	}

	cl::Context context(device);
	reload_kernels(device, context, *this);

#if CL_HPP_TARGET_OPENCL_VERSION < 200
	queue = cl::CommandQueue(context, device);
#else
	queue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
#endif

	for (auto tensor : Tensor::ALL)
		tensor->initialize(this);
}

DeviceInstance& DeviceInstance::create(cl::Device& cl_device, int id)
{
	DeviceInstance& I = DeviceInstance::ALL[id];
	I.ID = id;
	I.device = cl_device;
	I.initialize();
	return I;
}

void Tensor::initialize(DeviceInstance* I) //this should be idempotent
{
	if (dimensions.empty()) {
		size = 0;
		pointer = nullptr;
	}
	else {
		allocate_mutex.lock();
		if (pointer == nullptr) { //idempotent
			pointer = new float[volume];
			size = volume * sizeof(float);
			memset(pointer, 0, size); //Every byte of initialized Tensor is starting from zero
		}
		allocate_mutex.unlock();

		if (I != nullptr && I->pointers.count(this) == 0) { //idempotent
			const auto& context = I->queue.getInfo<CL_QUEUE_CONTEXT>();
			I->buffers[this] = cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, size, pointer); //initialize from tensor itself
			I->pointers[this] = new float[volume];
			memcpy(I->pointers[this], pointer, size); //initialized by tensor's pointer
		}
	}
}

type::MiniBatch::MiniBatch(int size, int total) : batch_size(size)
{
	set_total_samples(total);
}

void type::MiniBatch::set_total_samples(int N)
{
	shape_with({ N + 1 });
	total_batches = N / batch_size;
	if (N % batch_size != 0 && (CLNET_TENSOR_GLOBALS & CLNET_VALUE_MISMATCH_WARN))
		cout << (N % batch_size) << " samples at tail were abandoned." << endl;
}

void type::MiniBatch::initialize(DeviceInstance* I)
{
	Tensor::initialize(I);
	if (I == nullptr)
		return;
	int* p = reinterpret_cast<int*>(I->pointers[this]);
	*p++ = -1;
	for (int i = 0, N = dimensions[0] - 1; i < N; i++)
		*p++ = i;
}

bool type::MiniBatch::has_next(DeviceInstance& I)
{
	int& current = *reinterpret_cast<int*>(I.pointers[this]);
	return ++current < total_batches;
}

void type::MiniBatch::reset(DeviceInstance& I)
{
	int* p = reinterpret_cast<int*>(I.pointers[this]);
	*p++ = -1;
	random_shuffle(p, p + total_batches * batch_size);
}

void type::Reshape::initialize(DeviceInstance* I)
{
	auto tensor = inputs[0]; //inputs[0]: input. Use input as the real storage
	tensor->initialize(I);
	allocate_mutex.lock();
	pointer = tensor->pointer;
	allocate_mutex.unlock();
	if (I == nullptr)
		return;
	I->pointers[this] = I->pointers[tensor];
	I->buffers[this] = I->buffers[tensor];
}

type::Reshape::~Reshape()
{
	pointer = nullptr;
}

void back::Reshape::initialize(DeviceInstance* I)
{
	auto tensor = peers[0]; //peers[0]: in_gradient. Use in_gradient as the real storage
	if (tensor == nullptr)
		return;
	tensor->initialize(I);
	allocate_mutex.lock();
	pointer = tensor->pointer;
	allocate_mutex.unlock();
	I->pointers[this] = I->pointers[tensor];
	I->buffers[this] = I->buffers[tensor];
}

back::Reshape::~Reshape()
{
	pointer = nullptr;
}

void DeviceInstance::free()
{
	set<float*> memory;
	for (auto iter : pointers)
		if (memory.count(iter.second) == 0) { //avoid deleting again
			delete iter.second;
			memory.insert(iter.second);
		}
}

void CL_CALLBACK gradients_event_callback(cl_event, cl_int, void * user_data)
{
	DeviceInstance* instance = reinterpret_cast<DeviceInstance*>(user_data);
	if (instance->gradients_state-- == 1) {
		queue_mutex.lock();
		gradients_queue.push(instance->ID);
		queue_mutex.unlock();
		notification.notify_all();
	}
}

void CL_CALLBACK parameters_event_callback(cl_event, cl_int, void * user_data)
{
	DeviceInstance* instance = reinterpret_cast<DeviceInstance*>(user_data);
	if (instance->parameters_state-- == 1) {
		queue_mutex.lock();
		parameters_queue.push(instance->ID);
		queue_mutex.unlock();
		notification.notify_all();
	}
}

void Updater::launch_global_updater_thread(DeviceInstance& I)
{
	global_updater = new thread(&Updater::global_updater_thread, this, ref(I));
}

void Updater::stop_global_updater_thread()
{
	thread* updater = global_updater;
	global_updater = nullptr;
	notification.notify_all();
	if (updater != nullptr) {
		updater->join();
		delete updater;
	}
}

void Updater::synchronize_device_parameters(DeviceInstance& I)
{
	if (I.parameters_state > 0)
		return;
	int i, N = peers.size();
#pragma omp parallel for
	for (i = 0; i < N; i++) {
		auto tensor = peers[i]->gradient;
		memcpy(I.pointers[tensor], I.pointers[tensor], tensor->size);
	}
	I.parameters_timestamp = parameters_timestamp;
	I.parameters_state = peers.size();
}

void Updater::global_updater_thread(DeviceInstance& I)
{
//	vector<cl::Buffer> parameters_buffer, gradients_buffer;
	Tensor& global_data = *new Tensor({}, {}, alias + "_global_data");
	const auto& context = I.queue.getInfo<CL_QUEUE_CONTEXT>();
	for (auto tensor : peers) {
		auto parameter = new Tensor({}, {}, alias + ":" + tensor->alias);
		global_data.peers.push_back(parameter);
		I.buffers[parameter] = cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, tensor->size, I.pointers[tensor->gradient]);
		auto gradient = new Tensor({}, {}, "gradient(" + alias + ":" + tensor->alias + ")");
		global_data.inputs.push_back(gradient);
		I.buffers[gradient] = cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, tensor->size);
//		parameters_buffer.push_back(cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR, tensor->size, tensor->pointer));
//		gradients_buffer.push_back(cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_ALLOC_HOST_PTR, tensor->size));
	}

//	cl::Kernel& kernel = I.kernels[this];
	cl::Event event;
	vector<cl::Event> updates;
	while (global_updater != nullptr) {
		notification.wait(notification_lock, [] { return !gradients_queue.empty() || global_updater == nullptr; }); //TODO: global_updater == nullptr is needed?

		bool have_data = !gradients_queue.empty();
		while (!gradients_queue.empty()) {
			queue_mutex.lock();
			int ID = gradients_queue.front();
			gradients_queue.pop();
			queue_mutex.unlock();

			DeviceInstance& instance = DeviceInstance::ALL[ID];
			run_globally(I, instance, global_data);
//			vector<cl::Event> preconditions;
//			for (int i = 0, N = inputs.size(); i < N; i++) {
//				auto tensor = inputs[i];
//				auto gradient = peers[i];
//				I.queue.enqueueWriteBuffer(gradients_buffer[i], CL_FALSE, 0, gradient->size, instance.pointers[gradient], NULL, &event);
//				preconditions.push_back(event);
//
//				kernel.setArg(0, parameters_buffer[i]);
//				kernel.setArg(1, gradients_buffer[i]);
//				kernel.setArg(2, learning_rate);
//				kernel.setArg(3, weight_decay);
//				cl::NDRange global(tensor->volume);
//				I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &preconditions, &event);
//				updates.push_back(event);
//			}
//			for (auto& ev : updates)
//				ev.wait();
			instance.gradients_state = inputs.size();
		}
		if (have_data) {
			updates.clear();
			for (int i = 0, N = inputs.size(); i < N; i++) {
				auto tensor = inputs[i];
				I.queue.enqueueReadBuffer(I.buffers[global_data.peers[i]], CL_FALSE, 0, tensor->size, I.pointers[tensor], NULL, &event);
				updates.push_back(event);
			}
			for (auto& ev : updates)
				ev.wait();
			parameters_timestamp = chrono::duration_cast<chrono::milliseconds>(chrono::system_clock::now().time_since_epoch()).count();
			for (auto& iter : DeviceInstance::ALL)
				synchronize_device_parameters(iter.second);
			updates.clear();
		}

		while (!parameters_queue.empty()) {
			queue_mutex.lock();
			int ID = parameters_queue.front();
			parameters_queue.pop();
			queue_mutex.unlock();

			DeviceInstance& instance = DeviceInstance::ALL[ID];
			if (parameters_timestamp <= instance.parameters_timestamp)
				continue;
			synchronize_device_parameters(instance);
		}
	}
}

void wait_for_all_kernels_finished(DeviceInstance& I)
{
	for (auto iter : I.events)
		iter.second.wait();
}

vector<cl::Device>& OpenCL_::find_devices()
{
	if (devices != nullptr)
		return *devices;

	devices = new vector<cl::Device>;
	vector<cl::Platform> platforms;
	try {
		cl::Platform::get(&platforms);
	}
	catch (cl::Error&) {
		return *devices;
	}

	for (auto& platform : platforms) {
		vector<cl::Device> platform_devices;
		try {
			platform.getDevices(device_type, &platform_devices);
		}
		catch (cl::Error&) {
			continue;
		}
		for (auto& device : platform_devices)
			devices->push_back(device);
	}
	return *devices;
}

void OpenCL_::print_device_info(ostream& out)
{
	if (devices == nullptr)
		find_devices();

	cl_platform_id current = nullptr;
	for (size_t n = 0; n < devices->size(); n++) {
		auto& device = devices->at(n);
		const auto& platform_id = device.getInfo<CL_DEVICE_PLATFORM>();
		if (platform_id != current) {
			current = platform_id;
			cl::Platform platform0(current);
			out << "OpenCL Platforms: " << platform0.getInfo<CL_PLATFORM_NAME>() << endl;
			out << "Version:          " << platform0.getInfo<CL_PLATFORM_VERSION>() << endl;
			out << "Vendor:           " << platform0.getInfo<CL_PLATFORM_VENDOR>() << endl;
			out << "Profile:          " << platform0.getInfo<CL_PLATFORM_PROFILE>() << endl;
			out << "Platform Devices: " << endl;
		}
		string name = device.getInfo<CL_DEVICE_NAME>();
		auto deviceType = device.getInfo<CL_DEVICE_TYPE>();
		auto sizesItem = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

		out << "\tDevice Name:         " << name << endl;
		out << "\tclNET device ID:     " << n << endl;
		out << "\tType:                ";
		switch (deviceType) {
		case CL_DEVICE_TYPE_CPU: out << "CPU"; break;
		case CL_DEVICE_TYPE_GPU: out << "GPU"; break;
		default: out << "OTHER"; break;
		}
		out << endl;
		out << "\tVersion:             " << device.getInfo<CL_DEVICE_VERSION>() << '/' << device.getInfo<CL_DRIVER_VERSION>() << endl;
		out << "\tGlobal/Local Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << '/' << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << " bytes" << endl;
		out << "\tMax ComputeUnits:    " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
		out << "\tMax WorkItem Sizes:  [" << sizesItem[0];
		for (size_t i = 1; i < sizesItem.size(); i++)
			out << ',' << sizesItem[i];
		out << ']' << endl;
		out << "\tBuiltIn Kernels:     ";
		try {
			out << device.getInfo<CL_DEVICE_BUILT_IN_KERNELS>();
		} catch (cl::Error& e) {
			out << "Error in " << e.what() << " (" << e.err() << "): " << clErrorCodeDescriptions[-e.err()] << endl;
		}
		out << endl;
		out << "\tExtensions:          " << device.getInfo<CL_DEVICE_EXTENSIONS>() << endl;
		out << endl;
	}
}

void allocate_tensor(Tensor* current, void* data)
{
	current->initialize(static_cast<DeviceInstance*>(data));
	if (dynamic_cast<type::Structured*>(current) != nullptr) {
		set<Tensor*> visited;
		for (auto tensor : current->peers)
			tensor->launch(&visited, data, allocate_tensor);
	}
}

string millis_string(size_t time)
{
	string millis;
	if (time >= 1000)
		millis.append(to_string(time / 1000)).append("s.");
	millis.append(to_string(time % 1000)).append("ms");
	return millis;
}

void OpenCL_::run(Tensor& graph, vector<int> targetDeviceIDs, bool use_debugger)
{
	vector<cl::Device>& targetDevices = find_devices();
	if (targetDevices.empty()) {
		cout << "No OpenCL device found." << endl;
		return;
	}

	size_t no, deviceNum = targetDeviceIDs.size();
	auto updater = graph.peers.empty() || deviceNum == 1? nullptr : dynamic_cast<type::Updater*>(graph.peers[0]);
	thread_barrier barrier(deviceNum);
#pragma omp parallel for
	for (no = 0; no < deviceNum; no++) {
		try {
			int device_id = targetDeviceIDs[no];
			auto& device = targetDevices[device_id];
			const auto& name = device.getInfo<CL_DEVICE_NAME>();

			size_t time = MILLIS(0);
			auto& I = DeviceInstance::create(device, device_id);
			time = MILLIS(time);
			cout << "[" << I.ID << "] " << name << " (kernels build: " << millis_string(time) << ")" << endl;
			if (use_debugger && no == 0)
				launch_debugger_thread(I, graph);

			if (updater != nullptr) {
				I.gradients_state = updater->inputs.size();
				if (no == 0)
					updater->launch_global_updater_thread(I);
				barrier.wait();
			}

			set<Tensor*> visited;
			time = MILLIS(0);
			graph.launch(&visited, &I);

			time = MILLIS(time);
			cout << "[" << I.ID << "] run time: " << millis_string(time) << "." << endl;
			if (updater != nullptr) {
				barrier.wait();
				if (no == 0)
					updater->stop_global_updater_thread();
				barrier.wait();
			}
			I.free();
		}
		catch (cl::Error& e) {
			if (_current != nullptr)
				cout << "Current Tensor: " << type_name(_current) << ": " << _current->alias << endl;
			cout << "Error in " << e.what() << " (" << e.err() << "): " << clErrorCodeDescriptions[e.err() < MIN_ERROR_CODE? -USER_ERROR_DESCRIPTION_UNDEFINED : -e.err()] << endl;
		}
		catch (runtime_error& e) {
			cout << "Runtime error: " << e.what() << endl;
		}
	}
}

void display_tensor_name(Tensor* current, void* padding)
{
	string& pad = *static_cast<string*>(padding);
	cout << pad << type_name(current) << "\t\t" << current->alias;
	if (!current->dimensions.empty()) {
		cout << "[" << current->dimensions[0];
		for (size_t i = 1; i < current->dimensions.size(); i++)
			cout << "," << current->dimensions[i];
		cout << "]";
	}
	cout << endl;

	auto structure = dynamic_cast<type::Structured*>(current);
	if (structure == nullptr)
		return;
	set<Tensor*> visited;
	string indent = pad + "\t";
	auto body = structure->body();
	if (body != nullptr)
		body->launch(&visited, static_cast<void*>(&indent), display_tensor_name);
	auto others = structure->auxiliaries();
	if (!others.empty())
		cout << "-";
	for (auto aux : others)
		display_tensor_name(aux, static_cast<void*>(&indent));
}

void OpenCL_::print_tensor_structure(Tensor& graph)
{
	string padding;
	set<Tensor*> visited;
	graph.launch(&visited, static_cast<void*>(&padding), display_tensor_name);
}

void OpenCL_::deallocate_all_tensors()
{
	for (auto tensor : Tensor::ALL)
		delete tensor;
	Tensor::ALL.clear();
}

const char* clErrorCodeDescriptions[] = {
		"CL_SUCCESS", //0
		"CL_DEVICE_NOT_FOUND",
		"CL_DEVICE_NOT_AVAILABLE",
		"CL_COMPILER_NOT_AVAILABLE",
		"CL_MEM_OBJECT_ALLOCATION_FAILURE",
		"CL_OUT_OF_RESOURCES",
		"CL_OUT_OF_HOST_MEMORY",
		"CL_PROFILING_INFO_NOT_AVAILABLE",
		"CL_MEM_COPY_OVERLAP",
		"CL_IMAGE_FORMAT_MISMATCH",
		"CL_IMAGE_FORMAT_NOT_SUPPORTED", //-10
		"CL_BUILD_PROGRAM_FAILURE",
		"CL_MAP_FAILURE",
		"CL_MISALIGNED_SUB_BUFFER_OFFSET",
		"CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
		"CL_COMPILE_PROGRAM_FAILURE",
		"CL_LINKER_NOT_AVAILABLE",
		"CL_LINK_PROGRAM_FAILURE",
		"CL_DEVICE_PARTITION_FAILED",
		"CL_KERNEL_ARG_INFO_NOT_AVAILABLE",
		"", //-20
		"",
		"",
		"",
		"",
		"",
		"",
		"",
		"USER_ERROR_DESCRIPTION_UNDEFINED",
		"USER_GROUP_SIZE_NOT_BIG_ENOUGH",
		"CL_INVALID_VALUE", //-30
		"CL_INVALID_DEVICE_TYPE",
		"CL_INVALID_PLATFORM",
		"CL_INVALID_DEVICE",
		"CL_INVALID_CONTEXT",
		"CL_INVALID_QUEUE_PROPERTIES",
		"CL_INVALID_COMMAND_QUEUE",
		"CL_INVALID_HOST_PTR",
		"CL_INVALID_MEM_OBJECT",
		"CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
		"CL_INVALID_IMAGE_SIZE", //-40
		"CL_INVALID_SAMPLER",
		"CL_INVALID_BINARY",
		"CL_INVALID_BUILD_OPTIONS",
		"CL_INVALID_PROGRAM",
		"CL_INVALID_PROGRAM_EXECUTABLE",
		"CL_INVALID_KERNEL_NAME",
		"CL_INVALID_KERNEL_DEFINITION",
		"CL_INVALID_KERNEL",
		"CL_INVALID_ARG_INDEX",
		"CL_INVALID_ARG_VALUE", //-50
		"CL_INVALID_ARG_SIZE",
		"CL_INVALID_KERNEL_ARGS",
		"CL_INVALID_WORK_DIMENSION",
		"CL_INVALID_WORK_GROUP_SIZE",
		"CL_INVALID_WORK_ITEM_SIZE",
		"CL_INVALID_GLOBAL_OFFSET",
		"CL_INVALID_EVENT_WAIT_LIST",
		"CL_INVALID_EVENT",
		"CL_INVALID_OPERATION",
		"CL_INVALID_GL_OBJECT", //-60
		"CL_INVALID_BUFFER_SIZE",
		"CL_INVALID_MIP_LEVEL",
		"CL_INVALID_GLOBAL_WORK_SIZE",
		"CL_INVALID_PROPERTY",
		"CL_INVALID_IMAGE_DESCRIPTOR",
		"CL_INVALID_COMPILER_OPTIONS",
		"CL_INVALID_LINKER_OPTIONS",
		"CL_INVALID_DEVICE_PARTITION_COUNT",
		"CL_INVALID_PIPE_SIZE",
		"CL_INVALID_DEVICE_QUEUE" //-70
};

template <> string optional(unordered_map<string, string>& map, string name, string default_value)
{
	auto iter = map.find(name);
	return iter != map.end()? iter->second : default_value;
}

template <> int optional(unordered_map<string, string>& map, string name, int default_value)
{
	auto iter = map.find(name);
	return iter != map.end()? atoi(iter->second.c_str()) : default_value;
}

template <> size_t optional(unordered_map<string, string>& map, string name, size_t default_value)
{
	auto iter = map.find(name);
	return iter != map.end()? strtoull(iter->second.c_str(), nullptr, 10) : default_value;
}

template <> double optional(unordered_map<string, string>& map, string name, double default_value)
{
	auto iter = map.find(name);
	return iter != map.end()? atof(iter->second.c_str()) : default_value;
}

template <typename T> T optional(std::string name, T default_value)
{
	return optional<T>(key_values, name, default_value);
}

template string optional(string name, string default_value);
template int optional(string name, int default_value);
template size_t optional(string name, size_t default_value);
template double optional(string name, double default_value);

}
