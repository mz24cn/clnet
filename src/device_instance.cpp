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
#include <fstream>

#include <tensor.hpp>
#include <device_instance.hpp>

using namespace std;
using namespace clnet::type;

#define MSG_CODE(ID,CODE) (((int64) ID) << 32) + CODE
#define MSG_GRADIENTS_READY 1
#define MSG_PARAMETERS_READY 2
#define MSG_QUIT 0

namespace clnet {
extern Tensor* _current;

Logger logger;
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
queue<int64> message_queue;
mutex notification_mutex, queue_mutex, allocate_mutex;
unique_lock<mutex> notification_lock(notification_mutex);

void reload_kernels(const cl::Device& device, const cl::Context& context, DeviceInstance& I)
{
	unordered_map<Tensor*, string> tensor_kernels;
	string source = generate_kernel_sources(I, device, tensor_kernels);

	cl::Program program(context, source);
	cl_build_options.append(" -DMAX_WORK_GROUP_SIZE=").append(to_string(I.work_group_size));
	try {
		if (CLNET_TENSOR_GLOBALS & CLNET_OPENCL_SHOW_SOURCE)
			logger << "cl_build_options: " << cl_build_options << "\nsource code:\n" << source << endl;
		program.build(cl_build_options.c_str());
	}
	catch (cl::Error& e) {
		logger << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << endl;
		throw e;
	}

	for (auto& iter : tensor_kernels) {
		auto tensor = iter.first;
		stringstream ss;
		ss << iter.second;
		string kernel;
		while (ss >> kernel) {
			const char* name = kernel.c_str();
			I.kernels[tensor].push_back(cl::Kernel(program, name));
		}
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
		if (ID >= 0 || dynamic_cast<type::Weight*>(tensor) != nullptr || dynamic_cast<type::Bias*>(tensor) != nullptr || dynamic_cast<back::Gradient*>(tensor) != nullptr)
			tensor->initialize(this);
}

DeviceInstance& DeviceInstance::create(cl::Device& cl_device, int id)
{
	allocate_mutex.lock();
	DeviceInstance& I = DeviceInstance::ALL[id];
	allocate_mutex.unlock();
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
			size = volume * sizeof(float);
			pointer = new float[volume];
			memset(pointer, 0, size); //Every byte of initialized Tensor is starting from zero
		}
		allocate_mutex.unlock();

		if (I != nullptr && I->pointers.count(this) == 0) { //idempotent
			I->pointers[this] = new float[size / sizeof(float)]; // only rely on size, not on volume/dimensions
			memcpy(I->pointers[this], pointer, size); //initialized by tensor's pointer
			const auto& context = I->queue.getInfo<CL_QUEUE_CONTEXT>();
			I->buffers[this] = cl::Buffer(context, CL_MEM_READ_WRITE, size);
			download(*I); //initialize from tensor itself
		}
	}
}

type::MiniBatch::MiniBatch(int size, int total, bool shuffle) : batch_size(size), use_shuffle(shuffle)
{
	set_total_samples(total);
}

void type::MiniBatch::set_total_samples(int64 N)
{
	shape_with({ N + 1 });
	total_batches = int(N / batch_size);
	if (N % batch_size != 0 && (CLNET_TENSOR_GLOBALS & CLNET_VALUE_MISMATCH_WARN))
		logger << (N % batch_size) << " samples at tail were abandoned." << endl;
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
	if (use_shuffle)
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

void CL_CALLBACK assembling_event_callback(cl_event, cl_int, void * user_data)
{
	auto out_event = reinterpret_cast<AssemblingEvent*>(user_data);
	if (--out_event->counter == 0)
		out_event->event->setStatus(CL_COMPLETE);
}

void CL_CALLBACK gradients_event_callback(cl_event, cl_int, void * user_data)
{
	DeviceInstance* instance = reinterpret_cast<DeviceInstance*>(user_data);
	if (instance->gradients_state-- == 1) {
		queue_mutex.lock();
		message_queue.push(MSG_CODE(instance->ID, MSG_GRADIENTS_READY));
		queue_mutex.unlock();
		notification.notify_all();
	}
}

void CL_CALLBACK parameters_event_callback(cl_event, cl_int, void * user_data)
{
	DeviceInstance* instance = reinterpret_cast<DeviceInstance*>(user_data);
	if (instance->parameters_state-- == 1) {
		queue_mutex.lock();
		message_queue.push(MSG_CODE(instance->ID, MSG_PARAMETERS_READY));
		queue_mutex.unlock();
		notification.notify_all();
	}
}

void Updater::synchronize_device_parameters(DeviceInstance& I)
{
	if (I.ID < 0 || I.parameters_state > 0)
		return; //already disposed in MSG_GRADIENTS_READY
	int i, N = static_cast<int>(peers.size());
#pragma omp parallel for
	for (i = 0; i < N; i++) {
		auto parameter = peers[i];
		memcpy(I.pointers[parameter], parameter->pointer, parameter->size);
	}
	I.parameters_state = peers.size();
}

void Updater::global_updater_thread(DeviceInstance& I)
{
	bool running = true;
	while (running) {
		notification.wait(notification_lock, [] { return !message_queue.empty(); });

		queue_mutex.lock();
		int64 command = message_queue.front();
		message_queue.pop();
		queue_mutex.unlock();
		int64 message = command & 0xFFFFFFFF;
		int ID = static_cast<int>(command >> 32);
		DeviceInstance& source = DeviceInstance::ALL[ID];

		switch (message) {
		case MSG_GRADIENTS_READY:
			run_globally(I, source);
			for (auto& iter : DeviceInstance::ALL)
				synchronize_device_parameters(iter.second);
			break;

		case MSG_PARAMETERS_READY:
			synchronize_device_parameters(source);
			break;

		case MSG_QUIT:
			I.free();
			running = false;
			break;
		}
	}
}

void wait_for_all_kernels_finished(DeviceInstance& I)
{
	for (auto& iter : I.events)
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
			out << "OpenCL Platform:  " << platform0.getInfo<CL_PLATFORM_NAME>().c_str() << "\n";
			out << "Version:          " << platform0.getInfo<CL_PLATFORM_VERSION>().c_str() << "\n";
			out << "Vendor:           " << platform0.getInfo<CL_PLATFORM_VENDOR>().c_str() << "\n";
			out << "Profile:          " << platform0.getInfo<CL_PLATFORM_PROFILE>().c_str() << "\n";
			out << "Platform Devices: " << "\n";
		}
		string name = device.getInfo<CL_DEVICE_NAME>();
		auto deviceType = device.getInfo<CL_DEVICE_TYPE>();
		auto sizesItem = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

		out << "\tDevice Name:         " << name.c_str() << "\n";
		out << "\tclNET device ID:     " << n << "\n";
		out << "\tType:                ";
		switch (deviceType) {
		case CL_DEVICE_TYPE_CPU: out << "CPU"; break;
		case CL_DEVICE_TYPE_GPU: out << "GPU"; break;
		default: out << "OTHER"; break;
		}
		out << "\n\tVersion:             " << device.getInfo<CL_DEVICE_VERSION>().c_str() << '/' << device.getInfo<CL_DRIVER_VERSION>().c_str() << "\n";
		out << "\tGlobal/Local Memory: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << '/' << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << " bytes\n";
		out << "\tMax ComputeUnits:    " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
		out << "\tMax WorkItem Sizes:  [" << sizesItem[0];
		for (size_t i = 1; i < sizesItem.size(); i++)
			out << ',' << sizesItem[i];
		out << "]\n";
		out << "\tBuiltIn Kernels:     ";
		try {
			out << device.getInfo<CL_DEVICE_BUILT_IN_KERNELS>().c_str();
		} catch (cl::Error& e) {
			out << "Error in " << e.what() << " (" << e.err() << "): " << clErrorCodeDescriptions[-e.err()] << "\n";
		}
		out << "\n\tExtensions:          " << device.getInfo<CL_DEVICE_EXTENSIONS>().c_str();
		out << "\n" << endl;
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

void stop_global_updater_thread()
{
	queue_mutex.lock();
	message_queue.push(MSG_CODE(-1L, MSG_QUIT));
	queue_mutex.unlock();
	notification.notify_all();
	global_updater->join();
	delete global_updater;
}

void OpenCL_::run(Tensor& graph, vector<int> targetDeviceIDs, int debugger_device_id, int master_device_id)
{
	vector<cl::Device>& targetDevices = find_devices();
	if (targetDevices.empty()) {
		logger << "No OpenCL device found." << endl;
		return;
	}

	int no, deviceNum = (int) targetDeviceIDs.size();
	if (deviceNum == 1)
		CLNET_TENSOR_GLOBALS |= CLNET_RUN_ON_SINGLE_DEVICE;
	auto updater = graph.peers.empty() || deviceNum == 1? nullptr : dynamic_cast<type::Updater*>(graph.peers[0]);
	thread_barrier barrier(deviceNum);
	if (updater != nullptr) {
		auto& device = targetDevices[master_device_id];
		const auto& name = device.getInfo<CL_DEVICE_NAME>();

		size_t time = MILLIS(0);
		auto& I = DeviceInstance::create(device, -1);
		time = MILLIS(time);
		logger << "[master] runs on " << name.c_str() << " (" << master_device_id << ") (kernels build: " << millis_string(time) << ")" << endl;
		global_updater = new thread(&Updater::global_updater_thread, updater, ref(I));
	}

#pragma omp parallel for
	for (no = 0; no < deviceNum; no++) {
		try {
			int device_id = targetDeviceIDs[no];
			auto& device = targetDevices[device_id];
			const auto& name = device.getInfo<CL_DEVICE_NAME>();

			size_t time = MILLIS(0);
			time_t seconds = time / 1000;
			tm* current = localtime(&seconds);
			char start_time[32];
			sprintf(start_time, "%04d-%02d-%02d %02d:%02d:%02d", current->tm_year + 1900, current->tm_mon + 1, current->tm_mday, current->tm_hour, current->tm_min, current->tm_sec);
			auto& I = DeviceInstance::create(device, device_id);
			time = MILLIS(time);
			logger << "[" << I.ID << ",@"<< start_time << "] " << name.c_str() << " (kernels build: " << millis_string(time) << ")" << endl;
			if (debugger_device_id == device_id)
				launch_debugger_thread(I, graph);
			if (updater != nullptr) {
				I.gradients_state = updater->peers.size();
				barrier.wait();
			}

			set<Tensor*> visited;
			time = MILLIS(0);
			graph.launch(&visited, &I);
			time = MILLIS(time);
			logger << "[" << I.ID << "] run time: " << millis_string(time) << "." << endl;

			if (updater != nullptr) {
				barrier.wait();
				if (no == 0)
					stop_global_updater_thread();
				barrier.wait();
			}
			I.free();
		}
		catch (cl::Error& e) {
			if (_current != nullptr)
				logger << "Current Tensor: " << type_name(_current) << ": " << _current->alias << "\n";
			logger << "Error in device " << targetDeviceIDs[no] << ": " << e.what() << " (" << e.err() << "): " << clErrorCodeDescriptions[e.err() < MIN_ERROR_CODE? -USER_ERROR_DESCRIPTION_UNDEFINED : -e.err()] << endl;
		}
		catch (runtime_error& e) {
			logger << "Runtime error: " << e.what() << endl;
		}
	}
}

void display_tensor_name(Tensor* current, void* padding)
{
	string& pad = *static_cast<string*>(padding);
	logger << pad << type_name(current) << "\t\t" << current->alias;
	if (!current->dimensions.empty()) {
		logger << "[" << current->dimensions[0];
		for (size_t i = 1; i < current->dimensions.size(); i++)
			logger << "," << current->dimensions[i];
		logger << "]";
	}
	logger << "\n";

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
		logger << "-";
	for (auto aux : others)
		display_tensor_name(aux, static_cast<void*>(&indent));
}

void OpenCL_::print_tensor_structure(Tensor& graph)
{
	string padding;
	set<Tensor*> visited;
	graph.launch(&visited, static_cast<void*>(&padding), display_tensor_name);
	logger << endl;
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
	return iter != map.end()? stoi(iter->second) : default_value;
}

template <> size_t optional(unordered_map<string, string>& map, string name, size_t default_value)
{
	auto iter = map.find(name);
	return iter != map.end()? strtoull(iter->second.c_str(), nullptr, 10) : default_value;
}

template <> int64 optional(unordered_map<string, string>& map, string name, int64 default_value)
{
	auto iter = map.find(name);
	return iter != map.end()? strtoll(iter->second.c_str(), nullptr, 10) : default_value;
}

template <> double optional(unordered_map<string, string>& map, string name, double default_value)
{
	auto iter = map.find(name);
	return iter != map.end()? atof(iter->second.c_str()) : default_value;
}

template <> float optional(unordered_map<string, string>& map, string name, float default_value)
{
	auto iter = map.find(name);
	return iter != map.end()? (float) atof(iter->second.c_str()) : default_value;
}

template <typename T> T optional(std::string name, T default_value)
{
	return optional<T>(key_values, name, default_value);
}

template string optional(string name, string default_value);
template int optional(string name, int default_value);
template size_t optional(string name, size_t default_value);
template int64 optional(string name, int64 default_value);
template double optional(string name, double default_value);
template float optional(string name, float default_value);

Logger::Logger() : count(0) {}

Logger& Logger::operator +=(string filename)
{
	//This sentance leads to the stream never normally closed.Considering this method is designed to unexpected occasion such as driver crash, I reserved it for convenience.
	//For the occasion which concerns close behavior, use Logger::operator +=(ostream& os) instead.
	streams[count++] = new ofstream(filename, std::ofstream::binary | std::ofstream::out | std::ofstream::app);
	return *this;
}

Logger& Logger::operator +=(ostream& os)
{
	streams[count++] = &os;
	return *this;
}

stringstream& Logger::thread_buffer()
{
	stringstream* os;
	const auto& iter = buffers.find(this_thread::get_id());
	if (iter != buffers.end())
		os = &iter->second;
	else {
		safe_access.lock();
		os = &buffers[this_thread::get_id()];
		safe_access.unlock();
	}
	return *os;
}

Logger& Logger::operator <<(ostream& (*fp)(ostream&))
{
	auto& buffer = thread_buffer();
	buffer << fp;
	for (auto p = streams, end = p + count; p < end; p++)
		**p << buffer.str() << flush;
	buffer.str("");
	return *this;
}

}
