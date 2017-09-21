/*
 * main.cpp
 *
 *  Created on: 2017/1/31
 *      Author: ZhangHua
 */

#include <iostream>
#include <fstream>

#include <tensor.hpp>
#include <device_instance.hpp>

#ifndef _MSC_VER
	#ifdef __MINGW64__
		#define PATH_CHAR '\\'
	#else
		#define PATH_CHAR '/'
	#endif
#else
	#pragma warning(disable : 4996)
	#define PATH_CHAR '\\'
#endif

using namespace std;
using namespace clnet;

namespace clnet {
extern unordered_map<string, string> key_values;
extern Tensor* _breakpoint;
}

string locate_resources(string path)
{
	while (true) {
		ifstream license(path + "LICENSE");
		if (!license) {
			auto pos = path.find_last_of(PATH_CHAR, path.length() - 2);
			if (pos == string::npos)
				return string();
			path = path.substr(0, pos + 1);
		}
		else {
			license.close();
			break;
		}
	}
	return path;
}

int main(int argc, char** argv)
{
	OpenCL.location = locate_resources(argv[0]);
	if (OpenCL.location.empty()) {
		cout << "LICENSE file not found. Cannot locate resource file (argv[0]: " << argv[0] << "). Use powershell instead in Windows OS." << endl;
		return 1;
	}

	if (argc < 2) {
		cout << "OpenCLNet [model] [/[0,1,...]] [/p] [/ld] [/ds] [/os] [/nf] [/nd] [/ss] [/cpu] [:{key} {value}]\n";
		cout << "model\t\tcurrent support: MLP,MLP_softmax,charRNN,MNIST_CNN\n";
		cout << ":{key} {value}\tlet named parameter {key} equal to {value}\n";
		cout << "/ld\t\tlist devices\n";
		cout << "/[0,1,...]\trunning device no.\n";
		cout << "/p\t\tpredict mode\n";
		cout << "/ds\t\tdisplay structure\n";
		cout << "/os\t\tdisplay opencl source\n";
		cout << "/nf\t\tturn off fusion optimization\n";
		cout << "/nd\t\tturn off debugger thread\n";
		cout << "/ss\t\tstop on startup at root tensor (must turn on debugger)\n";
		cout << "/cpu\t\tuse CPU instead of GPU (GPU is default)\n";
		cout << "/nlogc\t\tturn off console output\n";
		cout << "/logf\t\tlog to file (clnet.log is default log file)" << endl;
		return 1;
	}

	Tensor* graph = nullptr;
	bool use_debugger = true, stop_on_startup = false, list_devices = false, display_structure = false, console_output = true, log_to_file = false;
	vector<int> devices;
	for (int i = 1; i < argc; i++) {
		string param(argv[i]);
		if (param.empty())
			return 1;
		else if (param[0] == ':')
			key_values[param.substr(1)] = argv[++i];
		else if (param == "/p")
			CLNET_TENSOR_GLOBALS |= CLNET_PREDICT_ONLY;
		else if (param[1] == '[' && param[param.length() - 1] == ']')
			parse_dimensions<int>(param.substr(1), &devices);
		else if (param == "/nd")
			use_debugger = false;
		else if (param == "/ss")
			stop_on_startup = true;
		else if (param == "/ld")
			list_devices = true;
		else if (param == "/ds")
			display_structure = true;
		else if (param == "/nf")
			CLNET_TENSOR_GLOBALS ^= CLNET_FEED_FORWARD_FUSION | CLNET_BACK_PROPAGATE_FUSION;
		else if (param == "/os")
			CLNET_TENSOR_GLOBALS |= CLNET_OPENCL_SHOW_SOURCE;
		else if (param == "/cpu")
			OpenCL.device_type = CL_DEVICE_TYPE_CPU;
		else if (param == "/nlogc")
			console_output = false;
		else if (param == "/logf")
			log_to_file = true;
		else
			key_values["model"] = param;
	}

	if (log_to_file) {
		logger += optional<string>("log_file", OpenCL.location + "clnet.log");
		for (auto p = argv, end = argv + argc; p < end; p++) {
			string param(*p);
			if (param.find(' ') != string::npos)
				param = "\"" + param + "\"";
			logger << param << " ";
		}
		logger << endl;
	}
	if (console_output)
		logger += cout;
	bool is_predict = CLNET_TENSOR_GLOBALS & CLNET_PREDICT_ONLY;
	if (devices.empty())
		devices = {0};
	extern T MLP();
	extern T MLP_softmax();
	extern T charRNN(bool is_predict);
	extern T MNIST_CNN(bool is_predict);
	extern T kernel_test();

	auto model = optional<string>("model", "MLP");
	if (model == "MLP")
		graph = &MLP();
	else if (model == "MLP_softmax")
		graph = &MLP_softmax();
	else if (model == "charRNN")
		graph = &charRNN(is_predict);
	else if (model == "MNIST_CNN")
		graph = &MNIST_CNN(is_predict);
	else if (model == "kernel_test")
		graph = &kernel_test();
	else if (model == "reference") { //run reference
		extern void train_on_device(int device);
		extern bool showInfo(std::ostream& out);
		showInfo(std::cout);
		train_on_device(devices[0]);
		return 0;
	}
	else {
		cout << "Unknown model: " << model << endl;
		return 1;
	}

	if (list_devices)
		OpenCL.print_device_info(cout);
	if (display_structure)
		OpenCL.print_tensor_structure(*graph);
	if (stop_on_startup)
		_breakpoint = graph;
	OpenCL.run(*graph, devices, use_debugger);
	return 0;
}

T kernel_test()
{
	auto a = new Tensor({32}, {}, "a");
	auto b = new Tensor({32}, {}, "b");
	auto result = new Tensor({32}, {}, "result");
	float a_data[32] = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			b_data[32] = {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
	a->initialize();
	b->initialize();
	memcpy(a->pointer, a_data, sizeof(a_data));
	memcpy(b->pointer, b_data, sizeof(b_data));

	return *new InstantTensor("kernel_test", {}, {a, b}, [result](InstantTensor* self, DeviceInstance& I) {
		auto& kernel = prepare_for_running_kernel(self, I);
		kernel.setArg(0, I.buffers[result]);
		kernel.setArg(1, I.buffers[self->peers[0]]);
		kernel.setArg(2, I.buffers[self->peers[1]]);
		kernel.setArg(3, 2);
//			cl::NDRange local(2);
		cl::NDRange global(2);
		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[result]);
		wait_for_all_kernels_finished(I);
		result->upload(I);
		for (float* p = I.pointers[result], *end = p + result->dimensions[0]; p < end; p++)
			cout << *p << "\n";
	}, [](InstantTensor* self, DeviceInstance& I) -> string{ //This example used as a trial to test OpenCL kernel code
		return std::string{R"CLC(
kernel void kernel_test(global float* out, global float* a, global float* b, const int index_size)
{
	const int GID = get_global_id(0);
	float16 result = ((global float16*) a)[GID] * ((global float16*) b)[GID];
	float8 r8 = result.lo + result.hi;
	float4 r4 = r8.lo + r8.hi;
	float2 r2 = r4.lo + r4.hi;
	out[GID] = r2.lo + r2.hi; 
}
)CLC"};
	});
}
