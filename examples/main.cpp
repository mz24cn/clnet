/*
 * main.cpp
 *
 *  Created on: 2017/1/31
 *      Author: ZhangHua
 */

#include <csignal>
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
extern int64 microseconds;
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
	signal(SIGINT, [](int signal) {
		logger << "User breaks by Ctrl+C." << endl;
		CLNET_TENSOR_GLOBALS |= CLNET_STEP_INTO_MODE;
		for (auto& iter : DeviceInstance::ALL)
			wait_for_all_kernels_finished(iter.second);
		exit(1);
	});

	OpenCL.location = locate_resources(argv[0]);
	if (OpenCL.location.empty()) {
		cout << "LICENSE file not found. Cannot locate kernel file via this executable program. Try accessing " << argv[0] << " in parent directory or in the form of absolute path." << endl;
		return 1;
	}

	if (argc < 2) {
		cout << "OpenCLNet [model] [/0,1,...] [/p] [/ld] [/ds] [/os] [/nf] [/nd] [/ss] [/all] [:{key} {value}]\n";
		cout << "model\t\tcurrent support: MLP,MLP_softmax,charRNN,MNIST_CNN,CIFAR_WRN\n";
		cout << "/ld\t\tlist devices\n";
		cout << "/all\t\tuse all device types including CPU and ACCELERATOR (GPU only is default)\n";
		cout << ":{key} {value}\tset named parameter with {key}, {value} pair\n";
		cout << "/0,1,...\trun device ID (use ':master' and ':debugger' to set global update and debugger thread run device)\n";
		cout << "/p\t\tpredict mode\n";
		cout << "/pm\t\tprint memory\n";
		cout << "/pp\t\tprint parameters\n";
		cout << "/pf\t\tenable profile mode\n";
		cout << "/ds\t\tdisplay structure\n";
		cout << "/os\t\tdisplay opencl source\n";
		cout << "/nf\t\tturn off fusion optimization\n";
		cout << "/nd\t\tturn off debugger thread\n";
		cout << "/ss\t\tstop on startup at root tensor (must turn on debugger)\n";
		cout << "/nlogc\t\tturn off console output\n";
		cout << "/logf\t\tlog to file ({project root}/clnet.log is default log file)" << endl;
		return 1;
	}

	Tensor* graph = nullptr;
	bool use_debugger = true, stop_on_startup = false, list_devices = false, display_structure = false, only_operator = false,
			console_output = true, log_to_file = false, print_memory = false, print_parameters = false, profile_mode = false;
	vector<int> devices;
	for (int i = 1; i < argc; i++) {
		string param(argv[i]);
		if (param.empty())
			return 1;
		else if (param[0] == ':' && i + 1 < argc)
			key_values[param.substr(1)] = argv[++i];
		else if (param == "/p")
			CLNET_TENSOR_GLOBALS |= CLNET_PREDICT_ONLY;
		else if (param == "/nd")
			use_debugger = false;
		else if (param == "/ss")
			stop_on_startup = true;
		else if (param == "/ld")
			list_devices = true;
		else if (param == "/ds")
			display_structure = true;
		else if (param == "/dso") {
			display_structure = true;
			only_operator = true;
		}
		else if (param == "/pm")
			print_memory = true;
		else if (param == "/pp")
			print_parameters = true;
		else if (param == "/pf")
			profile_mode = true;
		else if (param == "/nf")
			CLNET_TENSOR_GLOBALS ^= CLNET_FEED_FORWARD_FUSION | CLNET_BACK_PROPAGATE_FUSION;
		else if (param == "/os")
			CLNET_TENSOR_GLOBALS |= CLNET_OPENCL_SHOW_SOURCE;
		else if (param == "/all")
			OpenCL.device_type = CL_DEVICE_TYPE_ALL;
		else if (param == "/nlogc")
			console_output = false;
		else if (param == "/logf")
			log_to_file = true;
		else if (param[0] == '/') {
			if ((param[1] == '[' && param[param.length() - 1] == ']') || (param[1] >= '0' && param[1] <= '9')) //Linux shell strips '[' and ']' in "/[1,2]"
				parse_dimensions<int>(param.substr(1), &devices);
			else
				cout << "Unknown option " << param << " ignored." << endl;
		}
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
	int device_master = optional<int>("master", devices[0]);
	int device_debugger = optional<int>("debugger", use_debugger? devices[0] : -1);
	extern T MLP();
	extern T MLP_softmax();
	extern T charRNN(bool is_predict);
	extern T MNIST_CNN(bool is_predict);
	extern T CIFAR_WRN(bool is_predict);
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
	else if (model == "CIFAR_WRN")
		graph = &CIFAR_WRN(is_predict);
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
	if (display_structure) {
		OpenCL.print_tensor_structure(*graph, only_operator);
		logger << endl;
	}
	if (print_memory)
		OpenCL.print_tensor_memory();
	if (print_parameters)
		OpenCL.print_parameters(*graph);
	if (stop_on_startup)
		_breakpoint = graph;
	if (profile_mode)
		microseconds = 1;
	OpenCL.run(*graph, devices, device_debugger, device_master);
	return 0;
}
