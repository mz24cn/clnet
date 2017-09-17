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
		cout << "OpenCLNet [model] [/[0,1,...]] [/p] [/ld] [/ds] [/os] [/nf] [/nd] [/ss] [/cpu] [/: \"{sample}\"] [{key}:{value}]" << endl;
		cout << "model\t\tcurrent support: MLP,MLP_softmax,charRNN,MNIST_CNN" << endl;
		cout << "{key}:{value}\tprovide named parameters" << endl;
		cout << "/: \"{sample}\"\ttext parameter (named as 'sample')" << endl;
		cout << "/ld\t\tlist devices" << endl;
		cout << "/[0,1,...]\trunning device no." << endl;
		cout << "/p\t\tpredict mode" << endl;
		cout << "/ds\t\tdisplay structure" << endl;
		cout << "/os\t\tdisplay opencl source" << endl;
		cout << "/nf\t\tturn off fusion optimization" << endl;
		cout << "/nd\t\tturn off debugger thread" << endl;
		cout << "/ss\t\tstop on startup at root tensor (must turn on debugger)" << endl;
		cout << "/cpu\t\tuse CPU instead of GPU (GPU is default)" << endl;
		return 1;
	}

	Tensor* graph = nullptr;
	bool use_debugger = true, stop_on_startup = false, list_devices = false, display_structure = false;
	vector<int> devices;
	for (int i = 1; i < argc; i++) {
		string param(argv[i]);
		if (param.length() <= 1 || param[0] != '/') {
			auto n = param.find(':');
			if (n != string::npos)
				key_values[param.substr(0, n)] = param.substr(n + 1);
			else
				key_values["model"] = param;
		}
		else if (param == "/nd")
			use_debugger = false;
		else if (param == "/ss")
			stop_on_startup = true;
		else if (param[1] == '[' && param[param.length() - 1] == ']')
			parse_dimensions<int>(param.substr(1), &devices);
		else if (param == "/p")
			CLNET_TENSOR_GLOBALS |= CLNET_PREDICT_ONLY;
		else if (param == "/:")
			key_values["sample"] = argv[++i];
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
		else {
			cout << "Unknown parameter: " << param << endl;
			return 1;
		}
	}

	bool is_predict = CLNET_TENSOR_GLOBALS & CLNET_PREDICT_ONLY;
	if (devices.empty())
		devices = {0};
	extern T MLP();
	extern T MLP_softmax();
	extern T charRNN(bool is_predict);
	extern T MNIST_CNN(bool is_predict);

	auto model = optional<string>("model", "MLP");
	if (model == "MLP")
		graph = &MLP();
	else if (model == "MLP_softmax")
		graph = &MLP_softmax();
	else if (model == "charRNN")
		graph = &charRNN(is_predict);
	else if (model == "MNIST_CNN")
		graph = &MNIST_CNN(is_predict);
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
