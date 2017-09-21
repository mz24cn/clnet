/*
 * console.cpp
 *
 *  Created on: 2017/6/23
 *      Author: ZhangHua
 */

#include <condition_variable>
#include <iostream>
#include <sstream>
#include <fstream>
#include <thread>
#include <typeinfo>
#include <stdexcept>
#include <chrono>
#include <algorithm>

#include <tensor.hpp>
#include <device_instance.hpp>

using namespace std;
using namespace clnet::type;

int MAX_X = 40, MAX_Y = 10, MAX_Z = 5;

namespace clnet {
extern Tensor *_current, *_breakpoint;
extern size_t microseconds, breakpoint_hit_times;
extern unordered_map<Tensor*, size_t> kernels_cost;

thread *debugger;
condition_variable breakpoint;
mutex breakpoint_mutex;
unique_lock<mutex> breakpoint_lock(breakpoint_mutex);

// high resolution timer. Generally using MILLIS() is enough.
size_t MICROS(size_t microseconds)
{
	using namespace std::chrono;
	high_resolution_clock::time_point now = high_resolution_clock::now();
	return duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count() - microseconds;
}

string type_name(Tensor* tensor)
{
	if (tensor == nullptr)
		return "NULL";

	string name = typeid(*tensor).name();
#ifdef _MSC_VER
	name = name.substr(name.find_last_of(' ') + 1);
#else
	if (name[0] == 'N' && name[name.size() - 1] == 'E')
		name = name.substr(1, name.size() - 2);
	stringstream ss;
	ss << name;
	string item;
	name.clear();
	while (true) {
		int size = 0;
		ss >> size;
		if (size == 0)
			break;
		item.resize(size);
		for (int i = 0; i < size; i++)
			ss >> item[i];
		if (!name.empty())
			name.append("::");
		name.append(item);
	}
#endif
	return name;
}

void save_tensor(Tensor* tensor, ostream& os, DeviceInstance& I)
{
	os << tensor->alias << '\n';
	os << type_name(tensor) << '\n';
	os << sizeof(float) << '\n';
	os << tensor->dimensions.size() << '\n';
	for (auto dim : tensor->dimensions)
		os << dim << ' ';
	os << tensor->size << '\n';
	tensor->upload(I);
	os.write(reinterpret_cast<char*>(I.pointers[tensor]), tensor->size);
	os << '\n';
}

void save_tensor_as_csv(Tensor* tensor, const string& file, DeviceInstance& I, bool use_header)
{
	ofstream ofs(file);
	if (!ofs)
		throw runtime_error("failed to open file " + file + ".");
	if (use_header) {
		ofs << tensor->alias << ',';
		ofs << type_name(tensor) << ',';
		ofs << tensor->dimensions.size();
		for (auto dim : tensor->dimensions)
			ofs << ',' << dim;
		ofs << '\n';
	}

	tensor->upload(I);
	const float* pointer = I.pointers[tensor];
	int columns = tensor->dimensions.back();
	int rows = tensor->volume / columns;
	for (int i = 0; i < rows; i++) {
		for (int j = 1; j < columns; j++)
			ofs << *pointer++ << ',';
		ofs << *pointer++ << '\n';
	}
	ofs.close();
}

vector<Tensor*> load_tensors(istream& is, DeviceInstance& I)
{
	string typeName, name;
	vector<Tensor*> tensors;
	char separator;
	while (is >> name) {
		is >> typeName;
		Tensor* tensor = locate_tensor(name);
		int size, V;
		is >> size; //sizeof(float)
		is >> size; //tensor->dimensions.size()
		for (int i = 0; i < size; i++) {
			is >> V;
			if (tensor->dimensions.size() > static_cast<size_t>(i)) {
				if (tensor->dimensions[i] != V)
					throw runtime_error("incorrect tensor dimensions: " + to_string(V) + " for " + name);
			}
			else
				tensor->dimensions.push_back(V);
		}
		is >> size >> noskipws >> separator; //tensor->size
		tensor->initialize(&I);
		is.read(reinterpret_cast<char*>(I.pointers[tensor]), tensor->size);
		tensor->download(I);
		is >> separator;
		is >> skipws;
		tensors.push_back(tensor);
	}
	return tensors;
}

Tensor* load_tensor_from_csv(const string& file, DeviceInstance& I, Tensor* tensor)
{
	ifstream ifs(file);
	if (!ifs)
		throw runtime_error("failed to open file " + file + ".");

	stringstream ss;
	char comma;
	string line, name;
	int n, m;
	if (tensor == nullptr) {
		getline(ifs, line);
		n = line.find(',');
		name = line.substr(0, n);
		tensor = locate_tensor(name);
		m = line.find(',', ++n);
		name = line.substr(n, m - n); //type name
		name = line.substr(++m); //dimension size and dimensions

		vector<int64> dims;
		ss << name;
		ss >> n >> comma;
		if (tensor->dimensions.size() != static_cast<size_t>(n))
			throw runtime_error("wrong dimension vector size in file " + file + ".");
		for (int i = 0; i < n; i++) {
			ss >> m >> comma;
			if (m != tensor->dimensions[i])
				throw runtime_error("wrong dimension size in file " + file + ".");
			dims.push_back(m);
		}
	}

	int columns = tensor->dimensions.back();
	int rows = tensor->volume / columns;
	float* pointer = I.pointers[tensor];
	for (int i = 0; i < rows; i++) {
		getline(ifs, line);
		ss.str("");
		ss.clear();
		ss << line;
		for (int j = 0; j < columns; j++)
			ss >> *pointer++ >> comma;
	}
	tensor->download(I);
	return tensor;
}

Tensor* locate_tensor(string name)
{
	bool use_prefix = name[name.size() - 1] == '*';
	if (use_prefix)
		name = name.substr(0, name.size() - 1);
	for (auto tensor : Tensor::ALL)
		if (use_prefix && tensor->alias.find(name) == 0)
			return tensor;
		else if (tensor->alias == name)
			return tensor;
		else if (tensor->alias.empty() && type_name(tensor) == name)
			return tensor;
	throw runtime_error("tensor " + name + " not found.");
}

template <typename T> void parse_dimensions(string subprints, vector<T>* low, vector<T>* high, const vector<T>* limits, std::vector<T>* reshaped)
{
	if (subprints.empty())
		return;
	if (subprints[0] == '[' && subprints[subprints.length() - 1] == ']')
		subprints = subprints.substr(1, subprints.length() - 2);

	vector<string> ranges;
	stringstream ss;
	ss << subprints;
	string range;
	while (getline(ss, range, ','))
		ranges.push_back(range);

	for (auto& str : ranges) {
		auto n = str.find(':');
		if (n == string::npos) {
			low->push_back(atoi(str.data()));
			if (high != nullptr)
				high->push_back(INT_MIN);
		}
		else {
			auto left = str.substr(0, n);
			auto right = str.substr(n + 1, str.length() - n - 1);
			if (high != nullptr) {
				if (right.empty())
					high->push_back(INT_MAX);
				else
					high->push_back(atoi(right.data()));
			}
			int n_left = left.empty()? 0 : atoi(left.data());
			low->push_back(n_left);
		}
	}

	if (limits != nullptr) {
		if (limits->size() < low->size())
			throw runtime_error("too many subprints: [" + subprints + "]");
		else {
			int i, N;
			if (limits->size() > low->size()) {
				if (reshaped == nullptr)
					throw runtime_error("failed to reshape to " + subprints);
				int n = limits->at(0);
				for (i = 1, N = limits->size() - low->size() + 1; i < N; i++)
					n *= limits->at(i);
				reshaped->assign({ n });
				for (N = limits->size(); i < N; i++)
					reshaped->push_back(limits->at(i));
				limits = reshaped;
			}
			else if (reshaped != nullptr)
				reshaped->assign(limits->begin(), limits->end());

			for (i = 0, N = low->size(); i < N; i++)
				if (low->at(i) < -limits->at(i))
					throw runtime_error("index out of bounds: " + to_string(low->at(i)));
				else if (low->at(i) < 0)
					low->at(i) += limits->at(i);
				else if (low->at(i) >= limits->at(i))
					throw runtime_error("index out of bounds: " + to_string(low->at(i)));
			if (high != nullptr) {
				for (i = 0, N = high->size(); i < N; i++)
					if (high->at(i) == INT_MIN)
						high->at(i) = low->at(i) + 1;
					else if (high->at(i) < -limits->at(i))
						throw runtime_error("index out of bounds: " + to_string(high->at(i)));
					else if (high->at(i) <= 0)
						high->at(i) += limits->at(i);
					else if (high->at(i) == INT_MAX)
						high->at(i) = limits->at(i);
					else if (high->at(i) > limits->at(i))
						throw runtime_error("index out of bounds: " + to_string(high->at(i)));
			}
		}
	}
}

void describe_tensor(Tensor* tensor, bool only_name = true)
{
	if (tensor == nullptr) {
		logger << "NULL" << endl;
		return;
	}
	if (only_name) {
		logger << tensor->alias << "[";
		if (!tensor->dimensions.empty())
			logger << tensor->dimensions[0];
		for (size_t i = 1; i < tensor->dimensions.size(); i++)
			logger << "," << tensor->dimensions[i];
		logger << "]: " << type_name(tensor) << endl;
		return;
	}

	logger << "\tthis:\t\t\t\t" << tensor;
	logger << "\n\ttype:\t\t\t" << type_name(tensor);
	logger << "\n\talias:\t\t\t" << tensor->alias;
	logger << "\n\tvolume:\t\t\t" << tensor->volume;

	logger << "\n\tdimensions:\t\t[";
	if (!tensor->dimensions.empty())
		logger << tensor->dimensions[0];
	for (size_t i = 1; i < tensor->dimensions.size(); i++)
		logger << "," << tensor->dimensions[i];
	logger << "]";

	logger << "\n\tsize:\t\t\t\t" << tensor->size << " bytes";
	logger << "\n\tpointer:\t\t\t" << tensor->pointer;
	logger << "\n\tgradient:\t\t";
	describe_tensor(tensor->gradient);

	logger << "\tinputs:\n";
	for (auto input : tensor->inputs) {
		logger << "\t\t";
		describe_tensor(input);
	}

	logger << "\tpeers:\n";
	for (auto peer : tensor->peers) {
		logger << "\t\t";
		describe_tensor(peer);
	}
}

template <typename T> function<void(T&)> unitary_operation(string op, T value)
{
	if (op.empty())
		return [](T& a) { logger << a; };
	else if (op == "+=")
		return [value](T& a) { a += value; };
	else if (op == "=")
		return [value](T& a) { a = value; };
	else if (op == "*=")
		return [value](T& a) { a *= value; };
	else
		throw runtime_error("unsupported operation: " + op);
}

// Currently only allow display 1D, 2D and 3D data
template <typename T> void operate_tensor_data(Tensor* tensor, vector<int64>& low, vector<int64>& high, vector<int64>& reshaped, DeviceInstance& I, string op = "0", T value = (T)0)
{
	if (!reshaped.empty()) {
		logger << "data[";
		for (size_t i = 0; i < reshaped.size(); i++) {
			if (i > 0)
				logger << ",";
			if (high[i] - low[i] > 1)
				logger << low[i] << ":" << high[i];
			else
				logger << low[i];
			logger << "/" << reshaped[i];
		}
		logger << "] for ";
	}
	describe_tensor(tensor);
	int NZ = tensor->dimensions.size() < 3? 1 : tensor->dimensions[tensor->dimensions.size() - 3];
	int NR = tensor->dimensions.size() < 2? 1 : tensor->dimensions[tensor->dimensions.size() - 2];
	int NC = tensor->dimensions.back();
	if (tensor->volume > 0) {
		if (tensor->size == 0) { //now find the 'real' tensor
			auto p = I.pointers[tensor];
			for (auto target : Tensor::ALL)
				if (I.pointers[target] == p && tensor != target) {
					tensor = target;
					break;
				}
		}
	}
	else
		return;

	int which = op[0] - '0'; //0: watch cl::Buffer; 1: watch I.pointers; 2: watch tensor.pointer.
	op = op.substr(1);
	T* data;
	float* tmp = I.pointers[tensor];
	if (which == 0) {
		data = new T[tensor->volume];
		I.pointers[tensor] = reinterpret_cast<float*>(data);
		tensor->upload(I);
	}
	else
		data = reinterpret_cast<T*>(which == 1? tmp : tensor->pointer);

	int64 range_dimension = 0, ranges[64], subprints[64];
	if (reshaped.size() >= 64)
		throw runtime_error("too long subprint length.");
	for (size_t i = 0; i < reshaped.size(); i++) {
		ranges[i] = subprints[i] = low[i];
		if (low[i] < high[i] - 1)
			ranges[range_dimension++] = i;
	}
	auto shapes = reshaped.data();
	auto N = reshaped.size() - 1;
	auto element = [data, &subprints, N, shapes]() -> T& {
		int64 index = 0;
		for (size_t n = 0; n < N; n++)
			index = (index + subprints[n]) * shapes[n + 1];
		index += subprints[N];
		return data[index];
	};
	int64 &d0 = ranges[0], &d1 = ranges[1], &d2 = ranges[2];
	int64 &i = subprints[d0], &j = subprints[d1], &k = subprints[d2];

	auto operation = unitary_operation(op, value);
	function<void(int, int64)> number;
	if (op.empty())
		number = [](int type, int64 i) {
			switch (type) {
			case 0:
				logger << i << ":\n"; break;
			case 1:
				logger << i << ":\t"; break;
			case 2:
				logger << ","; break;
			case 3:
				logger << "\n"; break;
			case -1:
				logger << "\t"; break;
			}
		};
	else
		number = [](int, int64) {};

	if (reshaped.empty()) {
		int max_x = NC, max_y = NR, max_z = NZ;
		bool extra_x, extra_y;
		if (max_z > MAX_Z)
			max_z = MAX_Z;
		if (max_y > MAX_Y) {
			extra_y = true;
			max_y = MAX_Y;
		}
		else
			extra_y = false;
		if (max_x > MAX_X) {
			extra_x = true;
			max_x = MAX_X;
		}
		else
			extra_x = false;
		if (!op.empty())
			throw runtime_error("Update requires range. try to add \"[:]\".");

		for (int k = 0; k < max_z; k++) {
			logger << k << "\n";
			for (int i = 0; i < max_y; i++) {
				logger << i << ":\t" << data[k * NR * NC + i * NC];
				for (int j = 1; j < max_x; j++)
					logger << "," << data[k * NR * NC + i * NC + j];
				if (extra_x)
					logger << " ...";
				logger << "\n";
			}
			if (extra_y)
				logger << " ...\n";
		}
	}
	else if (range_dimension <= 1) {
		int iH = high[d0];
		i = low[d0];
		number(-1, i);
		operation(element());
		for (++i; i < iH; i++) {
			number(2, i);
			operation(element());
		}
		number(3, i);
	}
	else if (range_dimension == 2) {
		int iH = high[d0], jH = high[d1];
		for (i = low[d0]; i < iH; i++) {
			j = low[d1];
			number(1, i);
			operation(element());
			for (++j; j < jH; j++) {
				number(2, j);
				operation(element());
			}
			number(3, j);
		}
	}
	else if (range_dimension == 3) {
		int  iH = high[d0], jH = high[d1], kH = high[d2];
		for (i = low[d0]; i < iH; i++) {
			number(0, i);
			for (j = low[d1]; j < jH; j++) {
				k = low[d2];
				number(1, j);
				operation(element());
				for (++k; k < kH; k++) {
					number(2, k);
					operation(element());
				}
				number(3, k);
			}
		}
	}
	else
		throw runtime_error("too many ranges. try to reduce ':'.");
	if (which == 0) {
		if (!op.empty())
			tensor->download(I);
		I.pointers[tensor] = tmp;
		delete data;
	}
}

void debugger_thread(DeviceInstance& I, Tensor& graph)
{
	Tensor* last = nullptr;
	string command, name;
	logger << "[debugger] interactive thread started." << endl;

	while (true) {
		try {
			cin >> command;
			if (command == "g" || command == "goto") {
				if (CLNET_TENSOR_GLOBALS & CLNET_STEP_INTO_MODE) {
					CLNET_TENSOR_GLOBALS &= ~CLNET_STEP_INTO_MODE;
					logger << "[debugger] step into mode removed." << endl;
				}
				char c= cin.peek();
				if (c == '\n') {
					_breakpoint = nullptr;
					logger << "[debugger] breakpoint removed." << endl;
				}
				else {
					cin >> name;
					_breakpoint = locate_tensor(name);
					c = cin.peek();
					if (c == '\n')
						breakpoint_hit_times = 1;
					else {
						cin >> name;
						breakpoint_hit_times = atoi(name.data());
						logger << "[debugger] set breakpoint hit times: " << breakpoint_hit_times << endl;
					}
					breakpoint.notify_all();
				}
			}
			else if (command == "c" || command == "continue") {
				breakpoint_hit_times = 1;
				breakpoint.notify_all();
			}
			else if (command == "p" || command == "pause") {
				_breakpoint = graph.peers[0];
				breakpoint_hit_times = 1;
				logger << "[debugger] breakpoint added on " << _breakpoint->alias << "." << endl;
			}
			else if (command == "s" || command == "step") {
				CLNET_TENSOR_GLOBALS ^= CLNET_STEP_INTO_MODE;
				logger << "[debugger] step into mode " << ((CLNET_TENSOR_GLOBALS & CLNET_STEP_INTO_MODE)? "activated." : "removed.") << endl;
			}
			else if (command == "d" || command == "describe") {
				cin >> name;
				auto tensor = locate_tensor(name);
				describe_tensor(tensor, false);
			}
			else if (command == "pf" || command == "profile") {
				char c= cin.peek();
				if (c == '\n') {
					microseconds = microseconds > 0? 0 : 1;
					logger << "[debugger] profile mode " << (microseconds > 0? "activated." : "removed.") << endl;
				}
				else {
					cin >> name;
					if (name == "clear") {
						kernels_cost.clear();
						logger << "[debugger] profile results cleared." << endl;
					}
					else if (name == "list") {
						vector<Tensor*> tensors;
						for (auto iter : kernels_cost)
							if (iter.second > 0)
								tensors.push_back(iter.first);
						auto compare = [](Tensor* a, Tensor* b) -> bool { return kernels_cost[a] > kernels_cost[b]; };
						sort(tensors.begin(), tensors.end(), compare);
						for (auto tensor : tensors)
							logger << tensor->alias << ": " << type_name(tensor) << ": \t\t" << millis_string(kernels_cost[tensor] / 1000) << endl;
					}
				}
			}
			else if (command == "rk" || command == "reload_kernel") {
				logger << "[debugger] waiting ...\n";
				const auto& device = I.queue.getInfo<CL_QUEUE_DEVICE>();
				const auto& context = I.queue.getInfo<CL_QUEUE_CONTEXT>();
				reload_kernels(device, context, I);
				logger << "[debugger] kernels reloaded." << endl;
			}
			else if (command == "save") {
				cin >> name;
				vector<Tensor*> tensors;
				for (auto tensor : Tensor::ALL)
					if (dynamic_cast<type::Bias*>(tensor) != nullptr || dynamic_cast<type::Weight*>(tensor) != nullptr)
						tensors.push_back(tensor);

				string tensor_names;
				if (name == "csv") {
					cin >> name;
					for (size_t i = 0; i < tensors.size(); i++) {
						string file = OpenCL.location + name + tensors[i]->alias + ".csv";
						save_tensor_as_csv(tensors[i], file, I);
						if (i > 0)
							tensor_names += ", ";
						tensor_names += tensors[i]->alias;
					}
				}
				else {
					ofstream ofs(OpenCL.location + name, ostream::binary);
					for (size_t i = 0; i < tensors.size(); i++) {
						save_tensor(tensors[i], ofs, I);
						if (i > 0)
							tensor_names += ", ";
						tensor_names += tensors[i]->alias;
					}
					ofs.close();
				}
				logger << "[debugger] parameters " << tensor_names << " saved." << endl;
			}
			else if (command == "load") {
				cin >> name;
				string tensor_names;
				if (name == "csv") {
					string prefix;
					cin >> prefix;
					getline(cin, tensor_names);
					stringstream ss;
					ss << tensor_names;
					vector<string> names;
					while (ss) {
						name.clear();
						ss >> name;
						if (!name.empty())
							names.push_back(name);
					}
					tensor_names.clear();
					for (auto file : names) {
						file = OpenCL.location + prefix + file + ".csv";
						auto tensor = load_tensor_from_csv(file, I);
						if (!tensor_names.empty())
							tensor_names += ", ";
						tensor_names += tensor->alias;
					}
				}
				else {
					ifstream ifs(OpenCL.location + name, istream::binary);
					vector<Tensor*> tensors = load_tensors(ifs, I);
					ifs.close();
					for (size_t i = 0; i < tensors.size(); i++) {
						if (i > 0)
							tensor_names += ", ";
						tensor_names += tensors[i]->alias;
					}
				}
				logger << "[debugger] parameters " << tensor_names << " loaded." << endl;
			}
			else if (command == "quit") {
				logger << "[debugger] debugger thread terminated." << endl;
				debugger = nullptr;
				_breakpoint = nullptr;
				breakpoint.notify_all();
			}
			else if (command == "exit") {
				logger << "[debugger] program exited." << endl;
				exit(1);
			}
			else if (command == "?" || command == "help") {
				logger << "[debugger] commands:\n";
				logger << "goto(g)    continue(c)    pause(p)    exit    save    load    reload_kernel(rk)    profile(pf)    quit" << endl;
			}
			else {
				vector<int64> low, high, reshaped;
				string subprints;
				int which;
				if (command[0] == '^') {
					which = command[1] == '^'? 2 : 1;
					command = command.substr(which);
				}
				else
					which  = 0;

				if (command[0] != '[') { //target for inputs or peers
					auto pos = command.find(".");
					if (pos != string::npos) {
						name = command.substr(0, pos);
						last = locate_tensor(name);
						auto attrib = command.substr(pos + 1);
						vector<Tensor*> tensors;
						if (attrib.find("peers") == 0) {
							tensors = last->peers;
							pos += 7;
						}
						else if (attrib.find("inputs") == 0) {
							tensors = last->inputs;
							pos += 8;
						}
						else if (attrib == "learning_rate") {
							auto tensor = dynamic_cast<type::StochasticGradientDescentUpdater*>(last);
							float& target = tensor->learning_rate;
							if (tensor == nullptr)
								throw runtime_error(last->alias + " is not type of type::StochasticGradientDescentUpdater.");
							logger << "[debugger] " << last->alias << ".learning_rate = " << target << endl;
							char c= cin.peek();
							if (c != '\n') {
								cin >> name;
								float value;
								cin >> value;
								unitary_operation<float>(name, value)(target);
								logger << "[debugger] " << last->alias << ".learning_rate " << name << " " << value << endl;
								if (name != "=")
									logger << "[debugger] " << last->alias << ".learning_rate = " << target << endl;
							}
							continue;
						}
						else if (attrib == "max_epoch") {
							auto tensor = dynamic_cast<type::IterativeOptimizer*>(last);
							int64& target = reinterpret_cast<int64&>(tensor->max_epoch);
							if (tensor == nullptr)
								throw runtime_error(last->alias + " is not type of type::StochasticGradientDescentUpdater.");
							logger << "[debugger] " << last->alias << ".max_epoch = " << target << endl;
							char c= cin.peek();
							if (c != '\n') {
								cin >> name;
								int64 value;
								cin >> value;
								unitary_operation<int64>(name, value)(target);
								logger << "[debugger] " << last->alias << ".max_epoch " << name << " " << value << endl;
								if (name != "=")
									logger << "[debugger] " << last->alias << ".max_epoch = " << target << endl;
							}
							continue;
						}
						else if (attrib == "L") {
							auto tensor = dynamic_cast<back::Loss*>(last);
							if (tensor == nullptr)
								throw runtime_error(type_name(last) + " is not of back::Loss type.");
							auto L = tensor->L(I);
							describe_tensor(last);
							logger << "Loss value: " << L << endl;
							continue;
						}
						auto pos2 = command.find("]", pos) + 1;
						auto n = atoi(command.substr(pos, pos2 - pos).data());
						last = tensors[n];
						if (command.size() > pos2 && command[pos2] == '[')
							subprints = command.substr(pos2, command.size() - pos2);
					}
					else if ((pos = command.find("[")) != string::npos) {
						name = command.substr(0, pos);
						last = locate_tensor(name);
						subprints = command.substr(pos, command.size() - pos);
					}
					else {
						name = command;
						last = locate_tensor(name);
					}
				}
				else {
					if (last == nullptr)
						last = _current;
					subprints = command;
				}

				string type("float");
				size_t n = subprints.rfind(":");
				if (n != string::npos && subprints.back() != ']') {
					type = subprints.substr(n + 1);
					subprints = subprints.substr(0, n);
				}

				char c = cin.peek();
				if (c != '\n') {
					cin >> name;
					if (name[0] == '[') {
						parse_dimensions<int64>(name, &reshaped);
						parse_dimensions<int64>(subprints, &low, &high, &reshaped);
						c = cin.peek();
						if (c != '\n')
							cin >> name;
						else
							name.clear();
					}
				}
				else
					name.clear();
				if (reshaped.empty())
					parse_dimensions<int64>(subprints, &low, &high, &last->dimensions, &reshaped);

				string op = to_string(which) + name;
				if (name.empty()) {
					if (type == "float")
						operate_tensor_data<float>(last, low, high, reshaped, I, op);
					else if (type == "int")
						operate_tensor_data<int>(last, low, high, reshaped, I, op);
					else if (type == "char")
						operate_tensor_data<char>(last, low, high, reshaped, I, op);
					else if (type == "int64")
						operate_tensor_data<int64>(last, low, high, reshaped, I, op);
					else if (type == "short")
						operate_tensor_data<short>(last, low, high, reshaped, I, op);
					else if (type == "double")
						operate_tensor_data<double>(last, low, high, reshaped, I, op);
				}
				else {
					if (type == "float") {
						float value;
						cin >> value;
						operate_tensor_data<float>(last, low, high, reshaped, I, op, value);
					}
					else if (type == "int") {
						int value;
						cin >> value;
						operate_tensor_data<int>(last, low, high, reshaped, I, op, value);
					}
					else if (type == "char") {
						char value;
						cin >> value;
						operate_tensor_data<char>(last, low, high, reshaped, I, op, value);
					}
					else if (type == "int64") {
						int64 value;
						cin >> value;
						operate_tensor_data<int64>(last, low, high, reshaped, I, op, value);
					}
					else if (type == "short") {
						short value;
						cin >> value;
						operate_tensor_data<short>(last, low, high, reshaped, I, op, value);
					}
					else if (type == "double") {
						double value;
						cin >> value;
						operate_tensor_data<double>(last, low, high, reshaped, I, op, value);
					}
					logger << "updated." << endl;
				}
			}
		}
		catch (runtime_error& e) {
			logger << "[debugger] error: " << e.what() << endl;
		}
	}
}

void launch_debugger_thread(DeviceInstance& I, Tensor& graph)
{
	debugger = new thread(debugger_thread, ref(I), ref(graph));
}

template void parse_dimensions<int>(std::string subprints, std::vector<int>* low, std::vector<int>* high, const std::vector<int>* limits, std::vector<int>* reshaped);

}
