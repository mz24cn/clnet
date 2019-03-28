/*
 * device_instance.hpp
 *
 *  Created on: 2017/5/13
 *      Author: ZhangHua
 */

#ifndef INCLUDE_DEVICE_INSTANCE_HPP_
#define INCLUDE_DEVICE_INSTANCE_HPP_

#include <atomic>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <sstream>

#define USER_ERROR_DESCRIPTION_UNDEFINED -28
#define USER_GROUP_SIZE_NOT_BIG_ENOUGH -29
#define MIN_ERROR_CODE -70
#define MILLIS(time) (MICROS() / 1000 - time)
#define ALIGN(total,size) ((total + size - 1) / size * size)

namespace clnet {
extern const char* clErrorCodeDescriptions[];
extern std::string cl_build_options;
struct Tensor;

struct DeviceInstance {
	int ID;
	cl::Device device;

	std::atomic<int> parameters_state;
	std::atomic<int> gradients_state;
	std::vector<cl::Event> precondition_events;

	int work_group_size;
	int local_memory_size;
	cl::CommandQueue queue;
	std::unordered_map<Tensor*, cl::Buffer> buffers;
	std::unordered_map<Tensor*, cl::Event> events;
	std::unordered_map<Tensor*, float*> pointers;
	std::unordered_map<Tensor*, std::vector<cl::Kernel>> kernels;

	explicit DeviceInstance() : ID(-1), parameters_state(0), gradients_state(0), work_group_size(0), local_memory_size(0) {}
	void initialize();
	void free();

	static DeviceInstance& create(cl::Device& cl_device, int id);
	static std::unordered_map<int, DeviceInstance> ALL;
};

//Reference: http://stackoverflow.com/questions/24465533/implementing-boostbarrier-in-c11
class thread_barrier
{
private:
	std::mutex mutex_;
	std::condition_variable cv;
	size_t count, times, threads_num;
public:
	explicit thread_barrier(size_t threads) : count(threads), times(0), threads_num(threads) { }
	void wait() {
		size_t current = times;
		std::unique_lock<std::mutex> lock{mutex_};
		if (--count == 0) {
			times++;
			count = threads_num;
			cv.notify_all();
		}
		else
			cv.wait(lock, [this, current] { return current != times; });
	}
};

struct AssemblingEvent {
	std::atomic<int> counter;
	cl::UserEvent* event;
	AssemblingEvent(int count, cl::UserEvent* pEvent) : counter(count), event(pEvent) {}
};

#define MAX_LOGGER_STREAMS 8
class Logger { //thread 'atomic' logger
	std::ostream* streams[MAX_LOGGER_STREAMS];
	int count;
	std::mutex safe_access;
	std::unordered_map<std::thread::id, std::stringstream> buffers;
	std::stringstream& thread_buffer();

public:
	Logger();
	Logger& operator +=(std::string filename);
	Logger& operator +=(std::ostream& os);

	template <typename T> Logger& operator <<(const T& content)
	{
		thread_buffer() << content;
		return *this;
	}
	Logger& operator <<(std::ostream& (*fp)(std::ostream&));
};
extern Logger logger;

class OpenCL_ {
	std::vector<cl::Device>* devices = nullptr;

public:
	cl_device_type device_type = CL_DEVICE_TYPE_GPU;
	std::string location;

	std::vector<cl::Device>& find_devices();
	void run(Tensor& graph, std::vector<int> targetDeviceIDs = {}, int debugger_device_id = -1, int master_device_id = -1);
	void deallocate_all_tensors();
	void print_device_info(std::ostream& out);
	void print_tensor_structure(Tensor& graph, bool onlyShowOperator = false);
	void print_tensor_memory();
	void print_parameters(Tensor& graph);
};
extern OpenCL_ OpenCL;

void reload_kernels(const cl::Device& device, const cl::Context& context, DeviceInstance& I);
std::string generate_kernel_sources(DeviceInstance& I, const cl::Device& device, std::unordered_map<Tensor*, std::string>& tensor_kernels);
cl::Kernel& prepare_for_running_kernel(Tensor* tensor, DeviceInstance& I, int number = 0);
void wait_for_all_kernels_finished(DeviceInstance& I);
int find_proper_local_size(int required, int work_group_size);

void CL_CALLBACK assembling_event_callback(cl_event, cl_int, void * user_data);
void CL_CALLBACK gradients_event_callback(cl_event, cl_int, void * user_data);
void CL_CALLBACK parameters_event_callback(cl_event, cl_int, void * user_data);
void launch_debugger_thread(DeviceInstance& I, Tensor& graph);
size_t MICROS(size_t microseconds = 0);
std::string millis_string(size_t time);

template <typename T> void parse_dimensions(std::string subprints, std::vector<T>* low, std::vector<T>* high = nullptr, const std::vector<T>* limits = nullptr, std::vector<T>* reshaped = nullptr);
template <typename T> void operate_tensor_data(Tensor* tensor, DeviceInstance& I, const std::vector<int64> low = {}, const std::vector<int64> high = {}, const std::vector<int64> reshaped = {}, std::string op = "0", T value = (T)0);

template <typename T> T optional(std::string name, T default_value);
template <typename T> T optional(std::unordered_map<std::string, std::string>& map, std::string name, T default_value);
template <typename T> bool read_file_content(const std::string file, std::basic_string<T>& content);
void describe_tensor(Tensor* tensor, bool only_name = true);
std::string formatWithComma(size_t num);

#define replace_once(str, key, value) str.replace(str.find(key), sizeof(key) - 1, value)
void replace_all(std::string& content, const std::string key, const std::string replace);
}
#endif /* INCLUDE_DEVICE_INSTANCE_HPP_ */
