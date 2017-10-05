/*
 * kernel_test.cpp
 *
 *  Created on: 2017Äê10ÔÂ5ÈÕ
 *      Author: ZhangHua
 */

#include <iostream>

#include <tensor.hpp>
#include <device_instance.hpp>

using namespace std;
using namespace clnet;

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
