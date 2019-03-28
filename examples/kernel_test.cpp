/*
 * kernel_test.cpp
 *
 *  Created on: 2017/10/5
 *      Author: ZhangHua
 *  The full version of kernel_test for matrix multiplication including comparison with other libraries (including clBLAS, clblast, MIOpenGemm, cublas, MKL),
 *  can be found at https://github.com/mz24cn/gemm_optimization
 */

#include <iostream>

#include <tensor.hpp>
#include <device_instance.hpp>

using namespace std;
using namespace clnet;

T kernel_test()
{
	T initializer = GeneralInitializer({}, 0, 2.34f);
	int M = optional<int>("M", 2048); //dim_hidden
	int N = optional<int>("N", 512); //batch_size
	int K = optional<int>("K", 2048); //dim_in
	int STEP = optional<int>("step", 4);
	bool parallel = optional<int>("parallel", false);
	T w = Weight({M, K}, "w", &initializer);
	T x = Data({N, K}, &initializer, "x");
	T result = Data({M, N}, nullptr, "gemm");

	T graph = *new InstantTensor("kernel_test", {&x, &w},
		[M, N, K, STEP, parallel, &result, &initializer](InstantTensor* self, DeviceInstance& I) {
		auto& kernel = prepare_for_running_kernel(self, I);
		T x = *self->inputs[0];
		T w = *self->inputs[1];
		kernel.setArg(0, I.buffers[&result]);
		kernel.setArg(1, I.buffers[&x]);
		kernel.setArg(2, I.buffers[&w]);
		kernel.setArg(3, nullptr);
//		kernel.setArg(3, I.buffers[&w]);
//		kernel.setArg(4, I.buffers[&x]);
//		kernel.setArg(5, I.buffers[&result]);

		int i = 0;
		for (int m = M; m >= 32; m /= STEP)
			for (int n = N; n >= 8; n /= STEP)
				for (int  k = K; k >= 32; k /= STEP) {
					int64 total = 1LL * M * N * K / m / n / k;
					size_t time = MICROS(0);
					for (int j = 0; j < total; j++) {
						kernel.setArg(4, m);
						kernel.setArg(5, k);
//						kernel.setArg(0, m);
//						kernel.setArg(1, n);
//						kernel.setArg(2, k);
						cl::NDRange global(m * n);
						I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[&result]);
						if (!parallel)
							wait_for_all_kernels_finished(I);
					}
					if (parallel)
						wait_for_all_kernels_finished(I);
					time = MICROS(time);
					float baseline = time / total / 1000.0f;
//					result.upload(I);
//					memcpy(result.pointer, I.pointers[&result], m * n * sizeof(float));
//					operate_tensor_data<float>(&result, I, {0, 0}, {1, 9}, result.dimensions, "2");

					//Run second version:
					size_t time2 = MICROS(0);
					for (int j = 0; j < total; j++)
						self->peers[i]->run(I);
					if (parallel)
						wait_for_all_kernels_finished(I);
					time2 = MICROS(time2);
					float millis = time2 / total / 1000.0f;
					float delta = 0;
//					result.upload(I);
//					operate_tensor_data<float>(&result, I, {0, 0}, {1, 9}, result.dimensions, "1");
//					auto p1 = result.pointer, p2 = I.pointers[&result];
//					for (int j = 0; j < m * n; j ++, p1++, p2++) {
//						float diff = *p1 - *p2;
//						delta += diff * diff;
//					}
//					delta /= m * n;

//					string str = "M=" + to_string(m) + " \tN=" + to_string(n) + " \tK=" + to_string(k);
					logger << "M=" << m << " \tN=" << n << " \tK=" << k << " \ttimes=" << total << ":\n";
					logger << "\tBaselineTime=" << time / 1000.0f << "ms, " << baseline << "ms";
					logger << " \tNewTime=" << time2 / 1000.0f << "ms, "  << millis << "ms" << " \tratio=" << millis / baseline << " \tdelta=" << delta << endl;
					i++;
				}
	}, {},
	[](InstantTensor* self) -> Tensor* { return nullptr; },
	[](InstantTensor* self) -> std::vector<Tensor*> { return self->peers; },
	[](InstantTensor* self, DeviceInstance& I) -> string{ //This example used as a trial to test OpenCL kernel code
		return string(R"CLC(
kernel void gemm(global float* out, const global float* in, const global float* weight, const global float* bias, 
		/*local float* tmp, */const int dim_hidden, const int dim_in)
{
	const int GID = get_global_id(0);
	const int n = GID / dim_hidden;
	const int hidden = GID % dim_hidden;
	const int weight_offset = hidden * dim_in;
	const int in_offset = n * dim_in;
	float z = bias != NULL? bias[hidden] : 0;

	for (int i = 0; i < dim_in; i++)
		z += weight[weight_offset + i] * in[in_offset + i];
	out[GID] = z;
}
)CLC");
	});

	for (int m = M; m >= 32; m /= STEP)
		for (int n = N; n >= 8; n /= STEP)
			for (int  k = K; k >= 32; k /= STEP)
				graph.peers.push_back(new InstantTensor("tiling_" + to_string(m)  + "_" + to_string(n) + "_" + to_string(k), {&x, &w}, {}, [m, n, k, parallel, &result](InstantTensor* self, DeviceInstance& I) {
					auto& kernel = prepare_for_running_kernel(self, I);
					T x = *self->inputs[0];
					T w = *self->inputs[1];
					kernel.setArg(0, I.buffers[&result]);
					kernel.setArg(1, I.buffers[&x]);
					kernel.setArg(2, I.buffers[&w]);
					kernel.setArg(3, nullptr);
					kernel.setArg(4, m);
					int local_size = find_proper_local_size(k, I.work_group_size);
					if (local_size > m * n)
						local_size = m * n;
					cl::NDRange global(m * n);
					cl::NDRange local(local_size);
					I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local/*cl::NullRange*/, &I.precondition_events, &I.events[&result]);
//					operate_tensor_data<float>(&result, I, {0, 0}, {2, 4}, result.dimensions);
					if (!parallel)
						wait_for_all_kernels_finished(I);
				}, [m, n, k](InstantTensor* self, DeviceInstance& I) -> string {
					auto code = string(R"CLC(
kernel void gemm_tiling_dim_in(global float* out, const global float* in, const global float* weight, const global float* bias,
		/*local float* tmp, */const int dim_hidden)
{
	const int GID = get_global_id(0);
	const int N = get_global_size(0) / dim_hidden;
	const int n = GID / dim_hidden;
	const int hidden = GID % dim_hidden;
	const int weight_offset = hidden * dim_in;
	const int in_offset = n * dim_in;
	float z = bias != NULL? bias[hidden] : 0;

	local float weight_tile[dim_in];
	local float in_tile[dim_in];
	for (int i = get_local_id(0); i < dim_in; i += get_local_size(0)) {
		weight_tile[i] = weight[weight_offset + i];
		in_tile[i] = in[in_offset + i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

//	// Loop over all tiles
//	const int numTiles = K / TS;
//	for (int t = 0; t < numTiles; t++) {
//		// Load one tile into local memory
//		const int tiledRow = TS * t + row;
//		const int tiledCol = TS * t + col;
//		weight_tile[col][row] = weight[tiledCol * M + globalRow];
//		in_tile[col][row] = B[globalCol * K + tiledRow];
//
//		// Synchronise to make sure the tile is loaded
//		barrier(CLK_LOCAL_MEM_FENCE);
//
//		// Perform the computation for a single tile
//		for (int k=0; k<TS; k++) {
//			acc += Asub[k][row] * Bsub[col][k];
//		}
//
//		// Synchronise before loading the next tile
//		barrier(CLK_LOCAL_MEM_FENCE);
//	}

//#pragma unroll
	for (int i = 0; i < dim_in; i++)
		z += weight_tile[i] * in_tile[i];
	out[GID] = z;
}
)CLC");
					replace_all(code, "dim_in", to_string(k));
//					auto code = string(R"CLC(
//kernel void gemm_unrolled_dim_in(global float* out, const global float* in, const global float* weight, const global float* bias,
//		/*local float* tmp, */const int dim_hidden)
//{
//	const int GID = get_global_id(0);
//	const int n = GID / dim_hidden;
//	const int hidden = GID % dim_hidden;
//	const int weight_offset = hidden * dim_in;
//	const int in_offset = n * dim_in;
//	float z = bias != NULL? bias[hidden] : 0;
//
//#pragma unroll
//	for (int i = 0; i < dim_in; i++)
//		z += weight[weight_offset + i] * in[in_offset + i];
//	out[GID] = z;
//}
//)CLC");
//					replace_all(code, "dim_in", to_string(k));
					return code;
				}));

	return graph;
}
