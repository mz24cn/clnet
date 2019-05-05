#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics: enable
#ifdef cl_khr_int64_base_atomics
	#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
	#define atom_type ulong
#else
	#define atom_type uint
#endif
#ifdef cl_amd_printf
//	#pragma OPENCL EXTENSION cl_amd_printf: enable
#endif
#ifndef work_group_barrier
	#define work_group_barrier barrier
#endif
#ifndef NULL
	#define NULL 0
#endif
#define gradient_set_type = 
#define _gradient(z) 1.0f

inline float sigmoid(float z)
{
	return 1.0f / (1.0f + exp(-z));
}
inline float sigmoid_gradient(float y)
{
	return y * (1.0f - y);
}

inline float relu(float z)
{
	return z > 0? z : 0;
}
inline float relu_gradient(float y)
{
	return y > 0? 1 : 0;
}

//tanh is predefined
inline float tanh_gradient(float y)
{
	return 1.0f - y * y;
}

inline float softrelu(float z)
{
	return log(1.0f + exp(z));
}
inline float softrelu_gradient(float y)
{
	return 1.0f - exp(-y);
}

inline float leakyrelu(float z)
{
	return z > 0? z : 0.25f * z;
}
inline float leakyrelu_gradient(float y)
{
	return y > 0? 1 : 0.25f;
}

inline float linear_regression(float y, float label)
{
	float delta = y - label;
	return delta * delta;
}
inline float linear_regression_gradient(float y, float label)
{
	return y - label;
}

//softmax is calculated by pow(z) / sum of pow
inline float negative_log_likelihood_gradient(float y, bool i_equal_j)
{
	return i_equal_j? y - 1.0f : y;
}

//Standard implementation
//Parallel: (batch_size * dim_hidden)
kernel void feed_forward_fully_connected_sigmoid(global float* out, const global float* in, const global float* weight, const global float* bias, 
		local float* tmp, const int dim_hidden, const int dim_in)
{
	const int GID = get_global_id(0);
	const int n = GID / dim_hidden;
	const int hidden = GID % dim_hidden;
	const int in_offset = n * dim_in;
	float z = bias != NULL? bias[hidden] : 0;

#pragma unroll
	for (int i = 0; i < dim_in; i++)
		z += weight[dim_hidden * i + hidden] * in[in_offset + i];
	out[GID] = sigmoid(z);
}

//Standard implementation
//Parallel: max(weight_Out_dim, batch_size) x weight_In_dim
kernel void back_propagate_fully_connected_softrelu_gradient(global float* in_grad, global float* weight_grad,
		global float* bias_grad,	const global float* weight, const global float* in, const global float* out,
		const global float* out_grad, const int dim_out, const int dim_in, const int batch_size)
{
	const int GID = get_global_id(0);
	const int k = GID % dim_in;
	const int n = GID / dim_in;

	if (n < dim_out) {
		float sum_weight_grad = 0, sum_bias_grad = 0;
		for (int j = 0; j < batch_size; j++) {
			const float in_j = in[j * dim_in + k];
			const float out_grad_j = softrelu_gradient(out[j * dim_out + n]) * out_grad[j * dim_out + n];
			sum_bias_grad += out_grad_j;
			sum_weight_grad += in_j * out_grad_j;
		}
		if (k == 0 && bias_grad != NULL)
			bias_grad[n] += sum_bias_grad;
		weight_grad[k * dim_out + n] += sum_weight_grad;
	}

	if (in_grad != NULL && n < batch_size) {
		float sum_in_grad = 0;
		for (int j = 0; j < dim_out; j++) {
			const float weight_j = weight[k * dim_out + j];
			const float out_grad_j = softrelu_gradient(out[n * dim_out + j]) * out_grad[n * dim_out + j];
			sum_in_grad += weight_j * out_grad_j;
		}
		in_grad[n * dim_in + k] gradient_set_type sum_in_grad;
	}
}

//Standard implementation
//Parallel: weight_Out_dim
kernel void back_propagate_fully_connected_softrelu_gradient_for_bias(global float* activation_grad, global float* bias_grad,
		const global float* out, const global float* out_grad, const int dim_out, const int dim_in, const int batch_size)
{
	const int n = get_global_id(0);

	float sum_bias_grad = 0;
	for (int j = 0; j < batch_size; j++) {
		const float out_grad_j = softrelu_gradient(out[j * dim_out + n]) * out_grad[j * dim_out + n];
		activation_grad[j * dim_out + n] = out_grad_j; 
		sum_bias_grad += out_grad_j;
	}
	if (bias_grad != NULL)
		bias_grad[n] += sum_bias_grad;
}

//Standard implementation
//Parallel: weight_Out_dim * weight_In_dim
kernel void back_propagate_fully_connected_softrelu_gradient_for_weight(global float* weight_grad, const global float* activation_grad,
		const global float* in, const int dim_out, const int dim_in, const int batch_size)
{
	const int n = get_global_id(0);
	const int pos = get_global_id(1);
	const int K = get_global_size(1);

	for (int k = pos; k < dim_in; k += K) {
		float sum_weight_grad = 0;
		for (int j = 0; j < batch_size; j++) {
			const float in_j = in[j * dim_in + k];
			const float out_grad_j = activation_grad[j * dim_out + n];
			sum_weight_grad += in_j * out_grad_j;
		}
		weight_grad[k * dim_out + n] += sum_weight_grad;
	}
}

//Standard implementation
//Parallel: batch_size x weight_In_dim
kernel void back_propagate_fully_connected_softrelu_gradient_for_data(global float* in_grad, const global float* weight,
		const global float* out, const global float* out_grad, const int dim_out, const int dim_in, const int batch_size)
{
	const int GID = get_global_id(0);
	const int k = GID % dim_in;
	const int n = GID / dim_in;

	float sum_in_grad = 0;
	for (int j = 0; j < dim_out; j++) {
		const float weight_j = weight[k * dim_out + j];
		const float out_grad_j = softrelu_gradient(out[n * dim_out + j]) * out_grad[n * dim_out + j];
		sum_in_grad += weight_j * out_grad_j;
	}
	in_grad[n * dim_in + k] gradient_set_type sum_in_grad;
}

//Parallel: (batch_size * dim_hidden * get_local_size(0))
//Note: choose local NDRange size near (2 * dim_in) when enqueue ASAP
kernel void feed_forward_fully_connected_softrelu(global float* out, const global float* in, const global float* weight, const global float* bias, 
		local float* tmp, const int dim_hidden, const int dim_in)
{
	const int GID = get_global_id(0);
// parallel for reduction: should be power of 2
// Use max parallel get_local_size(0) because OpenCL only synchorizes in a work group 
	const int parallel = get_local_size(0);
	const int n = GID / parallel / dim_hidden;
	const int hidden = (GID / parallel) % dim_hidden;
//	const int weight_offset = hidden * dim_in;
	const int in_offset = n * dim_in;
	float z = bias != NULL? bias[hidden] : 0;

// parallel addition, trade space for time
	const int pos = GID % parallel;
	float sum = 0;
// support any value for dim_in. Inefficient for dim_in < parallel / 2
	for (int index = pos; index < dim_in; index += parallel)
		sum += weight[dim_hidden * index + hidden] * in[in_offset + index];

	tmp[pos] = sum;
	work_group_barrier(CLK_LOCAL_MEM_FENCE);
	for (int stride = parallel / 2; stride > 0; stride = stride / 2) {
		if (pos < stride)
			tmp[pos] += tmp[pos + stride];
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (pos == 0)
		out[GID / parallel] = softrelu(z + tmp[0]);
}

//Parallel: (batch_size * get_local_size(0))
kernel void negative_log_likelihood_loss(global float* out_grad, global float* out, const global float* in, 
		const global float* label, local float* tmp, const int dim_in, const int batch_size)
{
	const int GID = get_global_id(0);
	const int parallel = get_local_size(0); //no more than dim_in
	const int n = GID / parallel;
	const int pos = GID % parallel;

	float max_value = in[n * dim_in];
	for (int index = 1; index < dim_in; index++)
		max_value = max(max_value, in[n * dim_in + index]);

	float sum = 0;
	for (int index = pos; index < dim_in; index += parallel) {
		const int k = n * dim_in + index;
		out[k] = exp(in[k] - max_value);
		sum += out[k];
	}

	tmp[pos] = sum;
	work_group_barrier(CLK_LOCAL_MEM_FENCE);
	for (int stride = parallel / 2; stride > 0; stride = stride / 2) {
		if (pos < stride)
			tmp[pos] += tmp[pos + stride];
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
	}
	sum = tmp[0];

	for (int index = pos; index < dim_in; index += parallel) {
		const int i = ((int) label[n]), k = n * dim_in + index;
		out[k] /= sum;
		out_grad[k] = negative_log_likelihood_gradient(out[k], index == i) / batch_size;
	}
}

//Parallel: (batch_size * get_local_size(0))
kernel void feed_forward_softmax(global float* out, const global float* in, local float* tmp, const int dim_in, const int batch_size)
{
	const int GID = get_global_id(0);
	const int parallel = get_local_size(0); //no more than dim_in
	const int n = GID / parallel;
	const int pos = GID % parallel;

	float max_value = in[n * dim_in];
	for (int index = 1; index < dim_in; index++)
		max_value = max(max_value, in[n * dim_in + index]);

	float sum = 0;
	for (int index = pos; index < dim_in; index += parallel) {
		const int k = n * dim_in + index;
		out[k] = exp(in[k] - max_value);
		sum += out[k];
	}

	tmp[pos] = sum;
	work_group_barrier(CLK_LOCAL_MEM_FENCE);
	for (int stride = parallel / 2; stride > 0; stride = stride / 2) {
		if (pos < stride)
			tmp[pos] += tmp[pos + stride];
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
	}
	sum = tmp[0];

	for (int index = pos; index < dim_in; index += parallel) {
		const int k = n * dim_in + index;
		out[k] /= sum;
	}
}

//Parallel: (batch_size * get_local_size(0))
kernel void linear_regression_loss(global float* out_grad, const global float* out, const global float* label)
{
	const int GID = get_global_id(0);
	out_grad[GID] = linear_regression_gradient(out[GID], label[GID]);
}

//Parallel: params->dims
kernel void update_parameters_by_stochastic_gradient_descent(global float* params, global float* params_grad,
		float learning_rate, float weight_decay)
{
	const int GID = get_global_id(0);
	params[GID] -= learning_rate * (params_grad[GID] + weight_decay * params[GID]);
	params_grad[GID] = 0;
}

//Parallel: params->dims
kernel void update_parameters_by_stochastic_gradient_descent_with_momentum(global float* params, global float* params_grad,
		float learning_rate, float weight_decay, float momentum, global float* velocity)
{
	const int GID = get_global_id(0);
	const float gradient = params_grad[GID] + weight_decay * params[GID];
	const float vt = momentum * velocity[GID] + gradient;
	params[GID] -= learning_rate * vt;
	velocity[GID] = vt;
	params_grad[GID] = 0;
}

//Parallel: (h_batch_size * h_dim_hidden)
kernel void feed_forward_LSTM_recurrent(global float* x_timestep, global float* out, const global float* x, const global float* hidden,
		const int timestep, const int sequence_length, const int dim_input, const int dim_hidden)
{
	const int GID = get_global_id(0);
	const int batch = GID / dim_hidden;
	const int j = GID % dim_hidden;
	const int offset = batch * sequence_length + abs(timestep);
	// fetch input timestep from batch-major data
	if (timestep >= 0) { //exclude the last call: need not generate x_timestep
		const int m = batch * dim_input, n = offset * dim_input;
		for (int index = j; index < dim_input; index += dim_hidden)
			x_timestep[m + index] = x[n + index]; //collect timestep from batch-major data
	}

	//save hidden result to out
	if (out != NULL) { //exclude the first call: no hidden output at this time
		const int k = (offset - 1) * dim_hidden + j;
		out[k] = hidden[GID];
	}
}

//Parallel: (h_batch_size * h_dim_hidden)
kernel void back_propagate_LSTM_recurrent(global float* hidden_grad, global float* x_grad, global float* x_timestep, const global float* out_grad, 
		const global float* x_timestep_grad, const global float* x, const int timestep, const int sequence_length, const int dim_input, const int dim_hidden)
{
	const int GID = get_global_id(0);
	const int batch = GID / dim_hidden;
	const int j = GID % dim_hidden;
	const int offset = batch * sequence_length + abs(timestep);
	//save hidden result from out_grad
	if (out_grad != NULL) { //exclude the first call: no hidden output at this time
		const int k = (offset - 1) * dim_hidden + j;
		hidden_grad[GID] += out_grad[k]; //add on back-propagation-through-time gradient
	}

	// put input gradient as batch-major data
	if (timestep > 0) { //exclude the last call: need not generate x_timestep_grad
		const int m = batch * dim_input, n = offset * dim_input;
		for (int index = j; index < dim_input; index += dim_hidden) {
			const int i = m + index, k = n + index;
			x_timestep[i] = x[k - dim_input]; //recover input
			x_grad[k] = x_timestep_grad[i]; //restore batch-major data from timestep data
		}
	}
	else if (timestep == 0) { //x_timestep is ready in the first time, and need not to be prepared in the last time, so both ignored)
		const int m = batch * dim_input, n = offset * dim_input;
		for (int index = j; index < dim_input; index += dim_hidden)
			x_grad[n + index] = x_timestep_grad[m + index]; //restore batch-major data from timestep data
	}
}

//Parallel: (batch_size * dim_hidden)
kernel void feed_forward_LSTM_cell(global float* C, global float* h, global float* gates_data/*cell_no_max * batch_size * 5*dim_hidden*/,
		const global float* z/*batch_size * 4*dim_hidden*/, local float* tmp, const int dim_hidden, int cell_no)
{
	const int GID = get_global_id(0);
	const int batch = GID / dim_hidden;
	const int i_g = batch * 4 * dim_hidden + (GID % dim_hidden);
	const int i_t = i_g + dim_hidden;
	const int f_g = i_t + dim_hidden;
	const int o_g = f_g + dim_hidden;
	const float in_gate= sigmoid(z[i_g]);
	const float C_candicate = tanh(z[i_t]);
	const float forget_gate = sigmoid(z[f_g]);
	const float out_gate = sigmoid(z[o_g]);
	const float C_prev = C[GID]; //initialized as zero for first timestamp
	const float C_t = forget_gate * C_prev + in_gate * C_candicate;
	const float tanh_C_t = tanh(C_t);

	if (gates_data != NULL) {
		global float* data = gates_data + cell_no * get_global_size(0) * 7;

		const float C_grad = out_gate * tanh_gradient(tanh_C_t);
		data[i_g] = C_candicate * sigmoid_gradient(in_gate);
		data[i_t] = in_gate * tanh_gradient(C_candicate);
		data[f_g] = C_prev * sigmoid_gradient(forget_gate);
		data[o_g] = tanh_C_t * sigmoid_gradient(out_gate);

		const int p = 4 * get_global_size(0) + GID;
		const int c_g = p + get_global_size(0);
		const int c_m = c_g + get_global_size(0);
		data[p] = h[GID]; //h_prev: initialized as zero for first timestamp
		data[c_g] = C_grad;
		data[c_m] = forget_gate;
	}
	C[GID] = C_t;
	h[GID] = out_gate * tanh_C_t;
}

//Parallel: (batch_size * dim_hidden)
kernel void back_propagate_LSTM_cell_gates(global float* z_grad, global float* h_prev, global float* cell_state_grad, const global float* h_grad,
		const global float* gates_data, local float* tmp, const int dim_hidden, const int batch_size, const int cell_no)
{
	const int GID = get_global_id(0);
	const int batch = GID / dim_hidden;
	const int i_g = batch * 4 * dim_hidden + (GID % dim_hidden);
	const int i_t = i_g + dim_hidden;
	const int f_g = i_t + dim_hidden;
	const int o_g = f_g + dim_hidden;
	const int p = 4 * get_global_size(0) + GID;
	const int c_g = p + get_global_size(0);
	const int c_m = c_g + get_global_size(0);

	const global float* data = gates_data + cell_no * get_global_size(0) * 7;
	const float h_grad_batch_one = h_grad[GID];
	const float C_grad = data[c_g];
	const float forget_gate = data[c_m];
	const float cell_grad = h_grad_batch_one * C_grad + cell_state_grad[GID];

	z_grad[i_g] = cell_grad * data[i_g];
	z_grad[i_t] = cell_grad * data[i_t];
	z_grad[f_g] = cell_grad * data[f_g];
	z_grad[o_g] = h_grad_batch_one * data[o_g];
	h_prev[GID] = data[p];
	cell_state_grad[GID] = cell_grad * forget_gate;
}

//Parallel: (dim_hidden * get_local_size(0))
kernel void back_propagate_fully_connected_LSTM_cell(global float* weight_h_grad, global float* weight_x_grad, global float* bias_grad,
		global float* h_grad, global float* x_grad, const global float* gates_data,const global float* x, const global float* weight_h,
		const global float* weight_x, local float* tmp, const int dim_input, const int dim_hidden, const int batch_size, const int cell_no)
{
	const int GID = get_global_id(0);
	const int parallel = get_local_size(0);
	const int j = GID / parallel;
	const int K = GID % parallel;
	const int i_g = j; //0 <= i_g <= dim_hidden
	const int i_t = i_g + dim_hidden;
	const int f_g = i_t + dim_hidden;
	const int o_g = f_g + dim_hidden;
	gates_data += cell_no * dim_hidden * batch_size * 7;

	const global float* data = gates_data;
	for (int k = K; k < dim_hidden; k += parallel) //folad for hidden input vector size (dim_hidden)
		for (int n = 0; n < batch_size; n++) {
			const float h_prev = data[4 * dim_hidden + k];
			const float h_grad_batch_one = h_grad[n * dim_hidden + j] / batch_size;
			weight_h_grad[i_g * dim_hidden + k] += h_grad_batch_one * data[i_g] * h_prev;
			weight_h_grad[i_t * dim_hidden + k] += h_grad_batch_one * data[i_t] * h_prev;
			weight_h_grad[f_g * dim_hidden + k] += h_grad_batch_one * data[f_g] * h_prev;
			weight_h_grad[o_g * dim_hidden + k] += h_grad_batch_one * data[o_g] * h_prev;
			data += dim_hidden * 5;
		}

	data = gates_data;
	for (int k = K; k < dim_input; k += parallel)
		for (int n = 0; n < batch_size; n++) {
			const float in = x[n * dim_input + k];
			const float h_grad_batch_one = h_grad[n * dim_hidden + j] / batch_size;
			weight_x_grad[i_g * dim_input + k] += h_grad_batch_one * data[i_g] * in;
			weight_x_grad[i_t * dim_input + k] += h_grad_batch_one * data[i_t] * in;
			weight_x_grad[f_g * dim_input + k] += h_grad_batch_one * data[f_g] * in;
			weight_x_grad[o_g * dim_input + k] += h_grad_batch_one * data[o_g] * in;
			data += dim_hidden * 5;
		}

	data = gates_data;
	if (K == 0)
		for (int n = 0; n < batch_size; n++) {
			const float h_grad_batch_one = h_grad[n * dim_hidden + j] / batch_size;
			bias_grad[i_g] += h_grad_batch_one * data[i_g];
			bias_grad[i_t] += h_grad_batch_one * data[i_t];
			bias_grad[f_g] += h_grad_batch_one * data[f_g];
			bias_grad[o_g] += h_grad_batch_one * data[o_g];

			for (int k = 0; k < dim_hidden; k++) //TODO: wait for parallelizing on local space: parallel reduce for sum
				h_grad[n * dim_hidden + j] += h_grad_batch_one * (data[k] + data[k + dim_hidden] + data[k + 2 * dim_hidden] + data[k + 3 * dim_hidden]) * weight_h[k * dim_hidden + j];
			data += dim_hidden * 5;
		}

	data = gates_data;
	for (int k = K; k < dim_input; k += parallel)
		for (int n = 0; n < batch_size; n++) {
			const float weight = weight_x[n * dim_input + k];
			const float h_grad_batch_one = h_grad[n * dim_hidden + j] / batch_size;
			x_grad[i_g * dim_input + k] = h_grad_batch_one * data[i_g] * weight_x[i_g * dim_input + k];
			x_grad[i_t * dim_input + k] = h_grad_batch_one * data[i_t] * weight_x[i_t * dim_input + k];
			x_grad[f_g * dim_input + k] = h_grad_batch_one * data[f_g] * weight_x[f_g * dim_input + k];
			x_grad[o_g * dim_input + k] = h_grad_batch_one * data[o_g] * weight_x[o_g * dim_input + k];
			data += dim_hidden * 5;
		}
}

kernel void parallel_add(global float* z, const global float* a, const global float* b)
{
	const int GID = get_global_id(0);
	z[GID] = a[GID] + b[GID];
}

kernel void parallel_plus(global float* a, const global float* b)
{
	const int GID = get_global_id(0);
	a[GID] += b[GID];
}

//Parallel: (batch_size * dim_hidden * get_local_size(0))
kernel void parallel_multiply(global float* out, const global float* weight, const global float* in, local float* tmp, const int dim_hidden, const int dim_in)
{
	const int GID = get_global_id(0);
	const int parallel = get_local_size(0);
	const int k = GID / parallel;
	const int n = k / dim_hidden;
	const int hidden = k % dim_hidden;
	const int weight_offset = hidden * dim_in;
	const int in_offset = n * dim_in;
	float z = 0;

	const int pos = GID % parallel;
	float sum = 0;
	for (int index = pos; index < dim_in; index += parallel)
		sum += weight[weight_offset + index] * in[in_offset + index];

	tmp[pos] = sum;
	work_group_barrier(CLK_LOCAL_MEM_FENCE);
	for (int stride = parallel / 2; stride > 0; stride = stride / 2) {
		if (pos < stride)
			tmp[pos] += tmp[pos + stride];
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (pos == 0)
		out[k] = softrelu(z + tmp[0]);
}

//Parallel: ((batch_size * out_depth) * out_height * out_width)
kernel void feed_forward_convolution_activation_relu_tiling(global float* out, const global float* weight/*out_depth * kernel_height * kernel_width * in_depth*/, const global float* bias,
		const global float* in/*batch_size * in_height * in_width * in_depth*/, const int in_height, const int in_width, const int in_depth,
		const int kernel_height, const int kernel_width, const int stride_height, const int stride_width,
		const int padding_height, const int padding_width, const int batch_size, local float* weight_local, local float* in_local)
{
	const int out_height = get_global_size(1);
	const int out_width = get_global_size(2);
	const int out_depth = get_global_size(0) / batch_size;
	const int n = get_global_id(0) / out_depth; //batch
	const int rout = get_global_id(1);
	const int cout = get_global_id(2);
	const int filter = get_global_id(0) % out_depth;
	const int offset = ((n * out_height + rout) * out_width + cout) * out_depth + filter;

	//Tiling on cube: (filter_local * group_width * in_width_local)
	const int filter_local = get_local_id(0);
	const int in_height_local = get_local_size(1);
	const int in_width_local = get_local_size(2);

	const int parallel_plane = in_height_local * in_width_local;
	const int pos_plane = get_local_id(1) * in_width_local + get_local_id(2);
	const int weight_filter_size = kernel_height * kernel_width * in_depth;
	for (int i = pos_plane; i < weight_filter_size; i += parallel_plane) {
		const int weight_depth = i % in_depth;
		const int weight_column = (i / in_depth) % kernel_width;
		const int weight_row = (i / in_depth / kernel_width) % kernel_height;
		const int weight_global_index = ((filter * kernel_height + weight_row) * kernel_width + weight_column) * in_depth + weight_depth;
		weight_local[filter_local * weight_filter_size + i] = weight[weight_global_index];
	}

	const int filter_size_local = get_local_size(0);
	const int rmin = rout * stride_height, rlocal_max = in_height_local * stride_height + 2 * padding_height;
	const int cmin = cout * stride_width, clocal_max = in_width_local * stride_width + 2 * padding_width;
	for (int depth = filter_local; depth < in_depth; depth += filter_size_local) {
		for (int rin = rmin, rlocal = get_local_id(1); rlocal < rlocal_max; rin += in_height_local, rlocal += in_height_local) {
			for (int cin = cmin, clocal = get_local_id(2); clocal < clocal_max; cin += in_width_local, clocal += in_width_local) {
				const int in_index_local = (rlocal * clocal_max + clocal) * in_depth + depth;
				if (rin - padding_height < 0 || cin - padding_width < 0 || rin - padding_height >= in_height || cin - padding_width >= in_width) {
					in_local[in_index_local] = 0;
					continue;
				}
				const int in_index = ((n * in_height + rin - padding_height) * in_width + cin - padding_width) * in_depth + depth;
				in_local[in_index_local] = in[in_index];
			}
		}
	}
	work_group_barrier(CLK_LOCAL_MEM_FENCE); //Finish in and weight loading by tiling

	const int rlocal = get_local_id(1);
	const int clocal = get_local_id(2);
	float sum = bias != NULL? bias[filter] : 0;
	// convolution operation for the image locations centered at (rout, cout)
	for (int kr = 0; kr < kernel_height; kr++)
		for (int kc = 0; kc < kernel_width; kc++) {
			int rin = rout * stride_height + kr - padding_height;
			int cin = cout * stride_width + kc - padding_width;
			if (rin < 0 || rin >= in_height || cin < 0 || cin >= in_width)
				continue;
#ifdef CONVOLUTION_VECTOR
			int weight_index = ((filter * kernel_height + kr) * kernel_width + kc) * in_depth;
			int in_index = ((n * in_height + rin) * in_width + cin) * in_depth;
			int channel = 16;
			float16 sum16 = 0;
			for (; channel <= in_depth; channel += 16, weight_index += 16, in_index += 16)
				sum16 += (*(global float16*) (weight + weight_index)) * (*(global float16*) (in + in_index)); //error CL_OUT_OF_RESOURCES on NVIDIA GTX1050Ti, driver version 382.05
			float8 sum8 = sum16.lo + sum16.hi;
			float4 sum4 = sum8.lo + sum8.hi;
			float2 sum2 = sum4.lo + sum4.hi;
			sum += sum2.lo + sum2.hi;
			for (channel -= 16; channel < in_depth; channel++) //cross channel
				sum += weight[weight_index++] * in[in_index++];
#else
			int rin_local = rlocal * stride_height + kr;
			int cin_local = clocal * stride_width + kc;
			int weight_index_local = ((filter_local * kernel_height + kr) * kernel_width + kc) * in_depth;
			int in_index_local = (rin_local * clocal_max + cin_local) * in_depth;
			for (int channel = 0; channel < in_depth; channel++) //cross channel
				sum += weight_local[weight_index_local++] * in_local[in_index_local++];
#endif
		}
	out[offset] = relu(sum);
}

//Parallel: (batch_size * out_height * out_width * out_depth)
kernel void feed_forward_convolution_activation_relu(global float* out, const global float* weight/*out_depth * kernel_height * kernel_width * in_depth*/, const global float* bias,
		const global float* in/*batch_size * in_height * in_width * in_depth*/, const int in_height, const int in_width, const int in_depth,
		const int kernel_height, const int kernel_width, const int stride_height, const int stride_width,
		const int padding_height, const int padding_width, const int batch_size)
{
	const int out_height = get_global_size(1);
	const int out_width = get_global_size(2);
	const int out_depth = get_global_size(0) / batch_size;
	const int n = get_global_id(0) / out_depth; //batch
	const int rout = get_global_id(1);
	const int cout = get_global_id(2);
	const int filter = get_global_id(0) % out_depth;
	const int offset = ((n * out_height + rout) * out_width + cout) * out_depth + filter;

	float sum = bias != NULL? bias[filter] : 0;
	// convolution operation for the image locations centered at (rout, cout)
	for (int kr = 0; kr < kernel_height; kr++)
		for (int kc = 0; kc < kernel_width; kc++) {
			int rin = rout * stride_height + kr - padding_height;
			int cin = cout * stride_width + kc - padding_width;
			if (rin < 0 || rin >= in_height || cin < 0 || cin >= in_width)
				continue;
			int weight_index = ((filter * kernel_height + kr) * kernel_width + kc) * in_depth;
			int in_index = ((n * in_height + rin) * in_width + cin) * in_depth;
#ifdef CONVOLUTION_VECTOR
			int channel = 16;
			float16 sum16 = 0;
			for (; channel <= in_depth; channel += 16, weight_index += 16, in_index += 16)
				sum16 += (*(global float16*) (weight + weight_index)) * (*(global float16*) (in + in_index)); //error CL_OUT_OF_RESOURCES on NVIDIA GTX1050Ti, driver version 382.05
			float8 sum8 = sum16.lo + sum16.hi;
			float4 sum4 = sum8.lo + sum8.hi;
			float2 sum2 = sum4.lo + sum4.hi;
			sum += sum2.lo + sum2.hi;
			for (channel -= 16; channel < in_depth; channel++) //cross channel
				sum += weight[weight_index++] * in[in_index++];
#else
			for (int channel = 0; channel < in_depth; channel++) //cross channel
				sum += weight[weight_index++] * in[in_index++];
#endif
		}
	out[offset] = relu(sum);
}

//Parallel: ((in_depth * out_depth) * kernel_height * kernel_width)
kernel void back_propagate_convolution_relu_gradient_for_weight(global float* weight_grad/*out_depth * kernel_height * kernel_width * in_depth*/,
		global float* bias_grad/*out_depth*/, const global float* in, const global float* out,
		const global float* out_grad, const int in_height, const int in_width, const int in_depth,
		const int out_height, const int out_width, const int out_depth, const int stride_height, const int stride_width,
		const int padding_height, const int padding_width, const int batch_size/*, const int local_height, const int local_width, const int local_depth,
		local float* out_local, local float* out_grad_local, local float* in_local*/) //Tiling version
{
	const int kernel_height = get_global_size(1);
	const int kernel_width = get_global_size(2);
	const int kr = get_global_id(1);
	const int kc = get_global_id(2);

	const int filter = get_global_id(0) / in_depth;
	const int kd = get_global_id(0) % in_depth;
//	const int filter = get_global_id(0) / local_depth / in_depth * local_depth + (get_global_id(0) % local_depth);
//	const int kd = (get_global_id(0) / local_depth) % in_depth;

	const int GID = ((filter * kernel_height + kr) * kernel_width + kc) * in_depth + kd;
	float sum_weight_grad = 0, sum_bias_grad = 0;
	int in_offset = kd;
	int out_offset = filter;

//	const int group_depth = get_local_size(0);
//	const int group_height = get_local_size(1);
//	const int group_width = get_local_size(2);
//	const int parallel_plane = group_height * group_width * group_depth / local_depth;
//	const int pos = ((get_local_id(0) / local_depth) * group_height + get_local_id(1)) * group_width + get_local_id(2);
//	const int out_filter_size = local_height * local_width * local_depth;
//	const int rlocal_max = local_height * stride_height + 2 * padding_height;
//	const int clocal_max = local_width * stride_width + 2 * padding_width;
//	const int kd_local = kd % (group_depth / local_depth);

	for (int n = 0; n < batch_size; n++, in_offset += in_height * in_width * in_depth, out_offset += out_height * out_width * out_depth)
//		for (int rout = 0; rout < out_height; rout += local_height)
//			for (int cout = 0; cout < out_width; cout += local_width) {
//				const int rmin = rout * stride_height;
//				const int cmin = cout * stride_width;
//				work_group_barrier(CLK_LOCAL_MEM_FENCE); //Prepare to update local memory
//				for (int i = pos; i < out_filter_size; i += parallel_plane) {
//					const int out_column_local = (i / local_depth) % local_width;
//					const int out_row_local = i / local_depth / local_width;
//					const int out_index = ((n * out_height + rout + out_row_local) * out_width + cout + out_column_local) * out_depth + filter;
//					const int out_index_local = (out_row_local * local_width + out_column_local) * local_depth + (filter % local_depth);
//					out_local[out_index_local] = out[out_index];
//					out_grad_local[out_index_local] = out_grad[out_index];
//				}
//
//				for (int rin_local = get_local_id(1), rin = rmin + rin_local - padding_height; rin_local < rlocal_max; rin += group_height, rin_local += group_height) {
//					for (int cin_local = get_local_id(2), cin = cmin + cin_local - padding_width; cin_local < clocal_max; cin += group_width, cin_local += group_width) {
//						const int in_index_local = (rin_local * clocal_max + cin_local) * group_depth / local_depth + kd_local;
//						if (rin < 0 || cin < 0) {
//							in_local[in_index_local] = 0;
//							continue;
//						}
//						const int in_index = ((n * in_height + rin) * in_width + cin) * in_depth + kd;
//						in_local[in_index_local] = in[in_index];
//					}
//				}
//				work_group_barrier(CLK_LOCAL_MEM_FENCE); //Finish in and weight loading by tiling
//
//				for (int rlocal = 0; rlocal < local_height; rlocal++) {
//					int rin_local = rlocal * stride_height + kr;
//					int rin = rmin + rin_local - padding_height;
//					if (rin < 0 || rin >= in_height)
//						continue;
//					for (int clocal = 0; clocal < local_width; clocal++) {
//						int cin_local = clocal * stride_width + kc;
//						int cin = cmin + cin_local - padding_width;
//						if (cin < 0 || cin >= in_width)
//							continue;
//
//						int in_index_local = (rin_local * clocal_max + cin_local) * group_depth / local_depth + kd_local;
//						int out_index_local = (rlocal * local_width + clocal) * local_depth + (filter % local_depth);
//						float out_gradient = out_grad_local[out_index_local];
//						float func_grad = relu_gradient(out_local[out_index_local]);
//						float data_local = in_local[in_index_local];
//						float gradient = func_grad * out_gradient;
//						sum_bias_grad += gradient;
//						sum_weight_grad += gradient * data_local;
//					}
//				}
//			}

		for (int rout = 0; rout < out_height; rout++) {
			int rin = rout * stride_height + kr - padding_height;
			if (rin < 0 || rin >= in_height)
				continue;
			for (int cout = 0; cout < out_width; cout++) {
				int cin = cout * stride_width + kc - padding_width;
				if (cin < 0 || cin >= in_width)
					continue;
				int in_index = in_offset + (rin * in_width + cin) * in_depth;
				int out_index = out_offset + (rout * out_width + cout) * out_depth;
				float out_gradient = out_grad[out_index];
				float func_grad = relu_gradient(out[out_index]);
				float data = in[in_index];
				float gradient = func_grad * out_gradient;
				sum_bias_grad += gradient;
				sum_weight_grad += gradient * data;
			}
		}

	weight_grad[GID] += sum_weight_grad;
	if (bias_grad != NULL && kr == 0 && kc == 0 && kd == 0)
		bias_grad[filter] += sum_bias_grad;
}

//Parallel: ((batch_size * in_depth) * in_height * in_width)
kernel void back_propagate_convolution_relu_gradient_for_input(global float* in_grad, const global float* weight, const global float* out,
		const global float* out_grad, const int kernel_height, const int kernel_width, const int in_depth,
		const int out_height, const int out_width, const int out_depth, const int stride_height, const int stride_width,
		const int padding_height, const int padding_width, const int batch_size/*, local float* out_local, local float* out_grad_local, local float* weight_local,
		const int local_weight_depth, const int local_height, const int local_width, const int local_depth*/)
{
//if (stride_height == 2 && kernel_width != 1)
//	return;
	const int in_height = get_global_size(1);
	const int in_width = get_global_size(2);
	const int n = get_global_id(0) / in_depth; //batch
	const int rin = get_global_id(1);
	const int cin = get_global_id(2);
	const int channel = get_global_id(0) % in_depth;

	const int GID = ((n * in_height + rin) * in_width + cin) * in_depth + channel;
//bool flag = n == 14 && rin == 14 && cin == 5 && channel == 25 && stride_height == 2;
//bool flag = GID == 293762;
	float sum_in_grad = 0;
	const int kernel_volume = kernel_height * kernel_width * in_depth;
	const float rout_min = max(0, (rin + padding_height - kernel_height + 1) / stride_height);
	const float rout_max = min(out_height - 1, (rin + padding_height) / stride_height);
	const float cout_min = max(0, (cin + padding_width - kernel_width + 1) / stride_width);
	const float cout_max = min(out_width - 1, (cin + padding_width) / stride_width);
	for (int rout = rout_min; rout <= rout_max; rout++) {
		int kr = rin + padding_height - rout * stride_height;
		if (kr < 0 || kr >= kernel_height)
			continue;
		for (int cout = cout_min; cout <= cout_max; cout++) {
			int kc = cin + padding_width - cout * stride_width;
			if (kc < 0 || kc >= kernel_width)
				continue;
//if (flag)
//	printf("rout:%d cout:%d kr:%d kc:%d\n", rout, cout, kr, kc);
			int out_index = ((n * out_height + rout) * out_width + cout) * out_depth;
			int weight_index = (kr * kernel_width + kc) * in_depth + channel;
			for (int filter = 0; filter < out_depth; filter++, out_index++, weight_index += kernel_volume) {
				float out_gradient = out_grad[out_index];
				float func_grad = relu_gradient(out[out_index]);
				float factor = weight[weight_index];
				sum_in_grad += func_grad * out_gradient * factor;
//				if (flag)
//					printf("	out_gradient(%d,%d):%g weight(%d,%d):%g sum_in_grad:%g\n", ((n * out_height + rout) * out_width + cout + 1), filter + 1, out_gradient, ((filter * kernel_height + kr) * kernel_width + kc + 1), channel + 1, factor, sum_in_grad);
			}
		}
	}
	in_grad[GID] gradient_set_type sum_in_grad;
//if (flag)
//	printf("in_grad:%g\n", sum_in_grad);
}

//Parallel: (batch_size * out_height * out_width * out_depth)
kernel void feed_forward_max_pooling(global float* out, const global float* in/*batch_size * in_height * in_width * in_depth*/,
		const int in_height, const int in_width, const int in_depth/*equals to out_depth*/,
		const int pool_height, const int pool_width, const int stride_height, const int stride_width,
		const int padding_height, const int padding_width, const int batch_size, global int* out_index)
{
	const int out_height = get_global_size(1);
	const int out_width = get_global_size(2);
	const int out_depth = in_depth;
	const int n = get_global_id(0) / out_depth; //batch
	const int rout = get_global_id(1);
	const int cout = get_global_id(2);
	const int filter = get_global_id(0) % out_depth;
	const int offset = ((n * out_height + rout) * out_width + cout) * out_depth + filter;

	float max = -FLT_MAX, max_index;
	for (int pr = 0; pr < pool_height; pr++)
		for (int pc = 0; pc < pool_width; pc++) {
			int rin = rout * stride_height + pr - padding_height;
			int cin = cout * stride_width + pc - padding_width;
			if (rin < 0 || rin >= in_height || cin < 0 || cin >= in_width) {
				if (max < 0) {
					max = 0;
					max_index = -1;
				}
				continue;
			}
			int in_index = ((n * in_height + rin) * in_width + cin) * in_depth + filter; //channel==filter
			if (in[in_index] > max) {
				max = in[in_index];
				max_index = in_index;
			}
		}
	out[offset] = max;
	out_index[offset] = max_index;
}

//Parallel: (batch_size * in_height * in_width * in_depth)
kernel void back_propagate_max_pooling(global float* in_grad, const global float* out_grad,
		const int out_height, const int out_width, const int out_depth/*equals to in_depth*/,
		const int pool_height, const int pool_width, const int stride_height, const int stride_width,
		const int padding_height, const int padding_width, const int batch_size, const global int* max_index)
{
	const int in_height = get_global_size(1);
	const int in_width = get_global_size(2);
	const int in_depth = out_depth;
	const int n = get_global_id(0) / in_depth; //batch
	const int rin = get_global_id(1);
	const int cin = get_global_id(2);
	const int channel = get_global_id(0) % in_depth;
	const int global_size = in_height * in_width * in_depth;
	const int offset = (rin * in_width + cin) * in_depth + channel;

	float gradient = 0;
	int in_index = ((n * in_height + rin) * in_width + cin) * in_depth + channel;
	for (int pr = 0; pr < pool_height; pr++)
		for (int pc = 0; pc < pool_width; pc++) {
			int rout = (rin - pr + padding_height) / stride_height;
			int cout = (cin - pc + padding_width) / stride_width;
			if (rout < 0 || rout >= out_height || cout < 0 || cout >= out_width)
				continue;
			int out_index = ((n * out_height + rout) * out_width + cout) * out_depth + channel; //filter==channel
			int index = (int) max_index[out_index];
			gradient += in_index == index? out_grad[out_index] : 0;
		}
	in_grad[in_index] = gradient;
}

//Parallel: (batch_size * out_height * out_width * out_depth)
kernel void feed_forward_average_pooling(global float* out, const global float* in/*batch_size * in_depth * in_height * in_width*/,
		const int in_height, const int in_width, const int in_depth,
		const int pool_height, const int pool_width, const int stride_height, const int stride_width,
		const int padding_height, const int padding_width, const int batch_size)
{
	const int out_height = get_global_size(1);
	const int out_width = get_global_size(2);
	const int out_depth = in_depth;
	const int n = get_global_id(0) / out_depth; //batch
	const int rout = get_global_id(1);
	const int cout = get_global_id(2);
	const int filter = get_global_id(0) % out_depth;
	const int offset = ((n * out_height + rout) * out_width + cout) * out_depth + filter;

	float sum = 0;
	for (int pr = 0; pr < pool_height; pr++)
		for (int pc = 0; pc < pool_width; pc++) {
			int rin = rout * stride_height + pr - padding_height;
			int cin = cout * stride_width + pc - padding_width;
			if (rin < 0 || rin >= in_height || cin < 0 || cin >= in_width)
				continue;
			int in_index = ((n * in_height + rin) * in_width + cin) * in_depth + filter; //channel==filter
			sum += in[in_index];
		}
	out[offset] = sum / pool_height / pool_width;
}

//Parallel: (batch_size * in_height * in_width * in_depth)
kernel void back_propagate_average_pooling(global float* in_grad, const global float* out_grad,
		const int out_height, const int out_width, const int out_depth/*equals to in_depth*/,
		const int pool_height, const int pool_width, const int stride_height, const int stride_width,
		const int padding_height, const int padding_width, const int batch_size)
{
	const int in_height = get_global_size(1);
	const int in_width = get_global_size(2);
	const int in_depth = out_depth;
	const int n = get_global_id(0) / in_depth; //batch
	const int rin = get_global_id(1);
	const int cin = get_global_id(2);
	const int channel = get_global_id(0) % in_depth;
	const int global_size = in_height * in_width * in_depth;
	const int offset = (rin * in_width + cin) * in_depth + channel;

	float gradient = 0;
	int in_index = ((n * in_height + rin) * in_width + cin) * in_depth + channel;
	for (int pr = 0; pr < pool_height; pr++)
		for (int pc = 0; pc < pool_width; pc++) {
			int rout = (rin - pr + padding_height) / stride_height;
			int cout = (cin - pc + padding_width) / stride_width;
			if (rout < 0 || rout >= out_height || cout < 0 || cout >= out_width)
				continue;
			int out_index = ((n * out_height + rout) * out_width + cout) * out_depth + channel; //filter==channel
			gradient += out_grad[out_index];
		}
	in_grad[in_index] = gradient / pool_height / pool_width;
}

//kernel sum pooling: omitted. use average pooling instead.

//Parallel: (num_hidden)
kernel void feed_forward_dropout(global float* out, const global float* mask/*num_hidden*/,
		local float* tmp, const int num_hidden, const float p, const int batch_size)
{
	int GID = get_global_id(0);

	for (int n = 0; n < batch_size; n++)
		out[n * get_global_size(0) + GID] *= mask[GID];
}

//Parallel: (num_hidden)
kernel void back_propagate_dropout(global float* out_grad, const global float* mask/*num_hidden*/,
		local float* tmp, const int num_hidden, const float p, const int batch_size, const float max_norm)
{
	int GID = get_global_id(0);

	for (int n = 0; n < batch_size; n++)
		out_grad[n * get_global_size(0) + GID] *= mask[GID];
}

//Parallel: (get_local_size(0) * vector_length)
kernel void feed_forward_embedding(global float* out, const global float* input,
		const global float* vector_weight, local float* tmp, const int dim_in, const int vector_length)
{
	const int GID = get_global_id(0);
	const int parallel = get_local_size(0);
	const int weight_offset = GID / parallel;

	for (int index = GID % parallel; index < dim_in; index += parallel)
		out[index * vector_length + weight_offset] = vector_weight[((int) input[index]) * vector_length + weight_offset];
}

//Parallel: (vector_length)
kernel void back_propagate_embedding(global float* vector_weight_grad, const global float* input,
		const global float* out_grad, local float* tmp, const int dim_in, const int vector_length, const int dim_vector_num)
{
	const int GID = get_global_id(0);
	for (int i = 0; i < dim_in; i++)
		vector_weight_grad[((int) input[i]) * vector_length + GID] += out_grad[i * vector_length + GID];
}

//Parallel: (dim_in * local(min(batch_size, work_group_size)))
//Parallel: (in_depth * local(min(batch_size * H * W, work_group_size))) // sum on batch_size * H * W for convolutional case
kernel void feed_forward_batch_normalization(global float* out, global float* deviation, global float* std_dev,
		global float* moving_mean, global float* moving_variance, const global float* in,
		const global float* gamma, const global float* beta, const float epsilon, const float momentum, const int dim_in, const int batch_size)
{
	const int GID = get_global_id(0);
	const int parallel = get_local_size(0); //parallel addition, trade space for time
	const int k = GID / parallel;
	const int n = GID % parallel;
	
	local float mean_[MAX_WORK_GROUP_SIZE], sigma2_[MAX_WORK_GROUP_SIZE];
	mean_[n] = 0;
	sigma2_[n] = 0;
	for (int i = n; i < batch_size; i += parallel)
		mean_[n] += in[i * dim_in + k];
	for (int stride = parallel / 2; stride > 0; stride = stride / 2) {
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
		if (n < stride)
			mean_[n] += mean_[n + stride];
	}
	work_group_barrier(CLK_LOCAL_MEM_FENCE);
	float mean = mean_[0] / batch_size;
	moving_mean[k] = moving_mean[k] * (1.0f - momentum) + mean * momentum;

	for (int i = n; i < batch_size; i += parallel) {
		const float delta = in[i * dim_in + k] - mean;
		deviation[i * dim_in + k] = delta;
		sigma2_[n] += delta * delta;
	}
	for (int stride = parallel / 2; stride > 0; stride = stride / 2) {
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
		if (n < stride)
			sigma2_[n] += sigma2_[n + stride];
	}
	work_group_barrier(CLK_LOCAL_MEM_FENCE);
	float sigma2 = sigma2_[0] / batch_size;
	moving_variance[k] = moving_variance[k] * (1.0f - momentum) + sigma2 * momentum;
	sigma2 = sqrt(sigma2 + epsilon);
	if (std_dev != NULL)
		std_dev[k] = sigma2;

	for (int i = n; i < batch_size; i += parallel)
		out[i * dim_in + k] = gamma[k] / sigma2 * deviation[i * dim_in + k] + beta[k];
}

//Parallel: (dim_in) // sum on batch_size
kernel void feed_forward_batch_normalization_small(global float* out, global float* deviation, global float* std_dev,
		global float* moving_mean, global float* moving_variance, const global float* in,
		const global float* gamma, const global float* beta, const float epsilon, const float momentum, const int dim_in, const int batch_size)
{
	const int k = get_global_id(0);

	float mean = 0;
	for (int i = 0; i < batch_size; i++)
		mean += in[i * dim_in + k];
	mean /= batch_size;
	moving_mean[k] = moving_mean[k] * (1.0f - momentum) + mean * momentum;

	float sigma2 = 0;
	for (int i = 0; i < batch_size; i++) {
		const float delta = in[i * dim_in + k] - mean;
		deviation[i * dim_in + k] = delta;
		sigma2 += delta * delta;
	}
	sigma2 = sigma2 / batch_size;
	moving_variance[k] = moving_variance[k] * (1.0f - momentum) + sigma2 * momentum;
	sigma2 = sqrt(sigma2 + epsilon);
	if (std_dev != NULL)
		std_dev[k] = sigma2;

	for (int i = 0; i < batch_size; i++)
		out[i * dim_in + k] = gamma[k] / sigma2 * deviation[i * dim_in + k] + beta[k];
}

//Parallel: (dim_in * batch_size)
//Parallel: (in_depth * (batch_size * H * W)) //for convolutional case
kernel void feed_forward_batch_normalization_for_inference(global float* out, const global float* moving_mean, const global float* moving_variance, 
		const global float* in, const global float* gamma, const global float* beta, const float epsilon, const int dim_in, const int batch_size)
{
	const int GID = get_global_id(0);
	const int k = GID / batch_size;
	const int n = GID % batch_size;

	out[n * dim_in + k] = gamma[k] / sqrt(moving_variance[k] + epsilon) * (in[n * dim_in + k] - moving_mean[k]) + beta[k];
}

//Parallel: (in_depth * local(min(batch_size * H * W, work_group_size))) //for convolutional case
kernel void back_propagate_batch_normalization(global float* in_grad, global float* gamma_grad, global float* beta_grad,
		const global float* gamma, const global float* deviation, const global float* std_dev, const global float* out_grad,
		global float* deviation_grad, const int dim_in, const int batch_size)
{
	const int GID = get_global_id(0);
	const int parallel = get_local_size(0); //parallel addition, trade space for time
	const int k = GID / parallel;
	const int n = GID % parallel;

	local float variance_grad_[MAX_WORK_GROUP_SIZE];
	local float mu_grad_[MAX_WORK_GROUP_SIZE], mu_tmp1_[MAX_WORK_GROUP_SIZE], mu_tmp2_[MAX_WORK_GROUP_SIZE];
	local float gamma_gradient_[MAX_WORK_GROUP_SIZE], beta_gradient_[MAX_WORK_GROUP_SIZE];
	variance_grad_[n] = 0;
	mu_grad_[n] = 0;
	mu_tmp1_[n] = 0;
	mu_tmp2_[n] = 0;
	gamma_gradient_[n] = 0;
	beta_gradient_[n] = 0;
	
	for (int i = n; i < batch_size; i += parallel) {
		deviation_grad[i * dim_in + k] = out_grad[i * dim_in + k] * gamma[k]; //dim_out == dim_in
		variance_grad_[n] += deviation_grad[i * dim_in + k] * deviation[i * dim_in + k];
		mu_tmp1_[n] += deviation_grad[i * dim_in + k];
		mu_tmp2_[n] += deviation[i * dim_in + k];
		gamma_gradient_[n] += out_grad[i * dim_in + k] * deviation[i * dim_in + k];
		beta_gradient_[n] += out_grad[i * dim_in + k];
	}
	for (int stride = parallel / 2; stride > 0; stride = stride / 2) {
		work_group_barrier(CLK_LOCAL_MEM_FENCE);
		if (n < stride) {
			variance_grad_[n] += variance_grad_[n + stride];
			mu_tmp1_[n] += mu_tmp1_[n + stride];
			mu_tmp2_[n] += mu_tmp2_[n + stride];
			gamma_gradient_[n] += gamma_gradient_[n + stride];
			beta_gradient_[n] += beta_gradient_[n + stride];
		}
	}
	work_group_barrier(CLK_LOCAL_MEM_FENCE);
	float variance_grad = variance_grad_[0];
	float mu_grad = mu_grad_[0], mu_tmp1 = mu_tmp1_[0], mu_tmp2 = mu_tmp2_[0];
	float gamma_gradient = gamma_gradient_[0], beta_gradient = beta_gradient_[0];

	variance_grad *= - 0.5f / pow(std_dev[k], 3);
	mu_grad = - mu_tmp1 / std_dev[k] - 2 * variance_grad / batch_size * mu_tmp2;
	for (int i = n; i < batch_size; i += parallel)
		in_grad[i * dim_in + k] gradient_set_type deviation_grad[i * dim_in + k] / std_dev[k] + 2 * variance_grad * deviation[i * dim_in + k] / batch_size + mu_grad / batch_size;
	gamma_gradient /= std_dev[k];
	gamma_grad[k] = gamma_gradient;
	beta_grad[k] = beta_gradient;
}

//Parallel: (dim_in * local(batch_size))
kernel void back_propagate_batch_normalization_small(global float* in_grad, global float* gamma_grad, global float* beta_grad,
		const global float* gamma, const global float* deviation, const global float* std_dev,
		const global float* out_grad, local float* deviation_grad, const int dim_in, const int batch_size)
{
	const int GID = get_global_id(0);
	const int k = GID / batch_size;
	const int n = GID % batch_size;

	deviation_grad[n] = out_grad[n * dim_in + k] * gamma[k]; //dim_out == dim_in
	work_group_barrier(CLK_LOCAL_MEM_FENCE);

	float variance_grad = 0;
	float mu_grad = 0, mu_tmp1 = 0, mu_tmp2 = 0;
	float gamma_gradient = 0, beta_gradient = 0;
	for (int i = 0; i < batch_size; i++) {
		variance_grad += deviation_grad[i] * deviation[i * dim_in + k];
		mu_tmp1 += deviation_grad[i];
		mu_tmp2 += deviation[i * dim_in + k];
		gamma_gradient += out_grad[i * dim_in + k] * deviation[i * dim_in + k];
		beta_gradient += out_grad[i * dim_in + k];
	}
	variance_grad *= - 0.5f / pow(std_dev[k], 3);
	mu_grad = - mu_tmp1 / std_dev[k] - 2 * variance_grad / batch_size * mu_tmp2;
	in_grad[n * dim_in + k] gradient_set_type deviation_grad[n] / std_dev[k] + 2 * variance_grad * deviation[n * dim_in + k] / batch_size + mu_grad / batch_size;
	gamma_gradient /= std_dev[k];
	gamma_grad[k] = gamma_gradient;
	beta_grad[k] = beta_gradient;
}

//Parallel: (sizeof(data))
kernel void feed_forward_activation_sigmoid(global float* out, const global float* in)
{
	const int GID = get_global_id(0);
	out[GID] = sigmoid(in[GID]);
}

//Parallel: (sizeof(data))
kernel void back_propagate_activation_sigmoid(global float* in_grad, const global float* out_grad, const global float* out)
{
	const int GID = get_global_id(0);
	in_grad[GID] gradient_set_type sigmoid_gradient(out[GID]) * out_grad[GID];
}

//Parallel: (in_size * local_size)
kernel void feed_forward_concatenate(global float* out, const global float* in, const int out_offset, const int out_stride, const int out_num, const int in_size)
{
	const int GID = get_global_id(0);
	const int parallel = get_local_size(0);
	const int k = GID / parallel;
	const int n = GID % parallel;

	for (int i = n; i < out_num; i += parallel)
		out[out_stride * i + out_offset + k] = in[i * in_size + k];
}

//Parallel: (out_size * local_size)
kernel void feed_forward_split(global float* out, const global float* in, const int in_offset, const int in_stride, const int in_num, const int out_size)
{
	const int GID = get_global_id(0);
	const int parallel = get_local_size(0);
	const int k = GID / parallel;
	const int n = GID % parallel;

	for (int i = n; i < in_num; i += parallel)
		out[i * out_size + k] = in[in_stride * i + in_offset + k];
}
