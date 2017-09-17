/*
 * MNIST_CNN.cpp
 *
 *  Created on: 2017/8/18
 *      Author: ZhangHua
 */

#include <iostream>
#include <algorithm>

#include <tensor.hpp>
#include <device_instance.hpp>
#include <image_io.hpp>

using namespace std;
using namespace clnet;

class MNISTImageIterator : public type::MiniBatch, public type::Structured {
public:
	MNISTImageIterator(string path, int size) : type::MiniBatch(size) {
		peers.push_back(read_mnist_images(path + "train-images.idx3-ubyte", "train_images", size));
		peers.push_back(read_mnist_labels(path + "train-labels.idx1-ubyte", "train_labels", size));
		peers.push_back(read_mnist_images(path + "t10k-images.idx3-ubyte", "test_images", size));
		peers.push_back(read_mnist_labels(path + "t10k-labels.idx1-ubyte", "test_labels", size));

		set_total_samples(peers[0]->dimensions[0]);
		peers.push_back(new type::MiniBatch(size, peers[2]->dimensions[0])); //peers[4]
	}

	void save_as_24bits_bmp(int start, int end, bool is_test_data, string path) {
		auto p = is_test_data? peers[2]->pointer : peers[0]->pointer;
		auto label = is_test_data? peers[3]->pointer : peers[1]->pointer;
		auto height = peers[0]->dimensions[1], width = peers[0]->dimensions[2];
		auto length = height * width;
		auto buffer = new unsigned char[length * 3];
		for (int i = start; i < end; i++) {
			auto data = p + i * length;
			for (int r = height - 1; r >= 0; r--)
				for (int c = 0; c < width; c++) {
					int offset = (r * width + c) * 3;
					buffer[offset] = buffer[offset + 1] = buffer[offset + 2] = (unsigned char) *data++;
				}
			generate_24bits_bmp(buffer, width, height, (path + to_string(i) + "-" + to_string(static_cast<int>(label[i])) + ".bmp").c_str());
		}
	}

	virtual std::string generate_source_code(DeviceInstance& I) override {
        std::string kernel{R"CLC(
kernel void load_mnist_images(global float* data, global float* label, const global float* images, const global float* labels, const global int* index, const int index_offset, const int index_size)
{
	const int GID = get_global_id(0);
	const int PIXELS = get_global_size(0);
	const global int* offset = index + index_offset + 1;
	for (int i = 0; i < index_size; i++)
		data[i * PIXELS + GID] = images[offset[i] * PIXELS + GID];
	for (int i = GID; i < index_size; i += PIXELS)
		label[i] = labels[offset[i]];
}
)CLC"};
		return kernel;
	}

	virtual void run(DeviceInstance& I) override {
		auto& kernel = prepare_for_running_kernel(this, I);
		kernel.setArg(0, I.buffers[peers[5]]);
		kernel.setArg(1, I.buffers[peers[6]]);
		int batch = reinterpret_cast<int*>(I.pointers[this])[0];
		if (batch < total_batches) {
			kernel.setArg(2, I.buffers[peers[0]]);
			kernel.setArg(3, I.buffers[peers[1]]);
			kernel.setArg(4, I.buffers[this]);
			kernel.setArg(5, batch * batch_size);
		}
		else { //run test data
			auto tester = peers[4];
			kernel.setArg(2, I.buffers[peers[2]]);
			kernel.setArg(3, I.buffers[peers[3]]);
			kernel.setArg(4, I.buffers[tester]);
			kernel.setArg(5, reinterpret_cast<int*>(I.pointers[tester])[0] * batch_size);
		}
		kernel.setArg(6, batch_size);
		cl::NDRange global(peers[0]->dimensions[1] * peers[0]->dimensions[2]);
		I.queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, cl::NullRange, &I.precondition_events, &I.events[peers[5]]);
		I.events[peers[6]] = I.events[peers[5]];
	}

	virtual void reset(DeviceInstance& I) override {
		type::MiniBatch::reset(I);
		download(I);
	}

	virtual void initialize(DeviceInstance* I) override {
		type::MiniBatch::initialize(I);
		download(*I);
		peers[4]->initialize(I);
		peers[4]->download(*I);
	}

	virtual std::vector<Tensor*> auxiliaries() override {
		return {peers[0], peers[1], peers[2], peers[3]};
	}
};

T MNIST_CNN(bool is_predict)
{
	const string mnist_folder = optional<string>("mnist_folder", "E:\\DataSets\\MNIST\\");
	const int batch_size = optional<int>("batch_size", 32);
	auto iterator = new MNISTImageIterator(mnist_folder, batch_size);
	vector<int64> dims{batch_size, iterator->peers[0]->dimensions[1], iterator->peers[0]->dimensions[2], 1};
	T data = *new Tensor(dims, {}, "train_images_data");
	data.dependent_on(iterator);

	//when filters3 > 480, AMD R9 295X2 (Hawaii) device runs unstably, randomly cause program to hung
	const int kernel_size = 5, stride = 1, filters1 = 20, filters2 = 50, filters3 = 480, class_num = 10;
	const string activation = "tanh", pooling_type = "max";
	T conv1 = ConvolutionKernel(data, filters1, kernel_size, stride, activation, true, "conv1");
	T pool1 = Pooling(conv1, {2}, {}, pooling_type, true, "pool1");
	T conv2 = ConvolutionKernel(pool1, filters2, kernel_size, stride, activation, true, "conv2");
	T pool2 = Pooling(conv2, {2}, {}, pooling_type, true, "pool2");
	T conv3 = ConvolutionKernel(pool2, filters3, kernel_size, stride, activation, true, "conv3");
	T pool3 = Pooling(conv3, {2}, {}, pooling_type, true, "pool3");
	T reshape = Reshape(pool3, {pool3.dimensions[0], pool3.volume / pool3.dimensions[0]});

//	T reshape = Reshape(pool1, {pool1.dimensions[0], pool1.volume / pool1.dimensions[0]});
//	T reshape = Reshape(data, {data.dimensions[0], data.volume / data.dimensions[0]});
	T feature = FullyConnectedLayer(reshape, filters3, activation, "feature");
	T inference = FullyConnectedLayer(feature, class_num, "", "inference");
	if (is_predict)
		return inference;

	const float learning_rate = optional<double>("learning_rate", 0.0002), weight_decay = optional<double>("weight_decay", 0);
	const int max_iters = optional<int>("max_iters", 5000);
	T label = *new Tensor({batch_size}, {}, "train_images_label");
	label.dependent_on(iterator);
	T loss = SoftmaxLoss(inference, label);
	T SGD = StochasticGradientDescentUpdater(loss, learning_rate, weight_decay);
	T initializer = XavierNormalDistributionInitializer(SGD, 0, 2.34f);

	const int N_samples = iterator->peers[0]->dimensions[0];
	auto monitor = new InstantTensor("MNIST_CNN_monitor", {}, {}, [batch_size, N_samples, &loss](InstantTensor* self, DeviceInstance& I) {
		auto optimizer = static_cast<type::IterativeOptimizer*>(self->peers[0]);
		auto epoch = optimizer->current_epoch(I);

		float accuracy = static_cast<back::Loss*>(&loss)->L(I);
		accuracy = exp(-accuracy / batch_size);
		size_t duration = optimizer->milliseconds_since_last(I);
		string speed = epoch == 0? to_string(duration) + "ms" : to_string(1000.0f * N_samples / duration) + "/s";
		cout << "[" << I.ID << "," << epoch << "," << speed << "] accuracy: " << accuracy  << endl;
	});
	auto validator = new InstantTensor("MNIST_CNN_validator", {}, [iterator, class_num, &inference, &label](InstantTensor* self, DeviceInstance& I) {
		auto optimizer = static_cast<type::IterativeOptimizer*>(self->peers[0]);
		auto epoch = optimizer->current_epoch(I);
		if (epoch % 10 != 0)
			return;

		set<Tensor*> visited;
		auto tester = static_cast<type::MiniBatch*>(iterator->peers[4]);
		auto& offset = reinterpret_cast<int*>(I.pointers[tester])[0];
		int correct = 0, N = inference.dimensions[0];
		while (tester->has_next(I)) {
			visited.clear();
			inference.launch(&visited, &I);
			wait_for_all_kernels_finished(I);

			inference.upload(I);
			label.upload(I);
			float* output = I.pointers[&inference];
			float* labels = I.pointers[&label];
			for (int i = 0; i < N; i++, output += class_num, labels++)
				if ((max_element(output, output + class_num) - output) == *labels)
					correct++;
		}
		float accuracy = (int) (10000.0f * correct / iterator->peers[2]->dimensions[0]) / 100.0f;
		cout << "[" << I.ID << "," << epoch << "] test set accuracy: " << accuracy  << "%" << endl;
		offset = -1; //random_shuffle on test set is not needed.
	}, {}, [&inference](InstantTensor* self) -> Tensor*{ return &inference; });
	return IterativeOptimizer({&initializer}, {&SGD, iterator, monitor, validator}, max_iters);
}
