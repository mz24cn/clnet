/*
 * MNIST_CNN.cpp
 *
 *  Created on: 2017/8/18
 *      Author: ZhangHua
 */

#include <iostream>
#include <algorithm>
#include <fstream>

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

		set_total_samples(peers[0]->dimensions[0]/*640*/); //small value is used for debug
		peers.push_back(new type::MiniBatch(size, peers[2]->dimensions[0]/*160*/)); //peers[4]
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
		delete buffer;
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
		cl::NDRange global(peers[0]->volume / peers[0]->dimensions[0]);
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
	int batch_size;
	Tensor *tensor, *iterator = nullptr;
	if (is_predict) {
		batch_size = 1;
		tensor = &Data({1, 28, 28, 1}, nullptr, "inference_image_data"); //data will be automatically downloaded to device on every running
	}
	else {
		batch_size = optional<int>("batch_size", 32);
		const string mnist_folder = optional<string>("mnist_folder", "D:\\DataSets\\");
		iterator = new MNISTImageIterator(mnist_folder, batch_size);
		vector<int64> dims{batch_size, iterator->peers[0]->dimensions[1], iterator->peers[0]->dimensions[2], 1};
		tensor = new Tensor(dims, {}, "train_images_data");
		tensor->dependent_on(iterator);
	}

	//when filters3 > 480, AMD R9 295X2 (Hawaii) device runs unstably, randomly cause program to hung
	const int kernel_size = 5, stride = 1, filters1 = 20, filters2 = 50, filters3 = 480, class_num = 10;
	const string activation = "tanh", pooling_type = "max";
//	T plane = Reshape(*tensor, {tensor->dimensions.front(), tensor->volume / tensor->dimensions.front()}, "plane");
//	T FC1 = FullyConnectedLayer(plane, 512, "", "FC1");
//	T BN1 = BatchNormalizedLayer(FC1, 0.001, 0.9, "BN1");
//	T ACT1 = tanh(BN1);
//	T FC2 = FullyConnectedLayer(ACT1, 512, "", "FC2");
//	T BN2 = BatchNormalizedLayer(FC2, 0.001, 0.9, "BN2");
//	T reshape = tanh(BN2);
//	T FC1 = FullyConnectedLayer(plane, 512, activation, "FC1");
//	T FC2 = FullyConnectedLayer(FC1, 512, activation, "FC2");
//	T reshape = FC2;

//	T conv1 = ConvolutionLayer(*tensor, filters1, kernel_size, stride, "", true, false, "conv1");
//	T BN1 = BatchNormalizedLayer(conv1, 0.001, 0.9, "BN1");
//	T ACT1 = tanh(BN1);
//	T pool1 = Pooling(ACT1, {2}, {}, pooling_type, true, "pool1");
//	T conv2 = ConvolutionLayer(pool1, filters2, kernel_size, stride, "", true, false, "conv2");
//	T BN2 = BatchNormalizedLayer(conv2, 0.001, 0.9, "BN2");
//	T ACT2 = tanh(BN2);
//	T pool2 = Pooling(ACT2, {2}, {}, pooling_type, true, "pool2");
//	T reshape = Reshape(pool2, {pool2.dimensions[0], pool2.volume / pool2.dimensions[0]});

	T conv1 = ConvolutionLayer(*tensor, filters1, kernel_size, stride, activation, true, true, "conv1");
	T pool1 = Pooling(conv1, {2}, {}, pooling_type, true, "pool1");
	T conv2 = ConvolutionLayer(pool1, filters2, kernel_size, stride, activation, true, true, "conv2");
	T pool2 = Pooling(conv2, {2}, {}, pooling_type, true, "pool2");
//	T conv3 = ConvolutionLayer(pool2, filters3, 2, stride, activation, true, true, "conv3");
//	T pool3 = Pooling(conv3, {2}, {}, pooling_type, true, "pool3");
//	T reshape = Reshape(pool3, {pool3.dimensions[0], pool3.volume / pool3.dimensions[0]});
	T reshape = Reshape(pool2, {pool2.dimensions[0], pool2.volume / pool2.dimensions[0]});

	T feature = FullyConnectedLayer(reshape, filters3, activation, "feature");
	T inference = FullyConnectedLayer(feature, class_num, "", "inference");

	auto params_file = optional<string>("params_file", "D:\\DataSets\\MNIST_CNN.clnetparams");
	if (is_predict) {
		auto initializer = new InstantTensor("parameters_initializer", {}, {}, [params_file](InstantTensor* self, DeviceInstance& I) {
			ifstream ifs(params_file, istream::binary);
			if (!ifs)
				throw runtime_error("failed to open " + params_file);
			auto tensors = load_tensors(ifs, &I);
			ifs.close();
			logger << to_string(tensors.size()) << " parameters successfully loaded." << endl;
		});
		T output = Softmax(inference);
		auto predictor = new InstantTensor("charRNN_predictor", {initializer}, [&output](InstantTensor* self, DeviceInstance& I) {
			int N = output.volume;
			set<Tensor*> visited;
			output.launch(&visited, &I);
			wait_for_all_kernels_finished(I);

			output.upload(I);
			float* probability = I.pointers[&output];
			auto n = max_element(probability, probability + N) - probability;
			logger << "Image number: " << n << " probability: " << int(probability[n] * 100) << "%" << endl;
		}, {&inference}, [](InstantTensor* self) -> Tensor*{ return self->peers[0]; });

		const string file = optional<string>("file", "D:\\DataSets\\28x28.bmp");
		int width, height;
		unsigned char* buffer = read_24bits_bmp(file.c_str(), &width, &height);
		if (height != tensor->dimensions[1] || width != tensor->dimensions[2])
			throw runtime_error("wrong image size: " + file);
		tensor->initialize(nullptr);
		auto* p = tensor->pointer;
		for (int r = height - 1; r >= 0; r--)
			for (int c = 0; c < width; c++) {
				int offset = (r * width + c) * 3;
				*p++ = 0.3f * buffer[offset] + 0.59f * buffer[offset + 1] + 0.11f * buffer[offset + 2]; //generate grayscale image Tensor data
			}
		return *predictor;
	}

	const float learning_rate = optional<float>("learning_rate", 0.0064), weight_decay = optional<float>("weight_decay", 0);
	const int max_epochs = optional<int>("max_epochs", 5000);
	T label = *new Tensor({batch_size}, {}, "train_images_label");
	label.dependent_on(iterator);
	T loss = CrossEntropyLoss(inference, label);
	T SGD = StochasticGradientDescentUpdater(loss, learning_rate, weight_decay);
	T initializer = GeneralInitializer(SGD.peers);

	vector<Tensor*> parameters;
	for (auto tensor : Tensor::ALL)
		if (dynamic_cast<type::Parameter*>(tensor) != nullptr)
			parameters.push_back(tensor);

	const int N_samples = iterator->peers[0]->dimensions[0];
	auto monitor = new InstantTensor("MNIST_CNN_monitor", {}, {}, [batch_size, N_samples, &loss, parameters, params_file](InstantTensor* self, DeviceInstance& I) {
		auto optimizer = static_cast<type::IterativeOptimizer*>(self->peers[0]);
		auto epoch = optimizer->current_epoch(I);

		size_t duration = optimizer->milliseconds_since_last(I);
		string speed = epoch == 0? to_string(duration) + "ms" : to_string(1000.0f * N_samples / duration) + "/s";
		logger << "[" << I.ID << "," << epoch << "," << speed << "] train loss: " << static_cast<back::Loss*>(&loss)->L(I);

		ofstream ofs(params_file, ostream::binary);
		for (size_t i = 0; i < parameters.size(); i++)
			save_tensor(parameters[i], ofs, &I);
		ofs.close();
	});
	auto validator = new InstantTensor("MNIST_CNN_validator", {}, [iterator, class_num, &inference, &label](InstantTensor* self, DeviceInstance& I) {
//		auto optimizer = static_cast<type::IterativeOptimizer*>(self->peers[0]);
//		auto epoch = optimizer->current_epoch(I);
//		if (epoch % 10 != 0)
//			return;

		set<Tensor*> visited;
		auto tester = static_cast<type::MiniBatch*>(iterator->peers[4]);
		auto& offset = reinterpret_cast<int*>(I.pointers[tester])[0];
		int correct = 0, N = inference.dimensions[0];
		CLNET_TENSOR_GLOBALS |= CLNET_PREDICT_ONLY;
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
		CLNET_TENSOR_GLOBALS ^= CLNET_PREDICT_ONLY;
		float accuracy = (int) (10000.0f * correct / iterator->peers[2]->dimensions[0]) / 100.0f;
		logger << "\ttest set accuracy: " << accuracy  << "%" << endl;
		offset = -1; //random_shuffle on test set is not needed.
	}, {}, [&inference](InstantTensor* self) -> Tensor*{ return &inference; });
	return IterativeOptimizer({&initializer}, {&SGD, iterator, monitor, validator}, max_epochs);
}
