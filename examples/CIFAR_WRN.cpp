/*
 * CIFAR_WRN.cpp
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

class CIFARImageIterator : public type::MiniBatch, public type::Structured {
public:
	CIFARImageIterator(string path, int size, int num_class) : type::MiniBatch(size) {
		auto images = new Tensor({ALIGN(50000, size), 32, 32, 3}, {}, "train_images");
		auto labels = new Tensor({ALIGN(50000, size), 1}, {}, "train_labels");
		auto test_images = new Tensor({ALIGN(10000, size), 32, 32, 3}, {}, "test_images");
		auto test_labels = new Tensor({ALIGN(10000, size), 1}, {}, "test_labels");
		images->initialize(nullptr);
		labels->initialize(nullptr);
		test_images->initialize(nullptr);
		test_labels->initialize(nullptr);
		if (num_class == 10) {
			read_cifar10_images_and_labels(path + "data_batch_1.bin", 1, 0, 10000, images, labels);
			read_cifar10_images_and_labels(path + "data_batch_2.bin", 1, 10000, 10000, images, labels);
			read_cifar10_images_and_labels(path + "data_batch_3.bin", 1, 20000, 10000, images, labels);
			read_cifar10_images_and_labels(path + "data_batch_4.bin", 1, 30000, 10000, images, labels);
			read_cifar10_images_and_labels(path + "data_batch_5.bin", size, 40000, 10000, images, labels);
			read_cifar10_images_and_labels(path + "test_batch.bin", size, 0, 10000, test_images, test_labels);
		}
		else {
			read_cifar100_images_and_labels(path + "train.bin", size, num_class == 100, 50000, images, labels);
			read_cifar100_images_and_labels(path + "test.bin", size, num_class == 100, 10000, test_images, test_labels);
		}

		peers.push_back(images);
		peers.push_back(labels);
		peers.push_back(test_images);
		peers.push_back(test_labels);

		use_shuffle = true;
		set_total_samples(peers[0]->dimensions[0]/*640*/); //small value is used for debug
		peers.push_back(new type::MiniBatch(size, peers[2]->dimensions[0]/*160*/)); //peers[4]
	}

	void save_as_24bits_bmp(int start, int end, bool is_test_data, string path) {
		auto p = is_test_data? peers[2]->pointer : peers[0]->pointer;
		auto label = is_test_data? peers[3]->pointer : peers[1]->pointer;
		auto height = peers[0]->dimensions[1], width = peers[0]->dimensions[2];
		auto length = height * width * 3;
		auto buffer = new unsigned char[length];
		for (int i = start; i < end; i++) {
			auto data = p + i * length;
			for (int r = height - 1; r >= 0; r--)
				for (int c = 0; c < width; c++) {
					int offset = (r * width + c) * 3;
					buffer[offset] = (unsigned char) (*data++ * 255.0f + 0.5f); //round to integer
					buffer[offset + 1] = (unsigned char) (*data++ * 255.0f + 0.5f);
					buffer[offset + 2] = (unsigned char) (*data++ * 255.0f + 0.5f);
				}
			generate_24bits_bmp(buffer, width, height, (path + to_string(i) + "-" + to_string(static_cast<int>(label[i])) + ".bmp").c_str());
		}
		delete buffer;
	}

	virtual std::string generate_source_code(DeviceInstance& I) override {
        std::string kernel{R"CLC(
kernel void load_cifar_images(global float* data, global float* label, const global float* images, const global float* labels, const global int* index, const int index_offset, const int index_size)
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

T residualBlock(T& data, int k, int filters, int block, const string& name)
{
	const int kernel_size = 3, stride = block != 0 || block == -1? 1 : 2;

	string suffix = name + "_block" + to_string(block == -1? 0 : block);
	T activation0 = ReLU(BatchNormalizedLayer(data, 0.00001f, 0.1f, suffix + "_bn0"));
//	if (filters == 64)
//		return short_connection;
	T conv0 = ConvolutionLayer(activation0, filters * k, kernel_size, stride, "", true, false, suffix + "_conv0");
//	T dropout = DropOut(conv0, 0.3, "dropout");
	T activation1 = ReLU(BatchNormalizedLayer(conv0/*dropout*/, 0.00001f, 0.1f, suffix + "_bn1"));
	T conv1 = ConvolutionLayer(activation1, filters * k, kernel_size, 1, "", true, false, suffix + "_conv1");
	T short_connection = block == 0 || k > 1? ConvolutionLayer(activation0, filters * k, 1, stride, "", true, false, suffix + "_convdim") : data;
//	conv1 += short_connection;
//	return conv1;
	T addition = conv1 + short_connection;
	return addition;
}

T CIFAR_WRN(bool is_predict)
{
	const string cifar_folder = optional<string>("cifar_folder", "D:\\DataSets\\");
	const int batch_size = optional<int>("batch_size", 16); //can run on 4GB memory GPU
	const int class_num = optional<int>("class_num", 10);
	const int N = optional<int>("N", 4);
	const int width = optional<int>("width", 10);
	auto iterator = new CIFARImageIterator(cifar_folder, batch_size, class_num);
//	iterator->save_as_24bits_bmp(420, 660, false, "E:\\Temporary\\temp2\\");

	vector<int64> dims = iterator->peers[0]->dimensions;
	dims[0] = batch_size;
	Tensor* tensor = new Tensor(dims, {}, "train_images_data");
	tensor->dependent_on(iterator);

	//WRN-(6*N+4)-(width): default to WRN-28-10
	tensor = &ConvolutionLayer(*tensor, 16, 3, 1, "", true, false, "conv0"); //[32,32,32,16]
	for (int i = 0; i < N; i++) //group conv1
		tensor = &residualBlock(*tensor, width, 16, i == 0? -1 : i, "group0"); //[32,32,32,160]
	for (int i = 0; i < N; i++) //group conv2
		tensor = &residualBlock(*tensor, width, 32, i, "group1"); //[32,16,16,320]
	for (int i = 0; i < N; i++) //group conv3
		tensor = &residualBlock(*tensor, width, 64, i, "group2"); //[32,8,8,640]
	T activation2 = ReLU(BatchNormalizedLayer(*tensor, 0.00001f, 0.1f, "bn"));
	T pool = Pooling(activation2, {8}, {1}, "average", false, "pool");
	T reshape = Reshape(pool, {pool.dimensions[0], pool.volume / pool.dimensions[0]});
	T inference = FullyConnectedLayer(reshape, class_num, "", "inference");
	if (is_predict)
		return inference;

	const float learning_rate = optional<float>("learning_rate", 0.064), weight_decay = optional<float>("weight_decay", 0.0005), momentum = optional<float>("momentum", 0.9);
	const int max_epochs = optional<int>("max_epochs", 5000);
	T label = *new Tensor({batch_size}, {}, "train_images_label");
	label.dependent_on(iterator);
	T loss = CrossEntropyLoss(inference, label);
	T SGD = StochasticGradientDescentUpdater(loss, learning_rate, weight_decay, momentum);
	vector<Tensor*> parameters;
	for (auto tensor : Tensor::ALL)
		if (dynamic_cast<type::Parameter*>(tensor) != nullptr)
			parameters.push_back(tensor);
	T initializer = GeneralInitializer(parameters);

	auto clnetparams_file = optional<string>("params_file", "D:\\DataSets\\CIFAR_WRN.clnetparams");

	const int N_samples = iterator->peers[0]->dimensions[0];
	auto monitor = new InstantTensor("CIFAR_WRN_monitor", {}, {}, [batch_size, N_samples, &loss, parameters, clnetparams_file](InstantTensor* self, DeviceInstance& I) {
		auto optimizer = static_cast<type::IterativeOptimizer*>(self->peers[0]);
		auto epoch = optimizer->current_epoch(I);

		size_t duration = optimizer->milliseconds_since_last(I);
		string speed = epoch == 0? to_string(duration) + "ms" : to_string(1000.0f * N_samples / duration) + "/s";
		logger << "[" << I.ID << "," << epoch << "," << speed << "] train loss: " << static_cast<back::Loss*>(&loss)->L(I);

//		ofstream ofs(clnetparams_file, ostream::binary);
//		for (size_t i = 0; i < parameters.size(); i++)
//			save_tensor(parameters[i], ofs, &I);
//		ofs.close();
	});
	auto validator = new InstantTensor("CIFAR_WRN_validator", {}, [iterator, class_num, &inference, &label](InstantTensor* self, DeviceInstance& I) {
//		auto optimizer = static_cast<type::IterativeOptimizer*>(self->peers[0]);
//		auto epoch = optimizer->current_epoch(I);
//		if (epoch % 10 != 0)
//			return;

		set<Tensor*> visited;
		auto tester = static_cast<type::MiniBatch*>(iterator->peers[4]);
		auto& offset = reinterpret_cast<int*>(I.pointers[tester])[0];
		int correct = 0, N = inference.dimensions[0];
//		CLNET_TENSOR_GLOBALS |= CLNET_PREDICT_ONLY;
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
//		CLNET_TENSOR_GLOBALS ^= CLNET_PREDICT_ONLY;
		float accuracy = (int) (10000.0f * correct / iterator->peers[2]->dimensions[0]) / 100.0f;
		logger << "\ttest set accuracy: " << accuracy  << "%" << endl;
		offset = -1; //random_shuffle on test set is not needed.
	}, {}, [&inference](InstantTensor* self) -> Tensor*{ return &inference; });
	return IterativeOptimizer({&initializer}, {&SGD, iterator, monitor, validator}, max_epochs);
}
