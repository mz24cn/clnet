/*
 * character_RNN.cpp
 *
 *  Created on: 2017/5/31
 *      Author: ZhangHua
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>

#include <tensor.hpp>
#include <device_instance.hpp>

using namespace std;
using namespace clnet;

struct CharacterIndexer {
	vector<wchar_t> index_to_character;
	unordered_map<wchar_t, float> character_to_index;
	vector<vector<float>> index_sequences;

	float operator ()(wchar_t c) {
		return character_to_index[c];
	}

	wchar_t operator [](int n) {
		return index_to_character[n];
	}

	void build_character_index_from_file(string file) {
		wstring content;
		if (!read_file_content<wchar_t>(file, content))
			throw runtime_error("failed to open " + file);

		build_character_index(content);
		convert_text_to_sequences(content, '\n');
		sort(index_sequences.begin(), index_sequences.end(), [](const vector<float>& a, const vector<float>& b) { return a.size() < b.size(); });
	}

	void build_character_index(const wstring& content) {
		int n = 1;
		character_to_index['\0'] = 0; //padding character
		index_to_character.push_back(0); //padding character index
		for (auto c : content)
			if (character_to_index.find(c) == character_to_index.end()) {
				character_to_index[c] = n++;
				index_to_character.push_back(c);
			}
	}

	void save_character_index(const string file) {
		wofstream ofs(file, ios::binary);
		if (!ofs)
			throw runtime_error("failed to open " + file);

//		for (size_t i = 1; i < index_to_character.size(); i++)
//			ofs << index_to_character[i] << "=" << i << endl;
//		ofs.close();
		ofs.write(index_to_character.data() + 1, index_to_character.size() - 1);
		ofs.close();
	}

	void load_character_index(const string file) {
		wstring content;
		if (!read_file_content<wchar_t>(file, content))
			throw runtime_error("failed to open " + file);

		int n = 1;
		character_to_index[L'\0'] = 0;
		index_to_character.push_back(L'\0');
		for (auto c : content) {
			character_to_index[c] = (float) n++;
			index_to_character.push_back(c);
		}
	}

	void convert_text_to_sequences(const wstring& content, wchar_t spliter) {
		index_sequences.push_back(vector<float>());
		for (auto c : content)
			if (c == spliter && !index_sequences.back().empty())
				index_sequences.push_back(vector<float>());
			else
				index_sequences.back().push_back(character_to_index[c]);
	}
};

class SentenceIterator : public type::MiniBatch {
	CharacterIndexer indexer;
	size_t sequence_length;

public:
	SentenceIterator(string file, int minibatch) : MiniBatch(minibatch) {
		indexer.build_character_index_from_file(file);

		int N = indexer.index_sequences.size() / batch_size * batch_size; //total used samples
		indexer.index_sequences.resize(N);
		set_total_samples(N);
		sequence_length = indexer.index_sequences.back().size();
	}
	size_t max_sequence_length() {
		return sequence_length;
	}
	size_t vocabulary_size() {
		return indexer.index_to_character.size();
	}
	size_t sequences_size() {
		return indexer.index_sequences.size();
	}
	void save_dictionary(const string file) {
		indexer.save_character_index(file);
	}

	virtual void run(DeviceInstance& I) override {
		int* p = reinterpret_cast<int*>(I.pointers[this]);
		const int current = *p;
		const int* indices = p + 1;

		float *data = I.pointers[peers[0]], *pdata = data;
		for (int i = current * batch_size, end = i + batch_size; i < end; i++) {
			memcpy(pdata, indexer.index_sequences[indices[i]].data(), indexer.index_sequences[indices[i]].size() * sizeof(float));
			if (indexer.index_sequences[indices[i]].size() < sequence_length)
				memset(pdata + indexer.index_sequences[indices[i]].size(), 0, (sequence_length - indexer.index_sequences[indices[i]].size()) * sizeof(float));
			pdata += sequence_length;
		}

		if (CLNET_TENSOR_GLOBALS & CLNET_PREDICT_ONLY)
			return;
		float *label = I.pointers[peers[1]], *plabel = label;
		for (int i = current * batch_size, end = i + batch_size; i < end; i++) {
			memcpy(plabel, indexer.index_sequences[indices[i]].data() + 1, (indexer.index_sequences[indices[i]].size() - 1) * sizeof(float));
			memset(plabel + indexer.index_sequences[indices[i]].size() - 1, 0, (sequence_length - indexer.index_sequences[indices[i]].size() + 1) * sizeof(float)); //it always is zero in charRNN example.
			plabel += sequence_length;
		}
//		memcpy(plabel, data, sequence_length * batch * sizeof(float)); //making label same as data for debug
	}
};

Tensor* predict_charRNN(Tensor& graph, Tensor* lstm_initialzier, CharacterIndexer& indexer)
{
	auto params_file = optional<string>("params_file", OpenCL.location + "data/charRNN.clnetparams");
	auto initializer = new InstantTensor("parameters_initializer", {}, {}, [params_file](InstantTensor* self, DeviceInstance& I) {
		ifstream ifs(params_file, istream::binary);
		if (!ifs)
			throw runtime_error("failed to open " + params_file);
		auto tensors = load_tensors(ifs, &I);
		ifs.close();
		logger << to_string(tensors.size()) << " parameters successfully loaded." << endl;
	});

	auto sample = optional<string>("sample", "The ");
	auto data = locate_tensor("data");
	auto predictor = new InstantTensor("charRNN_predictor", {initializer, lstm_initialzier}, [sample, &graph, &indexer, data](InstantTensor* self, DeviceInstance& I) {
		int N = graph.volume;
		set<Tensor*> visited;
		vector<cl::Event> async;
		for (auto c : sample) {
			I.pointers[data][0] = indexer(c);
			data->download(I, &async);
			visited.clear();
			graph.launch(&visited, &I);
			wait_for_all_kernels_finished(I);
			logger << c;
		}

		int L = 129 * 3;
		for (int i = 0; i < L; i++) {
			graph.upload(I);
			float* output = I.pointers[&graph];
			auto n = max_element(output, output + N) - output;
			auto next = indexer[n];
			logger << char(next); //for English corpus

			I.pointers[data][0] = indexer(next);
			data->download(I, &async);
			visited.clear();
			graph.launch(&visited, &I);
			wait_for_all_kernels_finished(I);
		}
		logger << endl;
	}, {&graph}, [](InstantTensor* self) -> Tensor*{ return self->peers[0]; });

	return predictor;
}

T charRNN(bool is_predict)
{
//	const int num_embedding = 4, num_hidden = 4, batch_size = 2, max_iters = 5000, num_layer = 3; //for debug
//	auto generator = new BucketSentenceIterator(OpenCL.location + "data/obama0.txt", batch_size);
	const int num_embedding = 256, num_hidden = 256, batch_size = is_predict? 1 : 32, max_iters = 5000, num_layer = 3;

	const string corpus_file = optional<string>("corpus_file", OpenCL.location + "data/obama.txt");
	const string index_file = optional<string>("index_file", OpenCL.location + "data/obama.index");
	auto trainer = is_predict? nullptr : new SentenceIterator(corpus_file, batch_size);
	CharacterIndexer* indexer = nullptr;
	if (is_predict) {
		indexer = new CharacterIndexer;
		indexer->load_character_index(index_file);
	}
	const int S_LEN = is_predict? 1 : trainer->max_sequence_length();
	const int V = is_predict? indexer->index_to_character.size() : trainer->vocabulary_size();
	T data = Data({batch_size, S_LEN}, trainer, "data");

	T embedding_matrix = Weight({V, num_embedding}, "embedding_matrix");
	T embedding = Embedding(data, embedding_matrix);
	T lstm = LSTM(embedding, num_layer, num_hidden, /*0.2f*/0);

	T class_weight = Weight({num_hidden, V}, "class_weight");
	T class_bias = Bias({V}, "class_bias");
	T output = FullyConnectedLayer(lstm, class_weight, &class_bias, "", "FC");
	if (is_predict)
		return *predict_charRNN(output, lstm.peers[2], *indexer);

	trainer->save_dictionary(index_file);
	const float learning_rate = optional<float>("learning_rate", 0.02), weight_decay = 0;
	T label = Data({batch_size, S_LEN}, trainer, "label");
	T loss = CrossEntropyLoss(output, label);
	T SGD = StochasticGradientDescentUpdater(loss, learning_rate * S_LEN, weight_decay);

	T initializer = GeneralInitializer(SGD.peers);
	const int N_chars = batch_size * S_LEN, N_samples = trainer->sequences_size();
	auto monitor = new InstantTensor("charRNN_monitor", {}, {}, [N_chars, N_samples, &loss](InstantTensor* self, DeviceInstance& I) {
		auto optimizer = static_cast<type::IterativeOptimizer*>(self->peers[0]);
		auto epoch = optimizer->current_epoch(I);

		float loss_value = static_cast<back::Loss*>(&loss)->L(I);
		size_t duration = optimizer->milliseconds_since_last(I);
		string speed = epoch == 0? to_string(duration) + "ms" : to_string(1000.0f * N_samples / duration) + "/s";
		logger << "[" << I.ID << "," << epoch << "," << speed << "] loss: " << loss_value << " (perplexity: " << exp(-loss_value) << ")" << endl;
	});
//	Gradient(&output)->inputs.push_back(monitor); //compute accuracy for every mini batch
	return IterativeOptimizer({&initializer}, {&SGD, trainer, monitor}, max_iters);
}

// demo: solo function, we need not run in OpenCL.run()
void predictCharRNN(Tensor& predictor, int device_id)
{
	auto& I = DeviceInstance::create(OpenCL.find_devices()[device_id], device_id);

	set<Tensor*> visited;
	predictor.launch(&visited, &I);
}
