/**
 * Supports a maximum of 22 layers (including input and output layers).
 * 
 * NOTE:- The FNN does not save the updated weights and biases automatically after every batch is done
 * It leaves that to the main caller funciton. You have to call the FNN::save() function whenever you
 * wish to write the updated weights and biases. This will allow you to control what value is being written
 * so that weights and biases changes are not written when, for example, say the gradient explodes.
 */

#ifndef FNN_HPP
#define FNN_HPP
#endif

#include "cuda.hpp"
#include <vector>
#include <string>
#include <time.h>
#include <fstream>
#include <math.h>
#include <stack>

using namespace std;



class FNN {
private:
	int number_of_layers;
	vector<int> layers;
	int epochs, batch_size, epoch_dropout, samples_done;
	float learning_rate, decay_rate;
	int weights, biases;
	string file_name;
	ifstream file_in; ofstream file_out;
	float cost;
	bool cuda_gpu_found;

	//NN parameters. All are declared in the unified memory of GPU and CPU
	float *w[21], *b[21], *a[23], *dw[21], *db[21], *da[23], *o;

public:
	FNN(void);
	~FNN(void);
	void init(vector<int>, int, int, int, float, float, string);
	bool load(void);
	void save(void);
	void xavier_initialize(void);
	void allocate(void);
	string get_layers_sizes_string(void);
	int get_layers(void);
	float get_cost(void);
	int output(void);
	void forward_prop(vector<float>&);
	void backprop(int, int);
	void backprop(float*, int);
	void print_last_layer(void);
	string int_to_string(int);
	bool good(void);
	float cross_entropy(float y, float y_cap);
	bool NaN();
};

FNN::FNN() {
	cuda_gpu_found = cuda_init();
	cost = 0; samples_done = 0;
}

FNN::~FNN() {
	for (int i = 0; i < number_of_layers - 1; i++) {
		cuda_delete(&w[i]);
		cuda_delete(&dw[i]);
		cuda_delete(&b[i]);
		cuda_delete(&db[i]);
		cuda_delete(&a[i]);
		cuda_delete(&da[i]);
	}
	for (int i = number_of_layers - 1; i <= number_of_layers; i++) {
		cuda_delete(&a[i]);
		cuda_delete(&da[i]);
	}
	cuda_delete(&o);
	cuda_destroy();
}

bool FNN::good() {
	return cuda_gpu_found;
}

string FNN::int_to_string(int i) {
	string res = "";
	stack<char> temp;
	while (i) {
		temp.push((char)(i % 10) + '0');
		i /= 10;
	}
	while (!temp.empty()) {
		res = res + temp.top();
		temp.pop();
	}
	return res;
}

int FNN::output() {
	int max = 0;
	for (int i = 0; i < layers[number_of_layers - 1]; i++) {
		if (a[number_of_layers][i] > a[number_of_layers][max]) {
			max = i;
		}
	}
	return max;
}

float FNN::get_cost() {
	return ((float)(cost / batch_size));
}

int FNN::get_layers() {
	return number_of_layers;
}

string FNN::get_layers_sizes_string() {
	string output = "";
	for (int i = 0; i < number_of_layers - 1; i++) {
		output = output + int_to_string(layers[i]) + ", ";
	}
	output = output + int_to_string(layers[number_of_layers - 1]);
	return output;
}

void FNN::allocate() {
	for (int i = 1; i < number_of_layers; i++) {
		cuda_allocate(&w[i - 1], layers[i] * layers[i - 1]);
		cuda_allocate(&dw[i - 1], layers[i] * layers[i - 1]);
	}
	for (int i = 1; i < number_of_layers; i++) {
		cuda_allocate(&b[i - 1], layers[i]);
		cuda_allocate(&db[i - 1], layers[i]);
	}
	for (int i = 0; i < number_of_layers; i++) {
		cuda_allocate(&a[i], layers[i]);
		cuda_allocate(&da[i], layers[i]);
	}
	cuda_allocate(&a[number_of_layers], layers[number_of_layers - 1]);
	cuda_allocate(&da[number_of_layers], layers[number_of_layers - 1]);
	cuda_allocate(&o, layers[number_of_layers - 1]);
}

void FNN::xavier_initialize() {
	srand(time(NULL));
	for (int i = 0; i < number_of_layers - 1; i++) {
		float standard_deviation = sqrt((float)1 / layers[i]);
		int temp = layers[i] * layers[i + 1];
		for (int j = 0; j < temp; j++) {
			int random;
			do {
				random = rand() % ((int) (standard_deviation * 200000));
			} while (random == 0);
			w[i][j] = random;
			w[i][j] -= ((int)(standard_deviation * 100000));
			w[i][j] = ((float)(w[i][j] / 100000));
		}
	}
	for (int i = 0; i < number_of_layers - 1; i++) {
		for (int j = 0; j < layers[i + 1]; j++) {
			b[i][j] = 0;	
		}
	}
}

bool FNN::load() {
	file_in.open(file_name, ios::binary);
	if (!file_in.good()) {
		return 0;
	}
	file_in.read((char *) &number_of_layers, sizeof(number_of_layers));
	layers.clear();
	for (int i = 0; i < number_of_layers; i++) {
		int temp;
		file_in.read((char *) &temp, sizeof(temp));
		layers.push_back(temp);
	}
	file_in.read((char *) &epochs, sizeof(epochs));
	file_in.read((char *) &batch_size, sizeof(batch_size));
	file_in.read((char *) &epoch_dropout, sizeof(epoch_dropout));
	file_in.read((char *) &learning_rate, sizeof(learning_rate));
	file_in.read((char *) &decay_rate, sizeof(decay_rate));
	allocate();
	
	for (int i = 1; i < number_of_layers; i++) {
		int temp = layers[i] * layers[i - 1];
		for (int j = 0; j < temp; j++) {
			file_in.read((char *) &w[i - 1][j], sizeof(w[i - 1][j]));
		}
	}

	for (int i = 1; i < number_of_layers; i++) {
		for (int j = 0; j < layers[i]; j++) {
			file_in.read((char *) &b[i - 1][j], sizeof(b[i - 1][j]));
		}
	}
	file_in.close();
	return 1;
}

void FNN::save() {
	remove(file_name.c_str());
	file_out.open(file_name, ios::binary);

	file_out.write((char *) &number_of_layers, sizeof(number_of_layers));
	for (int i = 0; i < number_of_layers; i++) {
		int temp = layers[i];
		file_out.write((char *) &temp, sizeof(temp));
	}
	file_out.write((char *) &epochs, sizeof(epochs));
	file_out.write((char *) &batch_size, sizeof(batch_size));
	file_out.write((char *) &epoch_dropout, sizeof(epoch_dropout));
	file_out.write((char *) &learning_rate, sizeof(learning_rate));
	file_out.write((char *) &decay_rate, sizeof(decay_rate));
	
	for (int i = 1; i < number_of_layers; i++) {
		int temp = layers[i] * layers[i - 1];
		for (int j = 0; j < temp; j++) {
			file_out.write((char *) &w[i - 1][j], sizeof(w[i - 1][j]));
		}
	}

	for (int i = 1; i < number_of_layers; i++) {
		for (int j = 0; j < layers[i]; j++) {
			file_out.write((char *) &b[i - 1][j], sizeof(b[i - 1][j]));
		}
	}

	file_out.close();
}

void FNN::init(vector<int> layers, int epochs, int batch_size, int epoch_dropout, 
               float learning_rate, float decay_rate, string file_name) {
	this->file_name = file_name;
	if (!load()) {
		this->layers.clear();
		this->layers = layers;
		this->number_of_layers = layers.size();
		this->epochs = epochs;
		this->batch_size = batch_size;
		this->epoch_dropout = epoch_dropout;
		this->learning_rate = learning_rate;
		this->decay_rate = decay_rate;
		allocate();
		xavier_initialize();
		save();
	}
}

void FNN::forward_prop(vector<float> &ip) {
	for (int i = 0; i < layers[0]; i++) {
		a[0][i] = ip[i];
	}
	for (int i = 1; i < number_of_layers; i++) {
		cuda_matrix_mul(&a[i], &w[i - 1], &a[i - 1], layers[i], layers[i - 1], 1);
		cuda_matrix_add(&a[i], &a[i], &b[i - 1], layers[i], 1);
		cuda_ReLU(&a[i], &a[i], layers[i]);
	}
	cuda_softmax(&a[number_of_layers], &a[number_of_layers - 1], layers[number_of_layers - 1]);
}

void FNN::backprop(int op, int e) {
	float *output = new float[layers[number_of_layers - 1]];
	for (int i = 0; i < layers[number_of_layers - 1]; i++) {
		output[i] = 0;
	}
	output[op] = 1;
	backprop(output, e);
}

void FNN::print_last_layer() {
	printf("Last softmax layer looks like this:-\n");
	for (int i = 0; i < layers[number_of_layers - 1]; i++) {
		printf("%f\n", a[number_of_layers][i]);
	}
	printf("\n");
}

float FNN::cross_entropy(float y_cap, float y) {
	return -((y * log(y_cap)) + ((1 - y) * log(1 - y_cap)));
}

void FNN::backprop(float *op, int e) {
	cuda_copy_to_device(&o, &op, layers[number_of_layers - 1]);
	samples_done = (samples_done + 1) % batch_size;
	if (samples_done == 1) {
		cost = 0;
	}
	for (int i = 0; i < layers[number_of_layers - 1]; i++) {
		cost += cross_entropy(a[number_of_layers][i], op[i]);
	}
	cuda_cross_entropy_prime(&da[number_of_layers], &a[number_of_layers], &o, layers[number_of_layers - 1]);
	cuda_softmax_prime(&da[number_of_layers - 1], &a[number_of_layers], 
                        &da[number_of_layers], layers[number_of_layers - 1]);
	for (int i = number_of_layers - 1; i > 0; i--) {
		cuda_ReLU_prime_biases(&db[i - 1], &a[i], &da[i], layers[i]);
		cuda_ReLU_prime_others(&da[i - 1], &dw[i - 1], &w[i - 1], &a[i - 1], &a[i], &da[i], layers[i - 1], layers[i]);
	}
	if (samples_done == 0) {
		float lr = learning_rate * pow(decay_rate, (floor(e / epoch_dropout)));
		for (int i = 0; i < number_of_layers - 1; i++) {
			cuda_gradient_descent_step(&w[i], &dw[i], lr, batch_size, (layers[i] * layers[i + 1]));
			cuda_gradient_descent_step(&b[i], &db[i], lr, batch_size, layers[i + 1]);
		}
	}
}


#ifndef WINDOW_HPP
#include "window.hpp"
#endif
