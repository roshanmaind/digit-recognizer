#ifndef FNN_HPP
#include "FNN.hpp"
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <assert.h>
#include <math.h>


using namespace std;

string cmd_argument, mode = "none", file_name = ".././saved_FNN/mnist.fnn";
int epochs = 100, batch_size = 100, epoch_dropout = 3;
float learning_rate = 0.3, decay_rate = 0.5;
vector<int> layers;
ifstream images, labels;
FNN mnist;

int magic_number, number_of_images, image_rows, image_columns;
int progress;

void describe_options() {
	char ch;
	ifstream options_file(".././docs/options_help.txt");
	while (!options_file.eof()) {
		options_file.get(ch);
		printf("%c", ch);
	}
	options_file.close();
}

void get_arguments(int argc, char ***argv) {
	if (argc == 1) {
		describe_options();
		return;
	}
	
	for (int i = 1; i < argc; i++) {
		cmd_argument = (*argv)[i];
		if (cmd_argument == "-mode") {
			if (i + 1 < argc) {
				mode = (*argv)[i + 1];
			}
			continue;
		}
		if (cmd_argument == "-e") {
			if (i + 1 < argc) {
				epochs = atoi((*argv)[i + 1]);
			}
			continue;
		}
		if (cmd_argument == "-b") {
			if (i + 1 < argc) {
				batch_size = atoi((*argv)[i + 1]);
			}
			continue;
		}
		if (cmd_argument == "-l") {
			if (i + 1 < argc) {
				learning_rate = atof((*argv)[i + 1]);
			}
			continue;
		}
		if (cmd_argument == "-decay") {
			if (i + 1 < argc) {
				decay_rate = atof((*argv)[i + 1]);
			}
			continue;
		}
		if (cmd_argument == "-dropout") {
			if (i + 1 < argc) {
				epoch_dropout = atoi((*argv)[i + 1]);
			}
			continue;
		}
		if (cmd_argument == "-layers") {
			if (i + 1 < argc) {
				int l_num = atoi((*argv)[i + 1]);
				if (i + 1 + l_num < argc and l_num <= 20 and l_num > 0) {
					layers.clear();
					layers.push_back(784);
					for (int j = 1; j <= l_num; j++) {
						layers.push_back(atoi((*argv)[i + j + 1]));
					}
					layers.push_back(10);
				}
			}
			continue;
		}
		if (cmd_argument == "-name") {
			if (i + 1 < argc) {
				file_name = (*argv)[i + 1];
				file_name = ".././saved_FNN/" + file_name + ".fnn";
			}
			continue;
		}
	}
}

void update_progress(int so_far, int total) {
	int perc = ((double) so_far / total) * 100;
	if (perc != progress) {
		if (progress < 10) {
			printf("\b\b");
		} else {
			printf("\b\b\b");
		}
		progress = perc;
		printf("%d%%", progress);
	}
}

void read(ifstream &f, unsigned char* c) {
	f.read((char *) c, sizeof(unsigned char));
}

void read(ifstream &f, int *i) {
	unsigned char c1, c2, c3, c4;

	f.read((char *) i, sizeof(int));
    c1 = *i & 255;
    c2 = (*i >> 8) & 255;
    c3 = (*i >> 16) & 255;
    c4 = (*i >> 24) & 255;
	*i = ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + ((int)c4);
}

void train_model() {
	vector<float> input;
	unsigned char label, pixel;
	
	printf("Initiating training with following hyperparameters:-\n");
	cout << "File path: " << file_name << "\n";
	printf("Layers: %d\n", mnist.get_layers());
	cout << "Layers' sizes: " << mnist.get_layers_sizes_string() << "\n";
	printf("Epochs: %d\n", epochs);
	printf("Batch size: %d\n", batch_size);
	printf("Epochs dropout: %d\n", epoch_dropout);
	printf("Learning rate: %f\n", learning_rate);
	printf("Learning rate decay factor: %f\n\n", decay_rate);
	fflush(stdout);

	for (int e = 0; e < epochs; e++) {
		progress = 0;
		printf("Epoch #%d: Progress -> 0%%", e);
		fflush(stdout);

		images.open(".././data/train-images.idx3-ubyte", ios::binary);
		labels.open(".././data/train-labels-idx1-ubyte", ios::binary);

		read(images, &magic_number);
		read(labels, &magic_number);
		read(images, &number_of_images);
		read(labels, &number_of_images);
		read(images, &image_rows);
		read(images, &image_columns);

		for (int i = 0; i < number_of_images; i++) {
			input.clear();
			for (int x = 0; x < image_rows; x++) {
				for (int y = 0; y < image_columns; y++) {
					read(images, &pixel);
					input.push_back(((float)pixel) / 255);		//normalizing the input before feeding
					//printf("%f\n", ((float)pixel) / 255);
				}
			}
			read(labels, &label);
			mnist.forward_prop(input);
			mnist.backprop((int) label, e);
			update_progress(i + 1, number_of_images);
			fflush(stdout);
		}

		images.close();
		labels.close();
		printf("\nCost: %f\n\n", mnist.get_cost());
		if (mnist.get_cost() == 2.000000) {
			printf("Something probably became NaN. Last working version was saved. Terminating.\n");
			break;
		}
		mnist.save();
	}

	printf("Training complete!\n");
}

void test_all() {
	vector<float> input;
	unsigned char label, pixel;

	progress = 0;
	cout << "Testing 10k MNIST database images on " << file_name << "\n\n";
	printf("Progress -> 0%%");
	fflush(stdout);

	images.open(".././data/t10k-images-idx3-ubyte", ios::binary);
	labels.open(".././data/t10k-labels-idx1-ubyte", ios::binary);

	read(images, &magic_number);
	read(labels, &magic_number);
	read(images, &number_of_images);
	read(labels, &number_of_images);
	read(images, &image_rows);
	read(images, &image_columns);

	int pass = 0;
	for (int i = 0; i < number_of_images; i++) {
		input.clear();
		for (int x = 0; x < image_rows; x++) {
			for (int y = 0; y < image_columns; y++) {
				read(images, &pixel);
				input.push_back(((float)pixel) / 255);		//normalizing the input before feeding
			}
		}
		read(labels, &label);
		mnist.forward_prop(input);
		if (((int)label) == mnist.output()) {
			pass++;
		}
		update_progress(i + 1, number_of_images);
		fflush(stdout);
	}

	images.close();
	labels.close();

	printf("\nTest complete!\n");
	printf("Guessed images with an accuracy of %f%%\n", ((float) pass / number_of_images) * 100);
}

int main(int argc, char **argv) {
	//assigning default data to layers in case user doesn't provide it.
	layers.clear();
	layers.push_back(784);
	layers.push_back(200);
	layers.push_back(60);
	layers.push_back(10);
	
	get_arguments(argc, &argv);
	mnist.init(layers, epochs, batch_size, epoch_dropout, learning_rate, decay_rate, file_name);
	if (!mnist.good()) {
		printf("No CUDA GPU was found on the machine. Exiting...\n");
		return 0;
	}
	if (mode == "train") {
		train_model();
	} else if (mode == "test_all") {
		test_all();
	} else if (mode == "test_single") {
		WindowFile win(&argc, argv, mnist);
	} else if (mode == "test_draw") {
		WindowDraw win(&argc, argv, mnist);
	} else if (mode == "none") {
		
	} else {
		printf("Error: Invalid mode. Check documentation.\n");
	}
	return 0;
}
