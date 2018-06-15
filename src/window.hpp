/**
 *
 * In main, create objects of the following types for the respective purpose:-
 * WindowDraw 	- For taking an input digit from the user by drawing and processing it using the FNN
 * WindowFile 	- For taking an input digit from the file and processing it using the FNN
 *
 * Note:- 	All you have to do is make an object of the class with the following arguments to the constructor.
 * 			You are not supposed to call any other functions of the object from main function. The program will
 * 			terminate once the object has finished its job. The control will never return to main().
 *
 * 	WindowDraw(int* argc, char** argv, FNN& minst);
 * 	WindowFile(int* argc, char** argv, FNN& minst);
 *
 * And it is expected from you that you include FNN.hpp in main.cpp
 */
#ifndef WINDOW_HPP
#define WINDOW_HPP

#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/freeglut.h>
#include <fstream>
#include <iostream>
#include <vector>

#ifndef FNN_HPP
#include "FNN.hpp"
#endif

using namespace std;

const int WIN_WIDTH = 308;
const int WIN_HEIGHT = 348;
int mn, num_of_images, rows, columns;
int m, paint = 0;
unsigned char input_pixel, input_label;
FNN *fnn;

ifstream images_file, labels_file;
vector <float> inputs;

struct Box {
	int x1, x2, y1, y2;
	float r, g, b;
} pixels[28][28], buttons[2];


void read_file(ifstream &f, unsigned char* c) {
	f.read((char *) c, sizeof(unsigned char));	
}

void read_file(ifstream &f, int *i) {
	unsigned char c1, c2, c3, c4;

	f.read((char *) i, sizeof(int));
    c1 = *i & 255;
    c2 = (*i >> 8) & 255;
    c3 = (*i >> 16) & 255;
    c4 = (*i >> 24) & 255;
	*i = ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + ((int)c4);
}

class WindowFile {
private:
public:
	WindowFile(int*, char**, FNN&);
};

void update_pixels() {
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			pixels[i][j].r = pixels[i][j].g = pixels[i][j].b = inputs[i * 28 + j];
		}
	}
}

void init_boxes() {
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			pixels[i][j].x1 = j * 11;
			pixels[i][j].x2 = (j + 1) * 11;
			pixels[i][j].y1 = i * 11 + 40;
			pixels[i][j].y2 = (i + 1) * 11 + 40;
			pixels[i][j].r = pixels[i][j].g = pixels[i][j].b = 0;
		}
	}
	buttons[0].x1 = 0;
	buttons[0].x2 = 40;
	buttons[0].y1 = 0;
	buttons[0].y2 = 40;
	buttons[0].r = 1; buttons[0].g = buttons[0].b = 0;

	buttons[1].x1 = 268;
	buttons[1].x2 = 308;
	buttons[1].y1 = 0;
	buttons[1].y2 = 40;
	buttons[1].g = 1; buttons[1].r = buttons[1].b = 0;
}

void print_pixels() {
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			glBegin(GL_POLYGON);
				glColor3f(pixels[i][j].r, pixels[i][j].g, pixels[i][j].b);
				glVertex2i(pixels[i][j].x1, pixels[i][j].y1);
				glVertex2i(pixels[i][j].x1, pixels[i][j].y2);
				glVertex2i(pixels[i][j].x2, pixels[i][j].y2);
				glVertex2i(pixels[i][j].x2, pixels[i][j].y1);
			glEnd();
		}
	}
}

void print_buttons() {
	for (int i = 0; i < 2; i++) {
		glBegin(GL_POLYGON);
			glColor3f(buttons[i].r, buttons[i].g, buttons[i].b);
			glVertex2i(buttons[i].x1, buttons[i].y1);
			glVertex2i(buttons[i].x1, buttons[i].y2);
			glVertex2i(buttons[i].x2, buttons[i].y2);
			glVertex2i(buttons[i].x2, buttons[i].y1);
		glEnd();
	}
	//draw X
	glColor3f(1, 1, 1);
	glLineWidth(5);
	glBegin(GL_LINES);
		glVertex2i(10, 10);
		glVertex2i(30, 30);
	glEnd();
	glBegin(GL_LINES);
		glVertex2i(30, 10);
		glVertex2i(10, 30);
	glEnd();

	//draw ->
	glBegin(GL_LINES);
		glVertex2i(278, 20);
		glVertex2i(298, 20);
	glEnd();
	glBegin(GL_LINES);
		glVertex2i(288, 10);
		glVertex2i(298, 20);
	glEnd();
	glBegin(GL_LINES);
		glVertex2i(288, 30);
		glVertex2i(298, 20);
	glEnd();
	glLineWidth(1);
}

void show() {
	glClear(GL_COLOR_BUFFER_BIT);
	print_pixels();
	print_buttons();
	glFlush();
}

void check_clicked_pixels(int x, int y) {
	for (int i = 0; i < 28; i++) {
		for (int j = 0; j < 28; j++) {
			if (x >= pixels[i][j].x1 and x <= pixels[i][j].x2 and y >= pixels[i][j].y1 and y <= pixels[i][j].y2) {
				pixels[i][j].r = pixels[i][j].g = pixels[i][j].b = min(pixels[i][j].r + 0.6, (double)1);
				for (int a = i - 1; a <= i + 1; a++) {
					for (int b = j - 1; b <= j + 1; b++) {
						if (a != i or b != j) {
							pixels[a][b].r = pixels[a][b].g = pixels[a][b].b = min(pixels[a][b].r + 0.07, (double)1);
						}
					}
				}
			}
		}
	}
}

void click(int button, int state, int x, int y) {
	if(state == GLUT_DOWN) {
        if (x >= buttons[0].x1 and x <= buttons[0].x2 and y >= buttons[0].y1 and y <= buttons[0].y2) {
        	if (m) {
        		init_boxes();
        		buttons[0].r /= 2;
        		glutPostRedisplay();
        	} else {
        		buttons[0].r /= 2;
        		glutPostRedisplay();
        		glutLeaveMainLoop();
        	}
        } else if (x >= buttons[1].x1 and x <= buttons[1].x2 and y >= buttons[1].y1 and y <= buttons[1].y2) {
        	buttons[1].g /= 2;
        	glutPostRedisplay();
        	if (m) {
        		inputs.clear();
        		for (int i = 0; i < 28; i++) {
        			for (int j = 0; j < 28; j++) {
    					inputs.push_back(pixels[i][j].r);
        			}
        		}
        		fnn->forward_prop(inputs);
				printf("I think that looks like a %d\n\n", fnn->output());
				fflush(stdout);
        	} else {
        		inputs.clear();
        		for (int i = 0; i < 28; i++) {
        			for (int j = 0; j < 28; j++) {
        				read_file(images_file, &input_pixel);
        				inputs.push_back(((float)input_pixel) / 255);
        			}
        		}
        		read_file(labels_file, &input_label);
        		update_pixels();
        		fnn->forward_prop(inputs);
        		printf("That looks like a %d to me\n", fnn->output());
        		printf("Database says it's a %d\n\n", ((int)input_label));
        		fnn->print_last_layer();
        		fflush(stdout);
        		glutPostRedisplay();
        	}
    	} else {
    		paint = 1;
    	}
    }
    if(state == GLUT_UP) {
    	paint = 0;
        buttons[0].r = 1; buttons[1].g = 1;
        glutPostRedisplay();
    }
}

WindowFile::WindowFile(int *argc, char** argv, FNN& mnist) {
	fnn = &mnist;
	images_file.open(".././data/t10k-images-idx3-ubyte", ios::binary);
	labels_file.open(".././data/t10k-labels-idx1-ubyte", ios::binary);
	read_file(images_file, &mn);
	read_file(labels_file, &mn);
	read_file(images_file, &num_of_images);
	read_file(labels_file, &num_of_images);
	read_file(images_file, &rows);
	read_file(images_file, &columns);
	init_boxes();
	m = 0;
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(WIN_WIDTH, WIN_HEIGHT);
	glutInitWindowPosition(640, 320);
	glutCreateWindow("MNIST Inputs");
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0f, WIN_WIDTH, WIN_HEIGHT, 0.0f, 0.0f, 1.0f);
	glutDisplayFunc(show);
	glutMouseFunc(click);
	glutMainLoop();
}


class WindowDraw {
private:
public:
	WindowDraw(int*, char**, FNN&);
};

void paint_func(int x, int y) {
	if (paint) {
		check_clicked_pixels(x, y);
		glutPostRedisplay();
	}
}

WindowDraw::WindowDraw(int *argc, char** argv, FNN& mnist) {
	fnn = &mnist;
	init_boxes();
	m = 1;
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
	glutInitWindowSize(WIN_WIDTH, WIN_HEIGHT);
	glutInitWindowPosition(640, 320);
	glutCreateWindow("Draw");
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0f, WIN_WIDTH, WIN_HEIGHT, 0.0f, 0.0f, 1.0f);
	glutDisplayFunc(show);
	glutMouseFunc(click);
	glutMotionFunc(paint_func);
	glutMainLoop();
}

#endif