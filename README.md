# Digit Recognizer

Handwritten digit recognition ML program written in C++ from scratch with CUDA for parallel computations. The FNN.hpp is an 
extremely general implementation. One can move the three files, "FNN.hpp", "cuda.hpp" and "cuda.cu" to their project, include the
"FNN.hpp" and create an FNN object and use it. Just remove the last 3 lines from "FNN.hpp". It is the only bit of code which is
this project specific.

### Requirements
	1. Linux OS
	2. OpenGL library       (mesa-utils and freeglut3-dev)
	3. g++
	4. Nvidia GPU 	        (sorry ._.)
	5. nvcc                 (if you wish to make a new build)
If you don't have OpenGL or nvcc you can still use this program using the pre-built binary file provided in the /bin folder.
However you will still need an Nvidia GPU to run the program.

### Build

1. Right click the build.sh file. Click on "Properties". Go to "Permissions" tab and make sure the checkbox reading "Allow
executing file as a program" is checked.

2. Right click the empty_build.sh file. Click on "Properties". Go to "Permissions" tab and make sure the checkbox reading "Allow
executing file as a program" is checked.

3. In the terminal (navigated to the project directory) type 
	```
	./build.sh
	```
        
to build the project. You will find the produced binary in the "bin" folder.

4. To empty the bin folder type in the terminal (navigated to the project directory) 
	```
	./empty_build.sh
	```
	
### Use

Launch a terminal naviagted to the bin folder in this project. To launch the program, enter
```
./recog [options]
```

Check "options_help.txt" file in the "docs" folder of the project for a list of options and their meanings. Or just type 
```
./recog
```
One neural network is provided in the saved_FNN file of 2 hidden layers of 200 and 60 neurons each having an accuracy of
95.840004% named "mnist.fnn". It is the default choice of the program so if you don't specify the name of the FNN to be used,
that is the one which will be used by the program.
