rm -rf objects
rm -rf bin
mkdir objects
mkdir bin
cd ./src
for i in *.cpp; do g++ -c $i -o `basename $i .cpp`.o -lGL -lglut -lGLU; done
for i in *.cu; do nvcc -c $i -o `basename $i .cu`.o -lcublas; done
for i in ./*.o; do mv $i .././objects/; done
rm -f *.o
cd .././objects/
nvcc ./*.o -o .././bin/recog -lGL -lglut -lcublas
cd ..