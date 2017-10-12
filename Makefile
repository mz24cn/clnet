all:
	mkdir -p ./build/{examples,src}
	g++ -std=c++1y -m64 -I./include -I./providers/NVIDIA/include -O3 -fopenmp -Wall -c -fmessage-length=0 -Wdeprecated-declarations -o ./build/src/kernels.o ./src/kernels.cpp 
	g++ -std=c++1y -m64 -I./include -I./providers/NVIDIA/include -O3 -fopenmp -Wall -c -fmessage-length=0 -Wdeprecated-declarations -o ./build/src/debugger.o ./src/debugger.cpp 
	g++ -std=c++1y -m64 -I./include -I./providers/NVIDIA/include -O3 -fopenmp -Wall -c -fmessage-length=0 -Wdeprecated-declarations -o ./build/src/tensor.o ./src/tensor.cpp 
	g++ -std=c++1y -m64 -I./include -I./providers/NVIDIA/include -O3 -fopenmp -Wall -c -fmessage-length=0 -Wdeprecated-declarations -o ./build/src/device_instance.o ./src/device_instance.cpp 
	g++ -std=c++1y -m64 -I./include -I./providers/NVIDIA/include -O3 -fopenmp -Wall -c -fmessage-length=0 -Wdeprecated-declarations -o ./build/examples/main.o ./examples/main.cpp 
	g++ -std=c++1y -m64 -I./include -I./providers/NVIDIA/include -O3 -fopenmp -Wall -c -fmessage-length=0 -Wdeprecated-declarations -o ./build/examples/reference.o ./examples/reference.cpp 
	g++ -std=c++1y -m64 -I./include -I./providers/NVIDIA/include -O3 -fopenmp -Wall -c -fmessage-length=0 -Wdeprecated-declarations -o ./build/examples/character_RNN.o ./examples/character_RNN.cpp 
	g++ -std=c++1y -m64 -I./include -I./providers/NVIDIA/include -O3 -fopenmp -Wall -c -fmessage-length=0 -Wdeprecated-declarations -o ./build/examples/image_io.o ./examples/image_io.cpp 
	g++ -std=c++1y -m64 -I./include -I./providers/NVIDIA/include -O3 -fopenmp -Wall -c -fmessage-length=0 -Wdeprecated-declarations -o ./build/examples/MNIST_CNN.o ./examples/MNIST_CNN.cpp 
	g++ -std=c++1y -m64 -I./include -I./providers/NVIDIA/include -O3 -fopenmp -Wall -c -fmessage-length=0 -Wdeprecated-declarations -o ./build/examples/multi_layer_perception.o ./examples/multi_layer_perception.cpp 
	g++ -std=c++1y -m64 -I./include -I./providers/NVIDIA/include -O3 -fopenmp -Wall -c -fmessage-length=0 -Wdeprecated-declarations -o ./build/examples/kernel_test.o ./examples/kernel_test.cpp 
	g++ -L./providers/NVIDIA/lib64 -o ./build/OpenCLNet ./build/examples/MNIST_CNN.o ./build/examples/character_RNN.o ./build/examples/image_io.o ./build/examples/main.o ./build/examples/multi_layer_perception.o ./build/examples/reference.o ./build/examples/kernel_test.o ./build/src/debugger.o ./build/src/device_instance.o ./build/src/kernels.o ./build/src/tensor.o -lOpenCL -lgomp 

clean:
	rm ./build -R -f