add: AddTwoNumberOnGPU.c
	gcc AddTwoNumberOnGPU.c -Wall -o executable -l OpenCL

init: InitDataOnGPU.c
	gcc InitDataOnGPU.c -Wall -o executable -l OpenCL

run: executable
	./executable

clean: executable
	rm executable