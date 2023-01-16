default:
	@echo "make hello1"
	@echo "make hello2"
	@echo "make hello3"
	@echo "make run"
	@echo "make clean"

GettingStarted: GettingStarted.c
	gcc GettingStarted.c -Wall -o executable

hello1: Hello1.c
	gcc Hello1.c -Wall -o executable

hello2: Hello2.c
	gcc Hello2.c -Wall -o executable -lOpenCL

hello3: Hello3.c
	gcc Hello3.c -Wall -o executable -lOpenCL

run:
	./executable

clean:
	rm executable