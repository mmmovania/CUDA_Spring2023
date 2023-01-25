CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -l OpenCL

SOURCE = InitDataOnGPU.c
SOURCE = AddTwoNumberOnGPU.c

$(SOURCE:.c=): $(SOURCE)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run: $(SOURCE:.c=)
	./$^

clean:
	rm -f $(SOURCE:.c=) *.o *.out *.exe