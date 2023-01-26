CC = gcc
CFLAGS = -Wall -Wextra
LDFLAGS = -l OpenCL

SOURCE = GettingStarted.c
SOURCE = Hello1.c
SOURCE = Hello2.c
SOURCE = Hello3.c


$(SOURCE:.c=): $(SOURCE)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run: $(SOURCE:.c=)
	./$^

clean:
	rm -f $(SOURCE:.c=) *.o *.out *.exe