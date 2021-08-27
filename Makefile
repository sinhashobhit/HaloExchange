CC=mpicc #The compiler we are using
TARGET=halo #The name of the executable
all:
	$(CC) -o $(TARGET) src.c -lm

clean:
	rm $(TARGET)
