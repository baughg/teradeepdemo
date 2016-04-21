UNAME_P := $(shell uname -p)

CFLAGS = -O3 -Wall -c -fopenmp -fPIC -I../thnets -I/usr/include/freetype2
LIBS = -lm -lpthread -lfreetype -lopencv_highgui -lopencv_core -lopencv_imgproc
CXX = g++ -g 
CC = gcc -g

ifneq ($(filter aarc% arm%,$(UNAME_P)),)
	CFLAGS += -mfpu=neon -DUSEGLES2
	LIBS += -lEGL -lGLESv2
else
	LIBS += -lGL
endif

.PHONY : all
all : thnetsopencvdemo thnetsstillsdemo

.cpp.o:
	$(CXX) $(CFLAGS) $<

.c.o:
	$(CC) $(CFLAGS) $<

thnetsopencvdemo: $(LIBOBJS) thnetsopencvdemo.o 8cc.o
	$(CXX) -o $@ thnetsopencvdemo.o 8cc.o $(LIBS) -L/usr/local/lib -lthnets

thnetsstillsdemo: $(LIBOBJS) thnetsstillsdemo.o 8cc.o
	$(CXX) -o $@ thnetsstillsdemo.o 8cc.o $(LIBS) -L/usr/local/lib -lthnets

.PHONY : clean
clean :
	rm -f *.o thnetsopencvdemo thnetsstillsdemo
