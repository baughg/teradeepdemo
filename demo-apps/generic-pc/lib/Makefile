UNAME_S := $(shell uname -s)
UNAME_P := $(shell uname -p)
LBITS := $(shell getconf LONG_BIT)

INCLUDE = -I. -I/usr/local/include -I$(HOME)/torch/install/include
LIBOPTS = -shared -L/usr/local/lib/lua/5.1 -L/usr/local/lib
CFLAGS = -O3 -c -fpic -Wall
LINUXCAMERA_FILES = linuxcamera.o videocap.o
LIBFILES = linuxcamera.so
CC = gcc

ifneq ($(filter arm%,$(UNAME_P)),)
	CFLAGS += -mfpu=neon
endif

.PHONY : all
all : $(LIBFILES)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDE) $<

linuxcamera.so : $(LINUXCAMERA_FILES)
	$(CC) $(LINUXCAMERA_FILES) $(LIBOPTS) -o $@

install : $(LIBFILES)
	sudo cp $(LIBFILES) /usr/local/lib/lua/5.1/

uninstall :
	sudo rm /usr/local/lib/lua/5.1/linuxcamera.so

.PHONY : clean
clean :
	rm -f *.o $(LIBFILES)
