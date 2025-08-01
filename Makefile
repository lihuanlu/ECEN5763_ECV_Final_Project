INCLUDE_DIRS = -I/usr/include/opencv4
LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS= -O0 -g $(INCLUDE_DIRS) $(CDEFS)
LIBS= -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lrt -ltesseract -llept -lpthread

HFILES= 
CFILES= license_plate.cpp cam_plate.cpp

SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.cpp=.o}

all:	license_plate cam_plate

clean:
	-rm -f *.o *.d
	-rm -f license_plate cam_plate
	-rm -f *.jpg

license_plate: license_plate.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4 tesseract lept` $(LIBS)
	
cam_plate: cam_plate.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4 tesseract lept` $(LIBS)


depend:

.cpp.o: $(SRCS)
	$(CC) $(CFLAGS) -c $<
