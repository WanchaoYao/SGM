BIN = spsstereo

CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall -O3 -msse4.2
NVCCFLAGS = -Xcompiler "-Wall -O3 -msse4.2"
LIBS = /usr/lib/x86_64-linux-gnu/libpng.so

# files
CPP_SOURCES = SPSStereo.cpp spsstereo_main.cpp
CU_SOURCES = SGMStereo.cu
CPP_OBJS = $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS = $(patsubst %.cu, %.o, $(CU_SOURCES))

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) -o $@ $<

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) -o $@ $<

$BIN: $(CPP_OBJS) $(CU_OBJS)
	$(NVCC) -o $(BIN) $(CPP_OBJS) $(CU_OBJS) $(LDFLAGS) $(LIBS)

# You can uncomment line 25-26 and comment line 15-22 \
$BIN: $(CPP_SOURCES) $(CU_SOURCES)\
	$(NVCC) -o $(BIN) $(NVCCFLAGS) $(CPP_SOURCES) $(CU_SOURCES) $(LDFLAGS) $(LIBS)

clean:
	rm -rf $(BIN) *.o
