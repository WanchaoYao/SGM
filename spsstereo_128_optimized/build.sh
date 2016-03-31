#!/bin/sh
nvcc -o spsstereo -Xcompiler "-Wall -O3 -msse4.2" SGMStereo.cu SPSStereo.cpp spsstereo_main.cpp /usr/lib/x86_64-linux-gnu/libpng.so
