#Paper:  Differential Evolution with Ring Sub-Population Architecture for Optimization
 

All the code of the RSDE is contained in "RSDE.cpp" file.

Compilation is simple using gcc/g++:

g++ RSDE.cpp -o RSDE.exe -std=c++11 -O3 -march=corei7-avx -fexpensive-optimizations -fomit-frame-pointer

Please note that the compilation requires support of C++11 standard. 
You may omit everything after "-O3", however, these options give a significant boost on most systems.
This will create RSDE.exe, available for running.
Next, the main optimization loop will be started, writing data to "RSDE_(F)_(DIM).txt", 
where F and DIM are the function number and problem dimension.
