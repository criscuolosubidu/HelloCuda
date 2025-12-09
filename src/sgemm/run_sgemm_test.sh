!# /bin/bash

nvcc -O3 main.cu -o sgemm_test
nsys profile -o report_sgemm .\sgemm_test
ncu --set full -o report_sgemm -f .\sgemm_test