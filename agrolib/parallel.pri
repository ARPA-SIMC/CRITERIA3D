#Enable the openMP parallelization.
CONFIG += OMP_CONFIG

#Enable the link time optimization for the entire project.
#CONFIG += LTO_CONFIG

#Enable SF3D GPU acceleration. If a CUDA Toolkit is not present in the system this flag will be ignored.
#CONFIG += CUDA_CONFIG
CUDA_ARCH = sm_61

#Enable SF3D '.mat' log functionality. MCR_path needs to be manually setted
# CONFIG += MCR_CONFIG
MCR_PATH = "SET Path!"

include($$absolute_path(./parallelDetails.pri))

