#-----------------------------------------------------
#
#   soilFluxes3D library
#
#   Numerical solution for flow equations
#   of water and heat in the soil
#   in a three-dimensional domain
#
#   This project is part of CRITERIA3D distribution
#
#-----------------------------------------------------

QT -= gui

QMAKE_CXXFLAGS += -openmp:llvm -openmp:experimental
QMAKE_LFLAGS += -openmp:llvm -NODEFAULTLIB:msvcrt.lib -NODEFAULTLIB:cmt.lib -IGNORE:4217

TEMPLATE = lib
CONFIG += staticlib
CONFIG += c++20
CONFIG += debug_and_release

INCLUDEPATH += ../mathFunctions

SOURCES += \
    cpusolver.cpp \
    heat_new.cpp \
    otherFunctions.cpp \
    soilFluxes3D_new.cpp \
    soil_new.cpp \
    water_new.cpp \

HEADERS += \
    cpusolver.h \
    heat_new.h \
    logFunctions.h \
    macro.h \
    otherFunctions.h \
    soilFluxes3D_new.h \
    soil_new.h \
    solver_new.h \
    types_cpu.h \
    types_opt.h \
    water_new.h \

DISTFILES += \
    #

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/soilFluxes3D
    } else {
        TARGET = release/soilFluxes3D
    }
}
win32:{
    TARGET = soilFluxes3D
}

#CONFIG += CUDA_CONFIG

CONFIG(CUDA_CONFIG) {
    DEFINES += CUDA_ENABLED
    HEADERS += \
        cudaFunctions.h \
        gpuEntryPoints.h \
        gpusolver.h \
        types_gpu.h \

    # CUDA settings
    CUDA_SOURCES += cusparseExec.cu gpusolver.cpp
    CUDA_DIR = $$(CUDA_PATH) #"D:\App e giochi\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
    CUDA_ARCH = sm_61

    INCLUDEPATH  += $$CUDA_DIR/include
    QMAKE_LIBDIR += $$CUDA_DIR/lib/x64

    LIBS += -lcudart -lcuda -lcusparse

    cudaC_FLAGS = -std=c++20
    cudaL_FLAGS = -m64 -arch=sm_61 -Wno-deprecated-gpu-targets -std=c++20

    MSVCRT_LINK_FLAG_DEBUG = "/MDd"
    MSVCRT_LINK_FLAG_RELEASE = "/MD"

    # Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
    CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

    # Compile CUDA source files using NVCC
    cudaC.input = CUDA_SOURCES
    cudaC.output = ${QMAKE_FILE_BASE}_cuda.o
    cudaC.commands = $$CUDA_DIR\bin\nvcc -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE $$cudaL_FLAGS -dc $$cudaC_FLAGS $$CUDA_INC $$LIBS -o ${QMAKE_FILE_BASE}_cuda.o ${QMAKE_FILE_NAME}
    cudaC.dependency_type = TYPE_C
    cudaC.variable_out = CUDA_OBJ
    cudaC.variable_out += OBJECTS
    QMAKE_EXTRA_COMPILERS += cudaC

    # Linking CUDA source files using NVCC - needed for dynamic parallelism
    cudaL.input = CUDA_OBJ
    cudaL.output = cudaLinked.o
    cudaL.CONFIG += combine
    cudaL.commands = $$CUDA_DIR\bin\nvcc -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE $$cudaL_FLAGS -dlink -o cudaLinked.o ${QMAKE_FILE_NAME}
    cudaL.depend_command = $$CUDA_DIR/bin/nvcc -g -G -MD $CUDA_INC $NVCC_FLAGS ${QMAKE_FILE_NAME}         #seems not necessary
    QMAKE_EXTRA_COMPILERS += cudaL
}
