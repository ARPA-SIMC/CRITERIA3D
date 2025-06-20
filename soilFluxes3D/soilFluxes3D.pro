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

QT   -= gui

QMAKE_CXXFLAGS += -openmp:llvm
QMAKE_LFLAGS += -openmp:llvm

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release

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

# CUDA settings
CUDA_SOURCES = kernelTest.cu
CUDA_DIR = $$(CUDA_PATH) #"D:\App e giochi\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
CUDA_ARCH = sm_61

INCLUDEPATH  += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib/x64

LIBS += -lcudart -lcuda

NVCCFLAGS = --use_fast_math

# Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

MSVCRT_LINK_FLAG_DEBUG = "/MDd"
MSVCRT_LINK_FLAG_RELEASE = "/MD"

# Compile CUDA source files using NVCC
cuda.input = CUDA_SOURCES
cuda.output = ${QMAKE_FILE_BASE}.o
cuda.commands = $$CUDA_DIR/bin/nvcc.exe -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE -arch=$$CUDA_ARCH $$CUDA_INC $$LIBS -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
cuda.dependency_type = TYPE_C
QMAKE_EXTRA_COMPILERS += cuda


INCLUDEPATH += ../mathFunctions


SOURCES +=  \
    boundary.cpp \
    balance.cpp \
    dataLogging.cpp \
    testCUDAinProject.cpp \
    water.cpp \
    solver.cpp \
    memory.cpp \
    soilPhysics.cpp \
    soilFluxes3D.cpp \
    heat.cpp \
    extra.cpp


HEADERS += \
    macro.h \
    types.h \
    parameters.h \
    boundary.h \
    balance.h \
    water.h \
    solver.h \
    memory.h \
    soilPhysics.h \
    soilFluxes3D.h \
    extra.h \
    heat.h
