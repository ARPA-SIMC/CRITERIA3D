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

win32:{
    QMAKE_CXXFLAGS += -openmp:llvm -GL
    QMAKE_LFLAGS += -LTCG
}
unix:{
    QMAKE_CXXFLAGS += -fopenmp -flto
    QMAKE_LFLAGS += -fopenmp -flto
}

TEMPLATE = lib
CONFIG += staticlib
CONFIG += c++17
CONFIG += debug_and_release

INCLUDEPATH += $$absolute_path(..\mathFunctions)

SOURCES += \
    cpusolver.cpp \
    heat.cpp \
    otherFunctions.cpp \
    soilFluxes3DNew.cpp \
    soilPhysics.cpp \
    water.cpp

HEADERS += \
    cpusolver.h \
    heat.h \
    logFunctions.h \
    macro.h \
    otherFunctions.h \
    soilFluxes3DNew.h \
    soilPhysics.h \
    solver.h \
    types_cpu.h \
    types_opt.h \
    water.h

DISTFILES += \
    ToDoList.txt \
    temp_gpuSolver_cpp.txt

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/soilFluxes3DNew
    } else {
        TARGET = release/soilFluxes3DNew
    }
}
win32:{
    TARGET = soilFluxes3DNew
}

#CONFIG += CUDA_CONFIG

CONFIG(CUDA_CONFIG) {
    DEFINES += CUDA_ENABLED

    HEADERS += \
        gpusolver.h \
        types_gpu.h \

    SOURCES += \
        gpusolver.cpp \

    # CUDA settings
    CUDA_DIR = $$(CUDA_PATH) #"D:\App e giochi\NVIDIA GPU Computing Toolkit\CUDA\v12.9"
    CUDA_ARCH = sm_61
    CUDA_SOURCES += $$SOURCES
    SOURCES -= $$CUDA_SOURCES

    INCLUDEPATH  += $$CUDA_DIR/include
    QMAKE_LIBDIR += $$CUDA_DIR/lib/x64
    LIBS += -lcudart -lcuda -lcusparse

    QMAKE_CXXFLAGS -= -GL -flto
    QMAKE_LFLAGS -= -LTCG -flto
    HOST_C_FLAGS = $$join(QMAKE_CXXFLAGS,',','"','"')
    HOST_L_FLAGS = $$join(QMAKE_LFLAGS,',','"','"')

    cudaC_FLAGS = -DCUDA_ENABLED -m64  -std=c++17
    cudaL_FLAGS = -Wno-deprecated-gpu-targets -arch=sm_61

    CONFIG(debug, debug|release) {
        MSVCRT_LINK_FLAG = "/MDd"
        SUBPATH = debug
        cudaC_FLAGS += -g -G
    } else {
        MSVCRT_LINK_FLAG = "/MD"
        SUBPATH = release
    }

    # Prepare the extra compiler configuration (taken from the nvidia forum - i'm not an expert in this part)
    CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

    # Compile CUDA source files using NVCC
    cudaC.input = CUDA_SOURCES
    cudaC.output = $$SUBPATH/${QMAKE_FILE_BASE}_cuda.o
    cudaC.commands = $$CUDA_DIR\bin\nvcc -Xcompiler $$MSVCRT_LINK_FLAG -Xcompiler $$HOST_C_FLAGS $$cudaL_FLAGS -dc $$cudaC_FLAGS $$CUDA_INC $$LIBS -o $$SUBPATH/${QMAKE_FILE_BASE}_cuda.o -x cu ${QMAKE_FILE_NAME}
    cudaC.dependency_type = TYPE_C
    cudaC.variable_out = CUDA_OBJ
    cudaC.variable_out += OBJECTS
    QMAKE_EXTRA_COMPILERS += cudaC

    # Linking CUDA source files using NVCC - needed for dynamic parallelism
    cudaL.input = CUDA_OBJ
    cudaL.output = $$SUBPATH/cudaLinked.o
    cudaL.CONFIG += combine
    cudaL.commands = $$CUDA_DIR\bin\nvcc -Xcompiler $$MSVCRT_LINK_FLAG -Xlinker $$HOST_L_FLAGS $$cudaL_FLAGS -dlink -o $$SUBPATH/cudaLinked.o ${QMAKE_FILE_NAME}
    cudaL.depend_command = $$CUDA_DIR/bin/nvcc -g -G -MD $CUDA_INC $NVCC_FLAGS ${QMAKE_FILE_NAME}         #seems not necessary
    QMAKE_EXTRA_COMPILERS += cudaL
}
