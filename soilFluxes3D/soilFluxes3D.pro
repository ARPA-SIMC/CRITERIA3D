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

TEMPLATE = lib
CONFIG += staticlib
CONFIG += c++17
CONFIG += debug_and_release

INCLUDEPATH += $$absolute_path(../mathFunctions)

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

SOURCES += \
    cpusolver.cpp \
    heat.cpp \
    otherFunctions.cpp \
    soilFluxes3D.cpp \
    soilPhysics.cpp \
    water.cpp


HEADERS += \
    cpusolver.h \
    heat.h \
    macro.h \
    otherFunctions.h \
    soilFluxes3D.h \
    soilPhysics.h \
    solver.h \
    types.h \
    types_cpu.h \
    types_opt.h \
    water.h

DISTFILES += \
    ToDoList.txt


# parallel computing settings
include($$absolute_path(../parallel.pri))


contains(DEFINES, MCR_ENABLED) {
    HEADERS += \
        logFunctions.h

    SOURCES += \
        logFunctions.cpp
}

contains(DEFINES, CUDA_ENABLED) {
    HEADERS += \
        gpusolver.h \
        types_gpu.h \

    SOURCES += \
        gpusolver.cpp \

    # CUDA settings
    CUDA_SOURCES += $$SOURCES
    SOURCES -= $$CUDA_SOURCES

    # Compile CUDA source files using NVCC
    cudaC.input = CUDA_SOURCES
    cudaC.output = $$SUBPATH/${QMAKE_FILE_BASE}_cuda.o
    cudaC.commands = $$CUDA_DIR\bin\nvcc -Xcompiler $$MSVCRT_LINK_FLAG -Xcompiler $$HOST_C_FLAGS -Xcompiler $$HOST_DEFINES $$cudaL_FLAGS -dc $$cudaC_FLAGS $$CUDA_INC $$LIBS -o $$SUBPATH/${QMAKE_FILE_BASE}_cuda.o -x cu ${QMAKE_FILE_NAME}
    cudaC.dependency_type = TYPE_C
    cudaC.variable_out = CUDA_OBJ
    cudaC.variable_out += OBJECTS
    QMAKE_EXTRA_COMPILERS += cudaC

    # Linking CUDA source files using NVCC - needed for dynamic parallelism
    cudaL.input = CUDA_OBJ
    cudaL.output = $$SUBPATH/cudaLinked.o
    cudaL.CONFIG += combine
    cudaL.commands = $$CUDA_DIR\bin\nvcc -Xcompiler $$MSVCRT_LINK_FLAG -Xlinker $$HOST_L_FLAGS -Xcompiler $$HOST_DEFINES $$cudaL_FLAGS -dlink -o $$SUBPATH/cudaLinked.o ${QMAKE_FILE_NAME}
    cudaL.depend_command = $$CUDA_DIR/bin/nvcc -g -G -MD $CUDA_INC $NVCC_FLAGS ${QMAKE_FILE_NAME}         #seems not necessary
    QMAKE_EXTRA_COMPILERS += cudaL
}

# Old version (v1)
SOURCES += \
    old/old_boundary.cpp \
    old/old_balance.cpp \
    old/old_dataLogging.cpp \
    old/old_water.cpp \
    old/old_solver.cpp \
    old/old_memory.cpp \
    old/old_soilPhysics.cpp \
    old/old_soilFluxes3D.cpp \
    old/old_heat.cpp \
    old/old_extra.cpp

HEADERS += \
    old/old_macro.h \
    old/old_types.h \
    old/old_parameters.h \
    old/old_boundary.h \
    old/old_balance.h \
    old/old_water.h \
    old/old_solver.h \
    old/old_memory.h \
    old/old_soilPhysics.h \
    old/old_soilFluxes3D.h \
    old/old_extra.h \
    old/old_heat.h
