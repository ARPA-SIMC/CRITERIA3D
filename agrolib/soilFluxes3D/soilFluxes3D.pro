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

QMAKE_CXXFLAGS += -openmp:llvm -openmp:experimental -GL
QMAKE_LFLAGS += -openmp:llvm -IGNORE:4217 -LTCG

TEMPLATE = lib
CONFIG += staticlib
CONFIG += c++17
CONFIG += debug_and_release

INCLUDEPATH += ../mathFunctions

SOURCES += \
    boundary.cpp \
    balance.cpp \
    dataLogging.cpp \
    water.cpp \
    solver.cpp \
    memory.cpp \
    soilPhysics.cpp \
    soilFluxes3D.cpp \
    heat.cpp \
    extra.cpp \

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
