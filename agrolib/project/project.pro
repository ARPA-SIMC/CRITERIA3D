#-----------------------------------------------------
#
#   project library
#   This project is part of CRITERIA-3D distribution
#
#-----------------------------------------------------

QT  += core gui widgets charts sql xml

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
QMAKE_CXXFLAGS += -std=c++11

DEFINES += _CRT_SECURE_NO_WARNINGS

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/project
    } else {
        TARGET = release/project
    }
}
win32:{
    TARGET = project
}


INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo   \
            ../solarRadiation ../interpolation ../utilities     \
            ../netcdfHandler ../dbMeteoPoints ../dbMeteoGrid ../meteoWidget ../commonDialogs


SOURCES += \
    aggregation.cpp \
    dialogInterpolation.cpp \
    dialogProject.cpp \
    dialogRadiation.cpp \
    dialogSelection.cpp \
    dialogSettings.cpp \
    interpolationCmd.cpp \
    meteoMaps.cpp \
    project.cpp \
    shell.cpp


HEADERS += \
    aggregation.h \
    dialogInterpolation.h \
    dialogProject.h \
    dialogRadiation.h \
    dialogSelection.h \
    dialogSettings.h \
    interpolationCmd.h \
    meteoMaps.h \
    project.h \
    shell.h

FORMS +=

