#-----------------------------------------------------
#
#   meteo library
#   This project is part of CRITERIA-3D distribution
#
#-----------------------------------------------------

QT       -= core gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
QMAKE_CXXFLAGS += -std=c++11


unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/meteo
    } else {
        TARGET = release/meteo
    }
}
win32:{
    TARGET = meteo
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis

SOURCES += meteo.cpp \
    meteoPoint.cpp \
    meteoGrid.cpp \
    quality.cpp

HEADERS += meteo.h \
    meteoPoint.h \
    meteoGrid.h \
    quality.h

