#-----------------------------------------------------
#
#   drought library
#   This project is part of CRITERIA-3D distribution
#
#-----------------------------------------------------

QT       -= gui
QT       += sql xml

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17


unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/drought
    } else {
        TARGET = release/drought
    }
}
win32:{
    TARGET = drought
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo ../interpolation ../dbMeteoPoints

SOURCES +=   \
    drought.cpp

HEADERS +=  \
    drought.h

