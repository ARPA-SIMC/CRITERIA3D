#---------------------------------------------------
#
#   hydrall library
#   This project is part of CRITERIA3D distribution
#
#---------------------------------------------------

QT       -= core gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

#DEFINES += _CRT_SECURE_NO_WARNINGS


unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/hydrall
    } else {
        TARGET = release/hydrall
    }
}
win32:{
    TARGET = hydrall
}

INCLUDEPATH += ../../agrolib/crit3dDate ../../agrolib/mathFunctions ../../agrolib/soil ../../agrolib/crop ../../agrolib/gis

SOURCES += hydrall.cpp 


HEADERS += hydrall.h 

