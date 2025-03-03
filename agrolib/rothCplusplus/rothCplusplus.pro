#---------------------------------------------------
#
#   rothCplusplus library
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
        TARGET = debug/rothCplusplus
    } else {
        TARGET = release/rothCplusplus
    }
}
win32:{
    TARGET = rothCplusplus
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../soil ../crop ../gis

SOURCES += rothCplusplus.cpp 


HEADERS += rothCplusplus.h 

