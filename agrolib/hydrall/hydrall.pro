#---------------------------------------------------
#
#   hydrall library
#   This project is part of CRITERIA3D distribution
#
#---------------------------------------------------

QT      -= core gui

TEMPLATE = lib
CONFIG += staticlib

DEFINES += _CRT_SECURE_NO_WARNINGS

CONFIG += debug_and_release

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

INCLUDEPATH += ../crit3dDate ../mathFunctions ../soil ../crop

SOURCES += hydrall.cpp 


HEADERS += hydrall.h 


