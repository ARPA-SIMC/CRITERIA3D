#----------------------------------------------------
#
#   carbonNitrogen library
#   This project is part of CRITERIA-3D distribution
#
#----------------------------------------------------

QT -= gui

TARGET = carbonNitrogen
TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/carbonNitrogen
    } else {
        TARGET = release/carbonNitrogen
    }
}
win32:{
    TARGET = carbonNitrogen
}

INCLUDEPATH += ../mathFunctions ../soil

HEADERS += carbonNitrogen.h
	
SOURCES += carbonNitrogen.cpp

