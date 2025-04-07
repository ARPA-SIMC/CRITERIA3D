#---------------------------------------------------
#
#   grapevine library
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
        TARGET = debug/grapevine
    } else {
        TARGET = release/grapevine
    }
}
win32:{
    TARGET = grapevine
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../soil ../crop

SOURCES += grapevine.cpp \
    downyMildew.cpp \
    powderyMildew.cpp

HEADERS += grapevine.h \
    downyMildew.h \
    powderyMildew.h

