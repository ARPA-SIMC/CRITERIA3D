#-----------------------------------------------------
#
#   EISPACK Library
#   downloaded from https://people.sc.fsu.edu/~jburkardt/c_src/eispack/eispack.html
#    GNU LGPL license
#
#   This project is part of CRITERIA-3D distribution
#
#-----------------------------------------------------

QT   -= core gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/eispack
    } else {
        TARGET = release/eispack
    }
}
win32:{
    TARGET = eispack
}


SOURCES += \
    eispack.cpp

HEADERS += \
    eispack.h
