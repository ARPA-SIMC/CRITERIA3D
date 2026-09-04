#-------------------------------------------------------
#
#   crit3dDate library
#   this project is part of ARPAE agrolib distribution
#
#-------------------------------------------------------

QT   -= core gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

DEFINES += _CRT_SECURE_NO_WARNINGS


unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/crit3dDate
    } else {
        TARGET = release/crit3dDate
    }
}
win32:{
    TARGET = crit3dDate
}


SOURCES += \
    crit3dDate.cpp \
    crit3dTime.cpp

HEADERS += \
    crit3dDate.h



