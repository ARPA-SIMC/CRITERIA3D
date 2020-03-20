#----------------------------------------------------
#
#   customplot library
#
#----------------------------------------------------

QT   -= gui

TEMPLATE = lib
CONFIG += staticlib

DEFINES += _CRT_SECURE_NO_WARNINGS

CONFIG += debug_and_release
QMAKE_CXXFLAGS += -std=c++11

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/customplot
    } else {
        TARGET = release/customplot
    }
}
win32:{
    TARGET = customplot
}


SOURCES += \
    qcustomplot.cpp

HEADERS += \
    qcustomplot.h



