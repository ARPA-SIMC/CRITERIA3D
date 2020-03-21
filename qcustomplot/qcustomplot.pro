#----------------------------------------------------
#
#   qcustomplot library
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
        TARGET = debug/qcustomplot
    } else {
        TARGET = release/qcustomplot
    }
}
win32:{
    TARGET = qcustomplot
}


SOURCES += \
    qcustomplot.cpp

HEADERS += \
    qcustomplot.h



