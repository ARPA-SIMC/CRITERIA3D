#----------------------------------------------------
#
#  QCustomPlot library
#  Copyright (C) 2011-2018 Emanuel Eichhammer
#  GPL V.3 license
#
#----------------------------------------------------
#  Author: Emanuel Eichhammer
#  Website/Contact: http://www.qcustomplot.com/
#  Date: 25.06.18
#  Version: 2.0.1
#----------------------------------------------------

QT   += gui

TEMPLATE = lib
CONFIG += staticlib

DEFINES += _CRT_SECURE_NO_WARNINGS

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

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



