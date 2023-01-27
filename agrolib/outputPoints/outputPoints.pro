#------------------------------------------------------
#
#   outputPoints library
#   This project is part of CRITERIA-3D distribution
#
#------------------------------------------------------

QT       += network sql widgets

QT       -= gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/outputPoints
    } else {
        TARGET = release/outputPoints
    }
}
win32:{
    TARGET = outputPoints
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo ../utilities

HEADERS += \
    dbOutputPointsHandler.h \
    dialogNewPoint.h \
    outputPoints.h


SOURCES += \
    dbOutputPointsHandler.cpp \
    dialogNewPoint.cpp \
    outputPoints.cpp



