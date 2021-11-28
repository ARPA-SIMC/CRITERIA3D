#------------------------------------------------------
#
#   outputPoints library
#   This project is part of CRITERIA-3D distribution
#
#------------------------------------------------------

QT       += network sql

QT       -= gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
QMAKE_CXXFLAGS += -std=c++11

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

INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo

HEADERS += \
    dbOutputPointsHandler.h \
    outputPoints.h


SOURCES += \
    dbOutputPointsHandler.cpp \
    outputPoints.cpp



