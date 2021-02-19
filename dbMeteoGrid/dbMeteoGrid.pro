#-------------------------------------------------
#
#   dbMeteoGrid library
#   This project is part of CRITERIA-3D distribution
#
#-------------------------------------------------

QT       += sql xml core

QT       -= gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
QMAKE_CXXFLAGS += -std=c++11

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/dbMeteoGrid
    } else {
        TARGET = release/dbMeteoGrid
    }
}
win32:{
    TARGET = dbMeteoGrid
}

DEFINES += DBMETEOGRID_LIBRARY

INCLUDEPATH += ../mathFunctions ../crit3dDate ../gis ../meteo ../utilities


SOURCES += \
        dbMeteoGrid.cpp \

HEADERS += \
        dbMeteoGrid.h

