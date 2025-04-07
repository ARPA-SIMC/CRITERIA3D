#------------------------------------------------------
#
#   dbMeteoGrid library
#   This project is part of ARPAE agrolib distribution
#
#------------------------------------------------------

QT       += sql xml

QT       -= gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

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

