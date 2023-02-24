#----------------------------------------------------
#
#   gis library
#   This project is part of CRITERIA-3D distribution
#
#----------------------------------------------------

QT  -= core gui

TEMPLATE = lib
CONFIG += staticlib

DEFINES += _CRT_SECURE_NO_WARNINGS

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/gis
    } else {
        TARGET = release/gis
    }
}
win32:{
    TARGET = gis
}

INCLUDEPATH += ../mathFunctions ../crit3dDate

SOURCES += gis.cpp \
    gisIO.cpp \
    color.cpp \
    geoMap.cpp

HEADERS += gis.h \
    color.h \
    gisIO.h \
    geoMap.h

