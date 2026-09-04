#--------------------------------------------------------
#
#   gis library
#   This project is part of ARPAE agrolib distribution
#
#-------------------------------------------------------

QT  -= core gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

DEFINES += _CRT_SECURE_NO_WARNINGS

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
    color.cpp \
    geoMap.cpp \
    gisIO.cpp \
    watershed.cpp

HEADERS += gis.h \
    color.h \
    gisIO.h \
    geoMap.h \
    watershed.h
