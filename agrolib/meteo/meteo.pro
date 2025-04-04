#-------------------------------------------------------
#
#   meteo library
#   This project is part of ARPAE agrolib distribution
#
#-------------------------------------------------------

QT       -= core gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17


unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/meteo
    } else {
        TARGET = release/meteo
    }
}
win32:{
    TARGET = meteo
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis

SOURCES += meteo.cpp \
    meteoPoint.cpp \
    meteoGrid.cpp \
    quality.cpp

HEADERS += meteo.h \
    meteoPoint.h \
    meteoGrid.h \
    quality.h

