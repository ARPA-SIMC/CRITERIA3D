#-----------------------------------------------------
#
#   snow library
#   compute snow accumulation and melt
#   with a mono-dimensional energy balance
#
#   This project is part of CRITERIA-3D distribution
#
#-----------------------------------------------------

QT -= core gui

CONFIG += staticlib
CONFIG += debug_and_release

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/snow
    } else {
        TARGET = release/snow
    }
}
win32:{
    TARGET = snow
}

TEMPLATE = lib

INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo

SOURCES += \
    snow.cpp \
    snowMaps.cpp

HEADERS += \
    snow.h \
    snowMaps.h

