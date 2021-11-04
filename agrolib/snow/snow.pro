#-----------------------------------------------------
#
#   snow library
#   compute snow accumulation and melt
#   mono-dimensional energy balance
#
#   This library is part of CRITERIA-3D distribution
#
#-----------------------------------------------------

QT += core
QT -= gui

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

INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo ../solarRadiation

SOURCES += \
    snow.cpp \
    snowMaps.cpp

HEADERS += \
    snow.h \
    snowMaps.h

