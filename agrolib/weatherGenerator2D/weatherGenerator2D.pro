#---------------------------------------------------------------------
#
#   weatherGenerator2D library
#   Spatial weather generator model
#
#   This project is part of CRITERIA-3D distribution
#
#   Code translated from the MulGets model available online on:
#   https://it.mathworks.com/matlabcentral/fileexchange/47537-multi-site-stochstic-weather-generator--mulgets-
#
#---------------------------------------------------------------------

QT  -= gui
QT  += xml

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/weatherGenerator2D
    } else {
        TARGET = release/weatherGenerator2D
    }
}
win32:{
    TARGET = weatherGenerator2D
}

TEMPLATE = lib
CONFIG += staticlib

INCLUDEPATH += ../mathFunctions
INCLUDEPATH += ../eispack
INCLUDEPATH += ../crit3dDate
INCLUDEPATH += ../meteo
INCLUDEPATH += ../gis
INCLUDEPATH += ../crit3dDate
INCLUDEPATH += ../weatherGenerator

SOURCES += wg2D.cpp \
    randomset.cpp \
    wg2D_precipitation.cpp \
    wg2D_temperature.cpp \
    wg2doutputmanagement.cpp

HEADERS += wg2D.h
