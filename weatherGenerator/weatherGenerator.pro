#---------------------------------------------------------------------
#
#   weatherGenerator (1D) library
#   This project is part of CRITERIA3D distribution
#
#   Based on Richardson, C. W. and D. A. Wright,
#   WGEN: A model for generating daily weather variables, USDA, 1984
#
#---------------------------------------------------------------------


QT      += xml
QT      -= gui


TEMPLATE = lib
CONFIG += staticlib

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/weatherGenerator
    } else {
        TARGET = release/weatherGenerator
    }
}
win32:{
    TARGET = weatherGenerator
}


INCLUDEPATH += ../crit3dDate ../mathFunctions ../meteo ../gis ../waterTable

SOURCES += \
    timeUtility.cpp \
    parserXML.cpp \
    wgClimate.cpp \
    fileUtility.cpp \
    weatherGenerator.cpp

HEADERS += \
    timeUtility.h \
    parserXML.h \
    wgClimate.h \
    fileUtility.h \
    weatherGenerator.h
