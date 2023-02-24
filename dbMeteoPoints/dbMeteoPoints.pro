#------------------------------------------------------
#
#   dbMeteoPoints library
#   This project is part of CRITERIA-3D distribution
#
#------------------------------------------------------

QT       += network sql

QT       -= gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/dbMeteoPoints
    } else {
        TARGET = release/dbMeteoPoints
    }
}
win32:{
    TARGET = dbMeteoPoints
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo ../interpolation ../utilities

HEADERS += \
    dbAggregationsHandler.h \
    dbArkimet.h \
    dbMeteoPointsHandler.h \
    download.h \
    variablesList.h

SOURCES += \
    dbAggregationsHandler.cpp \
    dbArkimet.cpp \
    dbMeteoPointsHandler.cpp \
    download.cpp \
    variablesList.cpp


