#----------------------------------------------------
#
#   soil library
#   This project is part of CRITERIA-3D distribution
#
#----------------------------------------------------

QT -= gui
QT += core sql

TARGET = soil
TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/soil
    } else {
        TARGET = release/soil
    }
}
win32:{
    TARGET = soil
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../utilities

SOURCES += soil.cpp \
    soilDbTools.cpp

HEADERS += soil.h \
    soilDbTools.h

