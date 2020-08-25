#-------------------------------------------------
#
#   Utilities library
#   This project is part of CRITERIA-3D distribution
#
#-------------------------------------------------

QT      += core sql
QT      -= gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
QMAKE_CXXFLAGS += -std=c++11

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/utilities
    } else {
        TARGET = release/utilities
    }
}
win32:{
    TARGET = utilities
}

INCLUDEPATH += ../crit3dDate ../mathFunctions

SOURCES += \
    computationUnitsDb.cpp \
    logger.cpp \
    utilities.cpp

HEADERS += \
    computationUnitsDb.h \
    logger.h \
    utilities.h

