#------------------------------------------------------
#
#   Utilities library
#   This project is part of ARPAE agrolib distribution
#
#------------------------------------------------------

QT     += sql
QT     -= gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

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

