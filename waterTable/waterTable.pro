#-------------------------------------------------
#
#   WaterTable library import data, computation, draw
#
#-------------------------------------------------

QT      += core xml
QT      -= gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/waterTable
    } else {
        TARGET = release/waterTable
    }
}
win32:{
    TARGET = waterTable
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../meteo ../interpolation ../gis ../weatherGenerator

SOURCES += \
    importData.cpp \
    waterTable.cpp \
    well.cpp

HEADERS += \
    importData.h \
    waterTable.h \
    well.h

