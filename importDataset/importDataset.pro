#-------------------------------------------------
#
#   ImportDataset library
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
        TARGET = debug/importDataset
    } else {
        TARGET = release/importDataset
    }
}
win32:{
    TARGET = importDataset
}

INCLUDEPATH += ../crit3dDate ../mathFunctions

SOURCES += \
    dailyDataset.cpp \
    forecastDataset.cpp \
    pointDataset.cpp \
    varDataset.cpp

HEADERS += \
    dailyDataset.h \
    forecastDataset.h \
    pointDataset.h \
    varDataset.h

