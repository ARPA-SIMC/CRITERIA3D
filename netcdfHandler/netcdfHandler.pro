#-----------------------------------------------------
#
#   netCDF handler library
#   This project is part of CRITERIA-3D distribution
#
#-----------------------------------------------------

QT       -= core gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
QMAKE_CXXFLAGS += -std=c++11

DEFINES += _CRT_SECURE_NO_WARNINGS


unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/netcdfHandler
    } else {
        TARGET = release/netcdfHandler
    }
}
else:{
    TARGET = netcdfHandler
}


INCLUDEPATH += ../mathFunctions ../crit3dDate ../gis

win32:{
    INCLUDEPATH += $$(NC4_INSTALL_DIR)/include
}
macx:{
    INCLUDEPATH += /usr/local/include/
}

SOURCES += \
    netcdfHandler.cpp

HEADERS += \
    netcdfHandler.h

