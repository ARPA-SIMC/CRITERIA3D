#-----------------------------------------------------
#
#   drought library
#   This project is part of CRITERIA-3D distribution
#
#-----------------------------------------------------

QT       -= core gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
QMAKE_CXXFLAGS += -std=c++11


unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/drought
    } else {
        TARGET = release/drought
    }
}
win32:{
    TARGET = drought
}

INCLUDEPATH += ../crit3dDate ../mathFunctions 

SOURCES +=  

HEADERS += 

