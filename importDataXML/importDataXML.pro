#-------------------------------------------------
#
#   ImportDataXML library
#
#-------------------------------------------------

QT      += core sql xml
QT      -= gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
QMAKE_CXXFLAGS += -std=c++11

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/importDataXML
    } else {
        TARGET = release/importDataXML
    }
}
win32:{
    TARGET = importDataXML
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../dbMeteoPoints ../dbMeteoGrid

SOURCES += importDataXML.cpp \
    fieldXML.cpp \
    variableXML.cpp


HEADERS += importDataXML.h \
    fieldXML.h \
    variableXML.h


