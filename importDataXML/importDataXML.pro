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
        TARGET = debug/ImportDataXML
    } else {
        TARGET = release/ImportDataXML
    }
}
win32:{
    TARGET = ImportDataXML
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../dbMeteoPoints ../dbMeteoGrid

SOURCES += importDataXML.cpp \
    fieldXML.cpp


HEADERS += importDataXML.h \
    fieldXML.h


