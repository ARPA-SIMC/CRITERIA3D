#-------------------------------------------------
#
#   ImportDataXML library
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
        TARGET = debug/ImportDataXML
    } else {
        TARGET = release/ImportDataXML
    }
}
win32:{
    TARGET = ImportDataXML
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../dbMeteoPoints ../dbMeteoGrid

SOURCES += \


HEADERS += \


