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
CONFIG += c++11 c++14 c++17

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

INCLUDEPATH += ../crit3dDate ../mathFunctions ../meteo ../gis ../interpolation ../dbMeteoPoints ../dbMeteoGrid

SOURCES += importDataXML.cpp \
    fieldXML.cpp \
    importPropertiesCSV.cpp \
    variableXML.cpp


HEADERS += importDataXML.h \
    fieldXML.h \
    importPropertiesCSV.h \
    variableXML.h


