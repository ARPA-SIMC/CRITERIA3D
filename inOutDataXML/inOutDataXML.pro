#-------------------------------------------------
#
#   Import Export DataXML library
#
#-------------------------------------------------

QT      += sql xml
QT      -= gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/inOutDataXML
    } else {
        TARGET = release/inOutDataXML
    }
}
win32:{
    TARGET = inOutDataXML
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../meteo ../gis ../interpolation ../dbMeteoPoints ../dbMeteoGrid

SOURCES += inOutDataXML.cpp \
    fieldXML.cpp \
    variableXML.cpp


HEADERS += inOutDataXML.h \
    fieldXML.h \
    variableXML.h


