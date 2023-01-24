#----------------------------------------------------
#
#   Meteo Widget library
#   This project is part of CRITERIA-3D distribution
#
#
#----------------------------------------------------

QT  += widgets charts sql

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++14 c++17

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/meteoWidget
    } else {
        TARGET = release/meteoWidget
    }
}
macx:{
    CONFIG(debug, debug|release) {
        TARGET = debug/meteoWidget
    } else {
        TARGET = release/meteoWidget
    }
}
win32:{
    TARGET = meteoWidget
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo ../utilities ../commonDialogs ../commonChartElements


SOURCES += \
    dialogMeteoTable.cpp \
    dialogSelectVar.cpp \
    meteoTable.cpp \
    meteoWidget.cpp


HEADERS += \
    dialogMeteoTable.h \
    dialogSelectVar.h \
    meteoTable.h \
    meteoWidget.h


