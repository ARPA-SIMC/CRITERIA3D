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

INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo ../utilities ../commonDialogs


SOURCES += \
    dialogChangeAxis.cpp \
    dialogMeteoTable.cpp \
    dialogSelectVar.cpp \
    meteoTable.cpp \
    meteoWidget.cpp \
    callout.cpp


HEADERS += \
    dialogChangeAxis.h \
    dialogMeteoTable.h \
    dialogSelectVar.h \
    meteoTable.h \
    meteoWidget.h   \
    callout.h


