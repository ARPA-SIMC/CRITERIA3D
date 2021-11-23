#----------------------------------------------------
#
#   Proxy Widget library
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
        TARGET = debug/proxyWidget
    } else {
        TARGET = release/proxyWidget
    }
}
macx:{
    CONFIG(debug, debug|release) {
        TARGET = debug/proxyWidget
    } else {
        TARGET = release/proxyWidget
    }
}
win32:{
    TARGET = proxyWidget
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo ../utilities ../commonDialogs ../interpolation


SOURCES += \
    chartView.cpp \
    proxyCallout.cpp \
    proxyWidget.cpp


HEADERS += \
    chartView.h \
    proxyCallout.h \
    proxyWidget.h 


