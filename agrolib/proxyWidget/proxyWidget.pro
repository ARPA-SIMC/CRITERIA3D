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
CONFIG += c++11 c++14 c++17

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

INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo ../utilities ../commonDialogs ../commonChartElements ../interpolation


SOURCES += \
    chartView.cpp \
    localProxyWidget.cpp \
    proxyWidget.cpp


HEADERS += \
    chartView.h \
    localProxyWidget.h \
    proxyWidget.h 


