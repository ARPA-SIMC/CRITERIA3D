#----------------------------------------------------
#
#   Homogeneity Widget library
#   This project is part of PRAGA distribution
#
#
#----------------------------------------------------

QT  += widgets charts sql xml

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++14 c++17

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/homogeneityWidget
    } else {
        TARGET = release/homogeneityWidget
    }
}
macx:{
    CONFIG(debug, debug|release) {
        TARGET = debug/homogeneityWidget
    } else {
        TARGET = release/homogeneityWidget
    }
}
win32:{
    TARGET = homogeneityWidget
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo ../utilities ../dbMeteoPoints ../dbMeteoGrid ../phenology ../climate ../commonDialogs ../commonChartElements ../interpolation


SOURCES += \
    annualSeriesChartView.cpp \
    homogeneityChartView.cpp \
    homogeneityWidget.cpp


HEADERS += \
    annualSeriesChartView.h \
    homogeneityChartView.h \
    homogeneityWidget.h 


