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
    homogeneityChartView.cpp \
    homogeneityWidget.cpp


HEADERS += \
    homogeneityChartView.h \
    homogeneityWidget.h 


