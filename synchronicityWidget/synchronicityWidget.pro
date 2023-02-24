#----------------------------------------------------
#
#   Synchronicity Widget library
#   This project is part of PRAGA distribution
#
#----------------------------------------------------

QT  += widgets charts sql xml

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/synchronicityWidget
    } else {
        TARGET = release/synchronicityWidget
    }
}
macx:{
    CONFIG(debug, debug|release) {
        TARGET = debug/synchronicityWidget
    } else {
        TARGET = release/synchronicityWidget
    }
}
win32:{
    TARGET = synchronicityWidget
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo ../utilities ../dbMeteoPoints \
                ../dbMeteoGrid ../climate ../phenology ../commonDialogs ../commonChartElements ../interpolation


SOURCES += \
    interpolationChartView.cpp \
    synchronicityChartView.cpp \
    synchronicityWidget.cpp


HEADERS += \
    interpolationChartView.h \
    synchronicityChartView.h \
    synchronicityWidget.h 

