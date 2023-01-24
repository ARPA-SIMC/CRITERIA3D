#----------------------------------------------------
#
#   Point Statistics Widget library
#   This project is part of CRITERIA-3D distribution
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
        TARGET = debug/pointStatisticsWidget
    } else {
        TARGET = release/pointStatisticsWidget
    }
}
macx:{
    CONFIG(debug, debug|release) {
        TARGET = debug/pointStatisticsWidget
    } else {
        TARGET = release/pointStatisticsWidget
    }
}
win32:{
    TARGET = pointStatisticsWidget
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo ../utilities ../dbMeteoPoints ../dbMeteoGrid ../phenology ../climate ../commonDialogs ../commonChartElements ../interpolation


SOURCES += \
    dialogElaboration.cpp \
    pointStatisticsChartView.cpp \
    pointStatisticsWidget.cpp


HEADERS += \
    dialogElaboration.h \
    pointStatisticsChartView.h \
    pointStatisticsWidget.h 


