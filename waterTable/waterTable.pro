#-------------------------------------------------
#
#   WaterTable library import data, computation, draw
#
#-------------------------------------------------

QT      += widgets charts core xml
QT      -= gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/waterTable
    } else {
        TARGET = release/waterTable
    }
}
win32:{
    TARGET = waterTable
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../meteo ../interpolation ../gis ../commonChartElements

SOURCES += \
    dialogSelectWell.cpp \
    dialogSummary.cpp \
    importData.cpp \
    waterTable.cpp \
    waterTableChartView.cpp \
    waterTableWidget.cpp \
    well.cpp

HEADERS += \
    dialogSelectWell.h \
    dialogSummary.h \
    importData.h \
    waterTable.h \
    waterTableChartView.h \
    waterTableWidget.h \
    well.h

