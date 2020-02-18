#----------------------------------------------------
#
#   Crop Widget library
#   This project is part of CRITERIA-3D distribution
#
#   It requires Qwt library
#   https://qwt.sourceforge.io/index.html
#   Windows: set QWT_ROOT in environment variables
#
#----------------------------------------------------

QT  += widgets sql charts

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/cropWidget
    } else {
        TARGET = release/cropWidget
    }
}
macx:{
    CONFIG(debug, debug|release) {
        TARGET = debug/cropWidget
    } else {
        TARGET = release/cropWidget
    }
}
win32:{
    TARGET = cropWidget
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../utilities ../gis ../meteo ../soil ../crop ../criteriaModel


SOURCES += \
    crit3DChartView.cpp \
    cropWidget.cpp \
    dialogNewCrop.cpp \
    tabLAI.cpp \
    tabRootDepth.cpp

HEADERS += \
    crit3DChartView.h \
    cropWidget.h \
    dialogNewCrop.h \
    tabLAI.h \
    tabRootDepth.h
