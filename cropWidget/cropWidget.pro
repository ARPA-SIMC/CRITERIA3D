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

QT  += widgets sql charts printsupport

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

INCLUDEPATH += ../crit3dDate ../mathFunctions ../utilities ../gis ../meteo ../soil ../crop ../qcustomplot ../criteriaModel ../project


SOURCES += \
    ../../agrolib/project/formInfo.cpp \
    callout.cpp \
    cropWidget.cpp \
    dialogNewCrop.cpp \
    tabIrrigation.cpp \
    tabLAI.cpp \
    tabRootDensity.cpp \
    tabRootDepth.cpp \
    tabWaterContent.cpp

HEADERS += \
    ../../agrolib/project/formInfo.h \
    callout.h \
    cropWidget.h \
    dialogNewCrop.h \
    tabIrrigation.h \
    tabLAI.h \
    tabRootDensity.h \
    tabRootDepth.h \
    tabWaterContent.h
