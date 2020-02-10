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

QT  += widgets sql

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

INCLUDEPATH += ../crit3dDate ../mathFunctions ../utilities ../soil ../crop


SOURCES += \
    cropWidget.cpp

HEADERS += \
    cropWidget.h
