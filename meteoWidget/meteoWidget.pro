#----------------------------------------------------
#
#   Meteo Widget library
#   This project is part of CRITERIA-3D distribution
#
#   It requires Qwt library
#   https://qwt.sourceforge.io/index.html
#   Windows: set QWT_ROOT in environment variables
#
#----------------------------------------------------

QT  += widgets 

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/meteoWidget
    } else {
        TARGET = release/meteoWidget
    }
}
macx:{
    CONFIG(debug, debug|release) {
        TARGET = debug/meteoWidget
    } else {
        TARGET = release/meteoWidget
    }
}
win32:{
    TARGET = meteoWidget
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../meteo ../utilities


SOURCES += \
    meteoWidget.cpp


HEADERS += \
    meteoWidget.h


