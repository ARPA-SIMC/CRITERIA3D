#----------------------------------------------------
#
#   Meteo Widget library
#   This project is part of CRITERIA-3D distribution
#
#
#----------------------------------------------------

QT  += widgets charts

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

INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo ../utilities ../project


SOURCES += \
    ../../agrolib/project/formInfo.cpp \
    dialogMeteoTable.cpp \
    dialogSelectVar.cpp \
    meteoWidget.cpp \
    callout.cpp


HEADERS += \
    ../../agrolib/project/formInfo.h \
    dialogMeteoTable.h \
    dialogSelectVar.h \
    meteoWidget.h   \
    callout.h


