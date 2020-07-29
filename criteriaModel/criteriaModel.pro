#----------------------------------------------------------------
#
#   criteriaModel library
#
#   Water balance 1D
#   algorithms for soil water infiltration, redistribution,
#   capillary rise, crop water demand and irrigation.
#
#   This library is part of CRITERIA3D distribution
#
#----------------------------------------------------------------

QT      += core sql xml
QT      -= gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/criteriaModel
    } else {
        TARGET = release/criteriaModel
    }
}
win32:{
    TARGET = criteriaModel
}

INCLUDEPATH +=  ../crit3dDate ../mathFunctions ../gis ../meteo \
                ../dbMeteoGrid ../soil ../crop ../utilities

SOURCES += \
    criteria1DCase.cpp \
    criteria1DUnit.cpp \
    criteria1DdbMeteo.cpp \
    irrigationForecast.cpp \
    water1D.cpp

HEADERS += \
    criteria1DCase.h \
    criteria1DUnit.h \
    criteria1DdbMeteo.h \
    irrigationForecast.h \
    water1D.h

unix {
    target.path = /usr/lib
    INSTALLS += target
}

