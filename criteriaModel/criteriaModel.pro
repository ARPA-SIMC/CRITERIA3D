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
DEFINES += _CRT_SECURE_NO_WARNINGS

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
                ../dbMeteoGrid ../soil ../crop ../utilities ../soilFluxes3D/header

SOURCES += \
    carbonNitrogenModel.cpp \
    criteria1DCase.cpp \
    criteria1DMeteo.cpp \
    criteria1DProject.cpp \
    water1D.cpp

HEADERS += \
    carbonNitrogenModel.h \
    criteria1DCase.h \
    criteria1DError.h \
    criteria1DMeteo.h \
    criteria1DProject.h \
    water1D.h

unix {
    target.path = /usr/lib
    INSTALLS += target
}

