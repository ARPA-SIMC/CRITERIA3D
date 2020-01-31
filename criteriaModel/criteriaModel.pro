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

QT      += core sql
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
                ../soil ../crop ../utilities

SOURCES += \
    dbMeteoCriteria1D.cpp \
    modelCore.cpp \
    water1D.cpp \
    croppingSystem.cpp \
    criteriaModel.cpp

HEADERS += \
    dbMeteoCriteria1D.h \
    modelCore.h \
    water1D.h \
    croppingSystem.h \
    criteriaModel.h

unix {
    target.path = /usr/lib
    INSTALLS += target
}

