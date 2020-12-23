#-----------------------------------------------------
#
#   climate library
#   This project is part of CRITERIA-3D distribution
#
#-----------------------------------------------------

QT       -= gui
QT       += sql xml

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
QMAKE_CXXFLAGS += -std=c++11


unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/climate
    } else {
        TARGET = release/climate
    }
}
win32:{
    TARGET = climate
}


INCLUDEPATH +=  ../crit3dDate ../mathFunctions ../gis ../meteo ../interpolation \
                ../utilities ../dbMeteoPoints ../dbMeteoGrid

SOURCES += \
    climate.cpp \
    elaborationSettings.cpp \
    crit3dClimate.cpp \
    dbClimate.cpp \
    crit3dClimateList.cpp \
    crit3dElabList.cpp \
    crit3dAnomalyList.cpp

HEADERS += \
    dbClimate.h \
    climate.h \
    elaborationSettings.h \
    crit3dClimate.h \
    crit3dClimateList.h \
    crit3dElabList.h \
    crit3dAnomalyList.h
