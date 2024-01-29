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
CONFIG += c++11 c++14 c++17


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
                ../utilities ../dbMeteoPoints ../dbMeteoGrid ../phenology

SOURCES += \
    climate.cpp \
    crit3dDroughtList.cpp \
    crit3dPhenologyList.cpp \
    elaborationSettings.cpp \
    crit3dClimate.cpp \
    dbClimate.cpp \
    crit3dClimateList.cpp \
    crit3dElabList.cpp \
    crit3dAnomalyList.cpp

HEADERS += \
    crit3dDroughtList.h \
    crit3dPhenologyList.h \
    dbClimate.h \
    climate.h \
    elaborationSettings.h \
    crit3dClimate.h \
    crit3dClimateList.h \
    crit3dElabList.h \
    crit3dAnomalyList.h
