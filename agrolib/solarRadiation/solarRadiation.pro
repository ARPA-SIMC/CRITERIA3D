#-------------------------------------------------------
#
#   solarRadiation library
#   This project is part of ARPAE agrolib distribution
#
#   It uses code from:
#   G_calc_solar_position() by Markus Neteler
#
#-------------------------------------------------------

QT       -= core gui

TEMPLATE = lib
CONFIG += staticlib
CONFIG += c++17

win32:{
    QMAKE_CXXFLAGS += -openmp -GL
    QMAKE_LFLAGS   += -LTCG
}
unix:{
    QMAKE_CXXFLAGS += -fopenmp #-flto
    QMAKE_LFLAGS += -fopenmp #-flto
}
macx:{
    QMAKE_CXXFLAGS += -fopenmp #-flto
    QMAKE_LFLAGS += -fopenmp #-flto
}

CONFIG += debug_and_release

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/solarRadiation
    } else {
        TARGET = release/solarRadiation
    }
}
win32:{
    TARGET = solarRadiation
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo

SOURCES += \
    solPos.cpp \
    solarRadiation.cpp \
    sunPosition.cpp \
    radiationSettings.cpp \
    transmissivity.cpp

HEADERS += \
    solPos.h \
    sunPosition.h \
    radiationSettings.h \
    radiationDefinitions.h \
    solarRadiation.h \
    transmissivity.h

