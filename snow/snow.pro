#-----------------------------------------------------
#
#   snow library
#   Compute snow accumulation and melt
#   with energy balance
#
#   This library is part of CRITERIA-3D distribution
#
#-----------------------------------------------------

QT += core
QT -= gui


TARGET = snow
CONFIG += staticlib

TEMPLATE = lib

INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo ../solarRadiation

SOURCES += \
    snowMaps.cpp \
    snowPoint.cpp

HEADERS += \
    snowMaps.h \
    snowPoint.h
