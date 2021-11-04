#------------------------------------------------------
#
#   graphics library
#   contains graphics objects:
#   colorLegend, markers, raster and shape
#
#   This project is part of CRITERIA-3D distribution
#
#------------------------------------------------------

QT  += core gui widgets

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
QMAKE_CXXFLAGS += -std=c++11

DEFINES += _CRT_SECURE_NO_WARNINGS

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/graphics
    } else {
        TARGET = release/graphics
    }
}
win32:{
    TARGET = graphics
}


INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo \
               ../shapeHandler ../../mapGraphics


SOURCES += \
    colorLegend.cpp \
    mapGraphicsRasterObject.cpp \
    mapGraphicsShapeObject.cpp \
    rubberBand.cpp \
    stationMarker.cpp


HEADERS += \
    colorLegend.h \
    mapGraphicsRasterObject.h \
    mapGraphicsShapeObject.h \
    rubberBand.h \
    stationMarker.h

