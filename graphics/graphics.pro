#------------------------------------------------------
#
#   graphics library
#   colorLegend, markers, raster and shape
#
#   This project is part of ARPAE agrolib distribution
#
#------------------------------------------------------

QT  += widgets

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

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
    mapGraphicsRasterUtm.cpp \
    mapGraphicsShapeObject.cpp \
    rubberBand.cpp \
    squareMarker.cpp \
    stationMarker.cpp


HEADERS += \
    colorLegend.h \
    mapGraphicsRasterObject.h \
    mapGraphicsRasterUtm.h \
    mapGraphicsShapeObject.h \
    rubberBand.h \
    squareMarker.h \
    stationMarker.h

