#-----------------------------------------------------
#
#   GDAL handler library
#   This project is part of CRITERIA-3D distribution
#
#-----------------------------------------------------

QT       -= gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release

DEFINES += _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE


unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/gdalHandler
    } else {
        TARGET = release/gdalHandler
    }
}
else:{
    TARGET = gdalHandler
}


INCLUDEPATH += ../mathFunctions ../crit3dDate ../gis ../shapeHandler ../shapeHandler/shapelib


SOURCES += \
    gdalExtensions.cpp \
    gdalRasterFunctions.cpp \
    gdalShapeFunctions.cpp \

HEADERS += \
    gdalExtensions.h \
    gdalRasterFunctions.h \
    gdalShapeFunctions.h \


include(../gdal.pri)
