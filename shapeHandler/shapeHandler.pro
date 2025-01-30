#--------------------------------------------------------
#
#   Shapefile handler
#   This project is part of CRITERIA-3D distribution
#
#   The library includes code from:
#   - shapelib of Frank Warmerdam
#   http://shapelib.maptools.org/
#
#   - shapeObject.cpp of Erik Svensson
#   https://github.com/blueluna/shapes
#
#--------------------------------------------------------

QT    -= core gui

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/shapeHandler
    } else {
        TARGET = release/shapeHandler
    }
}
win32:{
    TARGET = shapeHandler
}

TEMPLATE = lib
CONFIG += staticlib

DEFINES += _CRT_SECURE_NO_WARNINGS

INCLUDEPATH =  shapelib  ../mathFunctions

SOURCES += \
    shapelib/dbfopen.c      \
    shapelib/safileio.c     \
    shapelib/sbnsearch.c    \
    shapelib/shpopen.c      \
    shapelib/shptree.c      \
    shapeObject.cpp         \
    shapeHandler.cpp


HEADERS += \
    shapelib/shapefil.h     \
    shapeHandler.h          \
    shapeObject.h

