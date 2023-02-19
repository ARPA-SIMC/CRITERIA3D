#===========================================================
#
# MapGraphics (modified)
# A tile-based "slippy map" library written in C++/Qt
# BSD licensed (see LICENSE)
# fork of:
# https://github.com/raptorswing/MapGraphics
#
#===========================================================

QT       += widgets network sql
greaterThan(QT_MAJOR_VERSION, 5): QT += core5compat

TEMPLATE = lib
CONFIG += staticlib

unix:{
    CONFIG(debug, debug|release) {
        TARGET = release/MapGraphics
    } else {
        TARGET = release/MapGraphics
    }
}
win32:{
    TARGET = MapGraphics
}

# uncomment for dynamic linking
#DEFINES += MAPGRAPHICS_LIBRARY

SOURCES += \
    ArrowObject.cpp \
    SquareObject.cpp \
    tileSources/CompositeTileSource.cpp \
    tileSources/GridTileSource.cpp \
    tileSources/WebTileSource.cpp \
    guts/PrivateQGraphicsScene.cpp \
    guts/PrivateQGraphicsObject.cpp \
    guts/Conversions.cpp \
    guts/MapTileGraphicsObject.cpp \
    guts/PrivateQGraphicsView.cpp \
    guts/MapGraphicsNetwork.cpp \
    guts/MapTileLayerListModel.cpp \
    guts/MapTileSourceDelegate.cpp \
    guts/PrivateQGraphicsInfoSource.cpp \
    MapGraphicsScene.cpp \
    MapGraphicsObject.cpp \
    MapGraphicsView.cpp \
    MapTileSource.cpp \
    PolygonObject.cpp \
    Position.cpp \
    LineObject.cpp \
    CircleObject.cpp


HEADERS += \
    ArrowObject.h \
    SquareObject.h \
    tileSources/CompositeTileSource.h \
    tileSources/GridTileSource.h \
    tileSources/WebTileSource.h \
    guts/MapTileLayerListModel.h \
    guts/MapTileSourceDelegate.h \
    guts/MapGraphicsNetwork.h \
    guts/PrivateQGraphicsInfoSource.h \
    guts/PrivateQGraphicsScene.h \
    guts/PrivateQGraphicsObject.h \
    guts/MapTileGraphicsObject.h \
    guts/PrivateQGraphicsView.h \
    guts/Conversions.h \
    MapGraphics_global.h \
    MapGraphicsObject.h \
    MapGraphicsView.h \
    MapTileSource.h \
    MapGraphicsScene.h \
    PolygonObject.h \
    Position.h \
    LineObject.h \
    CircleObject.h

