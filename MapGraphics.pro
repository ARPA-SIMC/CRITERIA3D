#===========================================================
#
# MapGraphics (modified)
# A tile-based "slippy map" library written in C++/Qt
# BSD licensed (see LICENSE)
# https://github.com/raptorswing/MapGraphics
#
#===========================================================

QT       += widgets network sql

TEMPLATE = lib
QMAKE_CXXFLAGS += -std=c++11

unix:{
    CONFIG += release
    TARGET = release/MapGraphics
}
win32:{
    TARGET = MapGraphics
}

DEFINES += MAPGRAPHICS_LIBRARY


SOURCES += MapGraphicsScene.cpp \
    MapGraphicsObject.cpp \
    MapGraphicsView.cpp \
    guts/PrivateQGraphicsScene.cpp \
    guts/PrivateQGraphicsObject.cpp \
    guts/Conversions.cpp \
    MapTileSource.cpp \
    tileSources/GridTileSource.cpp \
    guts/MapTileGraphicsObject.cpp \
    guts/PrivateQGraphicsView.cpp \
    tileSources/OSMTileSource.cpp \
    guts/MapGraphicsNetwork.cpp \
    tileSources/CompositeTileSource.cpp \
    guts/MapTileLayerListModel.cpp \
    guts/MapTileSourceDelegate.cpp \
#    guts/CompositeTileSourceConfigurationWidget.cpp \
    CircleObject.cpp \
    guts/PrivateQGraphicsInfoSource.cpp \
    PolygonObject.cpp \
    Position.cpp \
    LineObject.cpp

HEADERS += MapGraphicsScene.h\
        MapGraphics_global.h \
    MapGraphicsObject.h \
    MapGraphicsView.h \
    guts/PrivateQGraphicsScene.h \
    guts/PrivateQGraphicsObject.h \
    guts/Conversions.h \
    MapTileSource.h \
    tileSources/GridTileSource.h \
    guts/MapTileGraphicsObject.h \
    guts/PrivateQGraphicsView.h \
    tileSources/OSMTileSource.h \
    guts/MapGraphicsNetwork.h \
    tileSources/CompositeTileSource.h \
    guts/MapTileLayerListModel.h \
    guts/MapTileSourceDelegate.h \
#    guts/CompositeTileSourceConfigurationWidget.h \
    CircleObject.h \
    guts/PrivateQGraphicsInfoSource.h \
    PolygonObject.h \
    Position.h \
    LineObject.h


