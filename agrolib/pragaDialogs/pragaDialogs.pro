#------------------------------------------------------
#
#   pragaDialogs library
#   contains dialogs for PRAGA executable
#
#   This project is part of agrolib distribution
#
#------------------------------------------------------

QT  += core gui widgets sql xml

TEMPLATE = lib
CONFIG += staticlib
CONFIG += debug_and_release

DEFINES += _CRT_SECURE_NO_WARNINGS

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/pragaDialogs
    } else {
        TARGET = release/pragaDialogs
    }
}
win32:{
    TARGET = pragaDialogs
}

INCLUDEPATH +=  ../mathFunctions ../crit3dDate ../gis ../meteo ../phenology ../interpolation  \
                ../dbMeteoPoints ../dbMeteoGrid ../climate ../project

SOURCES += \
    dialogAddMissingStation.cpp \
    dialogAddRemoveDataset.cpp \
    dialogCellSize.cpp \
    dialogClimateFields.cpp \
    dialogDownloadMeteoData.cpp \
    dialogSelectDataset.cpp \
    dialogSeriesOnZones.cpp \
    dialogShiftData.cpp \
    dialogXMLComputation.cpp

HEADERS += \
    dialogAddMissingStation.h \
    dialogAddRemoveDataset.h \
    dialogCellSize.h \
    dialogClimateFields.h \
    dialogDownloadMeteoData.h \
    dialogSelectDataset.h \
    dialogSeriesOnZones.h \
    dialogShiftData.h \
    dialogXMLComputation.h

