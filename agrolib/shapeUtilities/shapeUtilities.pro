#--------------------------------------------------------
#
#   shapeUtilities
#   This project is part of CRITERIA-3D distribution
#
#--------------------------------------------------------

QT    += sql widgets

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

DEFINES += _CRT_SECURE_NO_WARNINGS

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/shapeUtilities
    } else {
        TARGET = release/shapeUtilities
    }
}
win32:{
    TARGET = shapeUtilities
}

# parallel computing settings
include($$absolute_path(../parallel.pri))

INCLUDEPATH =  ../crit3dDate ../mathFunctions ../gis ../shapeHandler ../utilities ../commonDialogs

SOURCES += \
    shapeFromCsv.cpp \
    shapeToRaster.cpp    \
    shapeUtilities.cpp   \
    unitCropMap.cpp      \
    zonalStatistic.cpp   \
    ../commonDialogs/formInfo.cpp


HEADERS += \
    shapeFromCsv.h \
    shapeToRaster.h    \
    shapeUtilities.h   \
    unitCropMap.h      \
    zonalStatistic.h    \
    ../commonDialogs/formInfo.h


