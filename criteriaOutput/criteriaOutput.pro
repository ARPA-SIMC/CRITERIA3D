#-----------------------------------------------------------
#
#   criteriaOutput
#   post-processing of CRITERIA-1D output
#   to csv or shapefile or aggregation (csv) file
#
#   This project is part of CRITERIA-1D distribution
#
#-----------------------------------------------------------

QT   -= gui
QT   += sql

TEMPLATE = lib
CONFIG += staticlib

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/criteriaOutput
    } else {
        TARGET = release/criteriaOutput
    }
}
macx:{
    CONFIG(debug, debug|release) {
        TARGET = debug/criteriaOutput
    } else {
        TARGET = release/criteriaOutput
    }
}
win32:{
    TARGET = criteriaOutput
}

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS
DEFINES += _CRT_SECURE_NO_WARNINGS


INCLUDEPATH +=  ../crit3dDate ../mathFunctions ../gis ../utilities \
                ../shapeHandler ../netcdfHandler ../shapeUtilities

SOURCES += \
    ../crop/cropDbQuery.cpp \
    criteriaAggregationVariable.cpp \
    criteriaOutputElaboration.cpp \
    criteriaOutputProject.cpp \
    criteriaOutputVariable.cpp

HEADERS += \
    ../crop/cropDbQuery.h \
    criteriaAggregationVariable.h \
    criteriaOutputElaboration.h \
    criteriaOutputProject.h \
    criteriaOutputVariable.h


# comment to compile without GDAL library
CONFIG += GDAL

GDAL:{
    DEFINES += GDAL
    INCLUDEPATH += ../gdalHandler
    include(../gdal.pri)
    }


