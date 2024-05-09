#-----------------------------------------------------
#
#   pragaProject library
#   contains project modules for PRAGA executable
#
#   This project is part of agrolib distribution
#
#-----------------------------------------------------

QT       += core gui widgets charts sql xml
greaterThan(QT_MAJOR_VERSION, 5): QT += core5compat

TEMPLATE = lib
CONFIG += staticlib
CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

DEFINES += _CRT_SECURE_NO_WARNINGS
DEFINES += NETCDF

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/pragaProject
    } else {
        TARGET = release/pragaProject
    }
}
win32:{
    TARGET = pragaProject
}

INCLUDEPATH +=  ../crit3dDate ../mathFunctions ../phenology ../meteo ../gis  \
                ../drought ../interpolation ../solarRadiation ../utilities  \
                ../outputPoints ../dbMeteoPoints ../dbMeteoGrid ../meteoWidget  \
                ../proxyWidget ../pointStatisticsWidget ../homogeneityWidget ../synchronicityWidget ../climate ../netcdfHandler  \
                ../graphics ../commonDialogs ../commonChartElements ../pragaDialogs ../inOutDataXML ../waterTable ../project


SOURCES += \
    dialogPragaProject.cpp \
    dialogMeteoComputation.cpp \
    dialogPragaSettings.cpp \
    dialogAnomaly.cpp \
    pragaMeteoMaps.cpp \
    saveClimaLayout.cpp \
    pragaProject.cpp \
    pragaShell.cpp


HEADERS  += \
    dialogPragaProject.h \
    dialogMeteoComputation.h \
    dialogPragaSettings.h \
    dialogAnomaly.h \
    pragaMeteoMaps.h \
    saveClimaLayout.h \
    pragaProject.h \
    pragaShell.h

