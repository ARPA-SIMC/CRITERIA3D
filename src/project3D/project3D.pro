#---------------------------------------------------
#
#   project3D library
#   This project is part of CRITERIA3D distribution
#
#---------------------------------------------------

QT   += widgets sql xml charts

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

#DEFINES += _CRT_SECURE_NO_WARNINGS


unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/project3D
    } else {
        TARGET = release/project3D
    }
}
win32:{
    TARGET = project3D
}

INCLUDEPATH +=  ../../agrolib/soilFluxes3D  \
                ../../agrolib/mathFunctions ../../agrolib/crit3dDate ../../agrolib/soil ../../agrolib/crop \
                ../../agrolib/gis ../../agrolib/meteo ../../agrolib/utilities  \
                ../../agrolib/solarRadiation ../../agrolib/interpolation  ../../agrolib/proxyWidget \
                ../../agrolib/dbMeteoPoints ../../agrolib/dbMeteoGrid ../../agrolib/commonChartElements \
                ../../agrolib/outputPoints ../../agrolib/waterTable ../../agrolib/project


SOURCES += \
project3D.cpp \
dialogWaterFluxesSettings.cpp 


HEADERS += \
project3D.h \
dialogWaterFluxesSettings.h

