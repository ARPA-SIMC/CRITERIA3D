#-----------------------------------------------------
#
#   project library
#   This project is part of ARPAE agrolib distribution
#
#-----------------------------------------------------

QT  += widgets charts sql xml

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

DEFINES += _CRT_SECURE_NO_WARNINGS

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/project
    } else {
        TARGET = release/project
    }
}
win32:{
    TARGET = project
}


INCLUDEPATH += ../crit3dDate ../mathFunctions ../gis ../meteo   \
            ../solarRadiation ../interpolation ../utilities ../netcdfHandler \
            ../dbMeteoPoints ../outputPoints ../dbMeteoGrid ../meteoWidget \
            ../commonDialogs ../commonChartElements ../climate \
            ../proxyWidget ../waterTable


SOURCES += \
    aggregation.cpp \
    dialogInterpolation.cpp \
    dialogPointDeleteData.cpp \
    dialogPointProperties.cpp \
    dialogProject.cpp \
    dialogRadiation.cpp \
    dialogSelection.cpp \
    dialogSelectionMeteoPoint.cpp \
    dialogSettings.cpp \
    interpolationCmd.cpp \
    meteoMaps.cpp \
    project.cpp \
    shell.cpp


HEADERS += \
    aggregation.h \
    dialogInterpolation.h \
    dialogPointDeleteData.h \
    dialogPointProperties.h \
    dialogProject.h \
    dialogRadiation.h \
    dialogSelection.h \
    dialogSelectionMeteoPoint.h \
    dialogSettings.h \
    interpolationCmd.h \
    meteoMaps.h \
    project.h \
    shell.h

FORMS +=

