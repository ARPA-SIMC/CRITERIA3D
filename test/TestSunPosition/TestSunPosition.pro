#-----------------------------------------------------------------
#
#   TestSunPosition
#   It shows the current sunrise and sunset at Bologna and Madrid
#
#   This project is part of CRITERIA3D distribution
#
#-----------------------------------------------------------------

QT       += core
QT       -= gui

TEMPLATE = app
CONFIG   += console
CONFIG   -= app_bundle

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/TestSunPosition
    } else {
        TARGET = release/TestSunPosition
    }
}
win32:{
    TARGET = TestSunPosition
}

INCLUDEPATH += ../../agrolib/crit3dDate ../../agrolib/mathFunctions ../../agrolib/gis ../../agrolib/solarRadiation

CONFIG += debug_and_release

CONFIG(debug, debug|release) {
    LIBS += -L../../agrolib/solarRadiation/debug -lsolarRadiation
    LIBS += -L../../agrolib/gis/debug -lgis
    LIBS += -L../../agrolib/mathFunctions/debug -lmathFunctions
    LIBS += -L../../agrolib/crit3dDate/debug -lcrit3dDate
} else {
    LIBS += -L../../agrolib/solarRadiation/release -lsolarRadiation
    LIBS += -L../../agrolib/gis/release -lgis
    LIBS += -L../../agrolib/mathFunctions/release -lmathFunctions
    LIBS += -L../../agrolib/crit3dDate/release -lcrit3dDate
}


SOURCES += main.cpp
