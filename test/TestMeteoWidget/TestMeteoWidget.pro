#-----------------------------------------------------------
#
#   TestMeteoWidget
#   This project is part of CRITERIA3D distribution
#
#-----------------------------------------------------------

QT       += core gui widgets charts sql

TEMPLATE = app

CONFIG += debug_and_release

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/TestMeteoWidget
    } else {
        TARGET = release/TestMeteoWidget
    }
}
macx:{
    CONFIG(debug, debug|release) {
        TARGET = debug/TestMeteoWidget
    } else {
        TARGET = release/TestMeteoWidget
    }
}
win32:{
    TARGET = TestMeteoWidget
}

INCLUDEPATH +=  ../../agrolib/crit3dDate ../../agrolib/mathFunctions ../../agrolib/gis \
                ../../agrolib/meteo ../../agrolib/utilities ../../agrolib/interpolation ../../agrolib/dbMeteoPoints ../../agrolib/meteoWidget

CONFIG(debug, debug|release) {
    LIBS += -L../../agrolib/meteoWidget/debug -lmeteoWidget
    LIBS += -L../../agrolib/dbMeteoPoints/debug -ldbMeteoPoints
    LIBS += -L../../agrolib/utilities/debug -lutilities
    LIBS += -L../../agrolib/meteo/debug -lmeteo
    LIBS += -L../../agrolib/gis/debug -lgis
    LIBS += -L../../agrolib/mathFunctions/debug -lmathFunctions
    LIBS += -L../../agrolib/crit3dDate/debug -lcrit3dDate
} else {

    LIBS += -L../../agrolib/meteoWidget/release -lmeteoWidget
    LIBS += -L../../agrolib/dbMeteoPoints/release -ldbMeteoPoints
    LIBS += -L../../agrolib/utilities/release -lutilities
    LIBS += -L../../agrolib/meteo/release -lmeteo
    LIBS += -L../../agrolib/gis/release -lgis
    LIBS += -L../../agrolib/mathFunctions/release -lmathFunctions
    LIBS += -L../../agrolib/crit3dDate/release -lcrit3dDate
}


SOURCES += \
        console.cpp \
        main.cpp
 

HEADERS += \
    console.h
