#-----------------------------------------------------------
#
#   CRITERIA3D
#   3D agro-hydrological model
#   This project is part of ARPAE CRITERIA-3D distribution
#
#-----------------------------------------------------------

QT  += network widgets sql xml charts
greaterThan(QT_MAJOR_VERSION, 5): QT += core5compat openglwidgets

TEMPLATE = app
TARGET = CRITERIA3D

VERSION = 1.0.6

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

INCLUDEPATH +=  ./shared  \
                ../../agrolib/soilFluxes3D/header  \
                ../../agrolib/crit3dDate ../../agrolib/mathFunctions \
                ../../agrolib/crop ../../agrolib/soil ../../agrolib/meteo ../../agrolib/gis \
                ../../agrolib/interpolation ../../agrolib/solarRadiation \
                ../../agrolib/outputPoints ../../agrolib/soilWidget ../../agrolib/utilities  \
                ../../agrolib/dbMeteoPoints ../../agrolib/outputPoints ../../agrolib/dbMeteoGrid \
                ../../agrolib/proxyWidget ../../agrolib/project \
                ../../agrolib/graphics  ../../agrolib/commonChartElements ../../agrolib/commonDialogs \
                ../../agrolib/meteoWidget ../../agrolib/waterTable  \
                ../../src/snow ../../src/hydrall ../../src/rothCplusplus ../../src/project3D \
                ../../mapGraphics

CONFIG(debug, debug|release) {
    LIBS += -L../../agrolib/graphics/debug -lgraphics
    win32:{
        LIBS += -L../../mapGraphics/debug -lMapGraphics
    }
    unix:{
        LIBS += -L../../mapGraphics/release -lMapGraphics
    }
    LIBS += -L../../src/project3D/debug -lproject3D
    LIBS += -L../../agrolib/project/debug -lproject
    LIBS += -L../../agrolib/proxyWidget/debug -lproxyWidget
    LIBS += -L../../agrolib/meteoWidget/debug -lmeteoWidget
    LIBS += -L../../agrolib/commonDialogs/debug -lcommonDialogs
    LIBS += -L../../agrolib/dbMeteoGrid/debug -ldbMeteoGrid
    LIBS += -L../../agrolib/dbMeteoPoints/debug -ldbMeteoPoints
    LIBS += -L../../agrolib/outputPoints/debug -loutputPoints
    LIBS += -L../../agrolib/soilWidget/debug -lsoilWidget
    LIBS += -L../../agrolib/crop/debug -lcrop
    LIBS += -L../../agrolib/soil/debug -lsoil
    LIBS += -L../../agrolib/utilities/debug -lutilities
    LIBS += -L../../agrolib/waterTable/debug -lwaterTable
    LIBS += -L../../src/snow/debug -lsnow
    LIBS += -L../../src/hydrall/debug -lhydrall
    LIBS += -L../../src/rothCplusplus/debug -lrothCplusplus
    LIBS += -L../../agrolib/commonChartElements/debug -lcommonChartElements
    LIBS += -L../../agrolib/solarRadiation/debug -lsolarRadiation
    LIBS += -L../../agrolib/interpolation/debug -linterpolation
    LIBS += -L../../agrolib/meteo/debug -lmeteo
    LIBS += -L../../agrolib/gis/debug -lgis
    LIBS += -L../../agrolib/soilFluxes3D/debug -lsoilFluxes3D
    LIBS += -L../../agrolib/mathFunctions/debug -lmathFunctions
    LIBS += -L../../agrolib/crit3dDate/debug -lcrit3dDate

} else {
    LIBS += -L../../agrolib/graphics/release -lgraphics
    LIBS += -L../../mapGraphics/release -lMapGraphics
    LIBS += -L../../src/project3D/release -lproject3D
    LIBS += -L../../agrolib/project/release -lproject
    LIBS += -L../../agrolib/proxyWidget/release -lproxyWidget
    LIBS += -L../../agrolib/meteoWidget/release -lmeteoWidget
    LIBS += -L../../agrolib/commonDialogs/release -lcommonDialogs
    LIBS += -L../../agrolib/dbMeteoGrid/release -ldbMeteoGrid
    LIBS += -L../../agrolib/dbMeteoPoints/release -ldbMeteoPoints
    LIBS += -L../../agrolib/outputPoints/release -loutputPoints
    LIBS += -L../../agrolib/soilWidget/release -lsoilWidget
    LIBS += -L../../agrolib/crop/release -lcrop
    LIBS += -L../../agrolib/soil/release -lsoil
    LIBS += -L../../agrolib/utilities/release -lutilities
    LIBS += -L../../agrolib/waterTable/release -lwaterTable
    LIBS += -L../../src/snow/release -lsnow
    LIBS += -L../../src/hydrall/release -lhydrall
    LIBS += -L../../src/rothCplusplus/release -lrothCplusplus
    LIBS += -L../../agrolib/commonChartElements/release -lcommonChartElements
    LIBS += -L../../agrolib/solarRadiation/release -lsolarRadiation
    LIBS += -L../../agrolib/interpolation/release -linterpolation
    LIBS += -L../../agrolib/meteo/release -lmeteo
    LIBS += -L../../agrolib/gis/release -lgis
    LIBS += -L../../agrolib/soilFluxes3D/release -lsoilFluxes3D
    LIBS += -L../../agrolib/mathFunctions/release -lmathFunctions
    LIBS += -L../../agrolib/crit3dDate/release -lcrit3dDate
}


HEADERS += \
    dialogModelProcesses.h \
    mainwindow.h \
    criteria3DProject.h \
    dialogLoadState.h \
    dialogSnowSettings.h \
    geometry.h \
    glWidget.h \
    viewer3D.h

SOURCES += \
    dialogModelProcesses.cpp \
    mainwindow.cpp \
    criteria3DProject.cpp \
    dialogLoadState.cpp \
    dialogSnowSettings.cpp \
    geometry.cpp \
    glWidget.cpp \
    main.cpp \
    viewer3D.cpp


FORMS += mainwindow.ui

DISTFILES += \
    Criteria3D.ico

RESOURCES += Criteria3D.ico

win32:
{
    RC_ICONS = Criteria3D.ico
}
