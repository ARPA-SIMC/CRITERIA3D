#-----------------------------------------------------
#
#   VINE3D
#   Simulations of vineyard ecosystem
#   with 3D soil water balance
#
#   this project is part of CRITERIA-3D distribution
#
#-----------------------------------------------------

QT  += core gui widgets charts xml sql network
greaterThan(QT_MAJOR_VERSION, 5): QT += core5compat

TARGET = VINE3D
TEMPLATE = app

DEFINES += VINE3D

INCLUDEPATH +=  ../../mapGraphics \
                ../../agrolib/crit3dDate ../../agrolib/mathFunctions ../../agrolib/soilFluxes3D/header \
                ../../agrolib/gis ../../agrolib/meteo ../../agrolib/interpolation \
                ../../agrolib/solarRadiation ../../agrolib/soil  \
                ../../agrolib/crop ../../agrolib/grapevine ../../agrolib/outputPoints \
                ../../agrolib/utilities ../../agrolib/dbMeteoPoints ../../agrolib/dbMeteoGrid \
                ../../agrolib/proxyWidget ../../agrolib/waterTable \
                ../../agrolib/commonDialogs ../../agrolib/project \
                ../../agrolib/graphics ../../agrolib/commonChartElements ../../agrolib/meteoWidget  \
                ../CRITERIA3D/shared

CONFIG += debug_and_release


CONFIG(debug, debug|release) {
    LIBS += -L../../agrolib/graphics/debug -lgraphics
    win32:{
        LIBS += -L../../mapGraphics/debug -lMapGraphics
    }
    unix:{
        LIBS += -L../../mapGraphics/release -lMapGraphics
    }
    LIBS += -L../../agrolib/project/debug -lproject
    LIBS += -L../../agrolib/proxyWidget/debug -lproxyWidget
    LIBS += -L../../agrolib/meteoWidget/debug -lmeteoWidget
    LIBS += -L../../agrolib/commonChartElements/debug -lcommonChartElements
    LIBS += -L../../agrolib/commonDialogs/debug -lcommonDialogs
    LIBS += -L../../agrolib/dbMeteoGrid/debug -ldbMeteoGrid
    LIBS += -L../../agrolib/dbMeteoPoints/debug -ldbMeteoPoints
    LIBS += -L../../agrolib/outputPoints/debug -loutputPoints
    LIBS += -L../../agrolib/utilities/debug -lutilities
    LIBS += -L../../agrolib/waterTable/debug -lwaterTable
    LIBS += -L../../agrolib/grapevine/debug -lgrapevine
    LIBS += -L../../agrolib/soil/debug -lsoil
    LIBS += -L../../agrolib/crop/debug -lcrop
    LIBS += -L../../agrolib/solarRadiation/debug -lsolarRadiation
    LIBS += -L../../agrolib/interpolation/debug -linterpolation
    LIBS += -L../../agrolib/meteo/debug -lmeteo
    LIBS += -L../../agrolib/gis/debug -lgis
    LIBS += -L../../agrolib/soilFluxes3D/debug -lsoilFluxes3D
    LIBS += -L../../agrolib/crit3dDate/debug -lcrit3dDate
    LIBS += -L../../agrolib/mathFunctions/debug -lmathFunctions
} else {
    LIBS += -L../../agrolib/graphics/release -lgraphics
    LIBS += -L../../mapGraphics/release -lMapGraphics
    LIBS += -L../../agrolib/project/release -lproject
    LIBS += -L../../agrolib/proxyWidget/release -lproxyWidget
    LIBS += -L../../agrolib/meteoWidget/release -lmeteoWidget
    LIBS += -L../../agrolib/commonChartElements/release -lcommonChartElements
    LIBS += -L../../agrolib/commonDialogs/release -lcommonDialogs
    LIBS += -L../../agrolib/dbMeteoGrid/release -ldbMeteoGrid
    LIBS += -L../../agrolib/dbMeteoPoints/release -ldbMeteoPoints
    LIBS += -L../../agrolib/outputPoints/release -loutputPoints
    LIBS += -L../../agrolib/utilities/release -lutilities
    LIBS += -L../../agrolib/waterTable/release -lwaterTable
    LIBS += -L../../agrolib/grapevine/release -lgrapevine
    LIBS += -L../../agrolib/soil/release -lsoil
    LIBS += -L../../agrolib/crop/release -lcrop
    LIBS += -L../../agrolib/solarRadiation/release -lsolarRadiation
    LIBS += -L../../agrolib/interpolation/release -linterpolation
    LIBS += -L../../agrolib/meteo/release -lmeteo
    LIBS += -L../../agrolib/gis/release -lgis
    LIBS += -L../../agrolib/soilFluxes3D/release -lsoilFluxes3D
    LIBS += -L../../agrolib/crit3dDate/release -lcrit3dDate
    LIBS += -L../../agrolib/mathFunctions/release -lmathFunctions
}

SOURCES += \
    ../CRITERIA3D/shared/project3D.cpp \
    ../CRITERIA3D/shared/dialogWaterFluxesSettings.cpp \
    atmosphere.cpp \
    dataHandler.cpp \
    disease.cpp \
    main.cpp \
    modelCore.cpp \
    plant.cpp \
    vine3DShell.cpp \
    waterBalance.cpp \
    vine3DProject.cpp \
    mainWindow.cpp


HEADERS += \
    ../CRITERIA3D/shared/project3D.h \
    ../CRITERIA3D/shared/dialogWaterFluxesSettings.h \
    atmosphere.h \
    dataHandler.h \
    disease.h \
    modelCore.h \
    plant.h \
    waterBalance.h \
    vine3DProject.h \
    mainWindow.h


FORMS += \
    mainWindow.ui \
