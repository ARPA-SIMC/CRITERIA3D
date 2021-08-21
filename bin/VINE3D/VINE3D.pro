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

TARGET = VINE3D
TEMPLATE = app

DEFINES += VINE3D

INCLUDEPATH +=  ../../mapGraphics \
                ../../agrolib/crit3dDate ../../agrolib/mathFunctions ../../agrolib/gis ../../agrolib/meteo \
                ../../agrolib/interpolation ../../agrolib/solarRadiation ../../agrolib/soil  \
                ../../agrolib/soilFluxes3D/header ../../agrolib/crop ../../agrolib/grapevine \
                ../../agrolib/utilities ../../agrolib/dbMeteoPoints ../../agrolib/dbMeteoGrid \
                ../../agrolib/importDataXML ../../agrolib/project \
                ../../agrolib/graphics  ../../agrolib/commonDialogs \
                ../../mapGraphics ../../agrolib/meteoWidget \
                ../CRITERIA3D/shared

CONFIG += debug_and_release


CONFIG(debug, debug|release) {
    LIBS += -L../../agrolib/graphics/debug -lgraphics
    LIBS += -L../../mapGraphics/debug -lMapGraphics
    LIBS += -L../../agrolib/project/debug -lproject
    LIBS += -L../../agrolib/importDataXML/debug -limportDataXML
    LIBS += -L../../agrolib/meteoWidget/debug -lmeteoWidget
    LIBS += -L../../agrolib/commonDialogs/debug -commonDialogs
    LIBS += -L../../agrolib/soil/debug -lsoil
    LIBS += -L../../agrolib/soilFluxes3D/debug -lsoilFluxes3D
    LIBS += -L../../agrolib/grapevine/debug -lgrapevine
    LIBS += -L../../agrolib/dbMeteoGrid/debug -ldbMeteoGrid
    LIBS += -L../../agrolib/dbMeteoPoints/debug -ldbMeteoPoints
    LIBS += -L../../agrolib/utilities/debug -lutilities
    LIBS += -L../../agrolib/solarRadiation/debug -lsolarRadiation
    LIBS += -L../../agrolib/interpolation/debug -linterpolation
    LIBS += -L../../agrolib/meteo/debug -lmeteo
    LIBS += -L../../agrolib/gis/debug -lgis
    LIBS += -L../../agrolib/crit3dDate/debug -lcrit3dDate
    LIBS += -L../../agrolib/mathFunctions/debug -lmathFunctions
} else {
    LIBS += -L../../agrolib/graphics/release -lgraphics
    LIBS += -L../../mapGraphics/release -lMapGraphics
    LIBS += -L../../agrolib/project/release -lproject
    LIBS += -L../../agrolib/importDataXML/release -limportDataXML
    LIBS += -L../../agrolib/meteoWidget/release -lmeteoWidget
    LIBS += -L../../agrolib/commonDialogs/release -lcommonDialogs
    LIBS += -L../../agrolib/soil/release -lsoil
    LIBS += -L../../agrolib/soilFluxes3D/release -lsoilFluxes3D
    LIBS += -L../../agrolib/grapevine/release -lgrapevine
    LIBS += -L../../agrolib/dbMeteoGrid/release -ldbMeteoGrid
    LIBS += -L../../agrolib/dbMeteoPoints/release -ldbMeteoPoints
    LIBS += -L../../agrolib/utilities/release -lutilities
    LIBS += -L../../agrolib/solarRadiation/release -lsolarRadiation
    LIBS += -L../../agrolib/interpolation/release -linterpolation
    LIBS += -L../../agrolib/meteo/release -lmeteo
    LIBS += -L../../agrolib/gis/release -lgis
    LIBS += -L../../agrolib/crit3dDate/release -lcrit3dDate
    LIBS += -L../../agrolib/mathFunctions/release -lmathFunctions
}

SOURCES += \
    atmosphere.cpp \
    dataHandler.cpp \
    disease.cpp \
    main.cpp \
    modelCore.cpp \
    plant.cpp \
    vine3DShell.cpp \
    waterBalance.cpp \
    vine3DProject.cpp \
    mainWindow.cpp \
    ../CRITERIA3D/shared/project3D.cpp


HEADERS += \
    atmosphere.h \
    dataHandler.h \
    disease.h \
    modelCore.h \
    plant.h \
    vine3DShell.h \
    waterBalance.h \
    vine3DProject.h \
    mainWindow.h \
    ../CRITERIA3D/shared/project3D.h

FORMS += \
    mainWindow.ui \
