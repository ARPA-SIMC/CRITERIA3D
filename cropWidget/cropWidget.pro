#----------------------------------------------------
#
#   Crop Widget library
#   This project is part of CRITERIA-3D distribution
#
#----------------------------------------------------

QT  += widgets sql xml charts printsupport

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/cropWidget
    } else {
        TARGET = release/cropWidget
    }
}
macx:{
    CONFIG(debug, debug|release) {
        TARGET = debug/cropWidget
    } else {
        TARGET = release/cropWidget
    }
}
win32:{
    TARGET = cropWidget
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../utilities ../gis ../meteo ../dbMeteoGrid  \
            ../soil ../carbonNitrogen ../crop ../qcustomplot ../criteriaModel ../commonDialogs \
            ../commonChartElements ../meteoWidget ../soilWidget


SOURCES += \
    cropWidget.cpp \
    dialogNewCrop.cpp \
    dialogNewProject.cpp \
    tabCarbonNitrogen.cpp \
    tabIrrigation.cpp \
    tabLAI.cpp \
    tabRootDensity.cpp \
    tabRootDepth.cpp \
    tabWaterContent.cpp

HEADERS += \
    cropWidget.h \
    dialogNewCrop.h \
    dialogNewProject.h \
    tabCarbonNitrogen.h \
    tabIrrigation.h \
    tabLAI.h \
    tabRootDensity.h \
    tabRootDepth.h \
    tabWaterContent.h
