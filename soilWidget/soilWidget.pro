#----------------------------------------------------
#
#   Soil Widget library
#   This project is part of CRITERIA-3D distribution
#
#
#----------------------------------------------------

QT  += widgets sql charts

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/soilWidget
    } else {
        TARGET = release/soilWidget
    }
}
macx:{
    CONFIG(debug, debug|release) {
        TARGET = debug/soilWidget
    } else {
        TARGET = release/soilWidget
    }
}
win32:{
    TARGET = soilWidget
}

INCLUDEPATH += ../crit3dDate ../mathFunctions ../soil ../utilities ../commonChartElements

SOURCES += \
    barHorizon.cpp \
    soilTable.cpp \
    soilWidget.cpp \
    tabHorizons.cpp \
    tabWaterRetentionCurve.cpp \
    tabWaterRetentionData.cpp \
    tabHydraulicConductivityCurve.cpp \
    tableDelegate.cpp \
    tableDelegateWaterRetention.cpp \
    dialogNewSoil.cpp \
    tableWaterRetention.cpp

HEADERS += \
    barHorizon.h \
    soilTable.h \
    soilWidget.h \
    tabHorizons.h \
    tabWaterRetentionCurve.h \
    tabWaterRetentionData.h \
    tabHydraulicConductivityCurve.h \
    tableDelegate.h \
    tableDelegateWaterRetention.h \
    tableWidgetItem.h \
    dialogNewSoil.h \
    tableWaterRetention.h
