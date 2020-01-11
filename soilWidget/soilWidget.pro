#----------------------------------------------------
#
#   Soil Widget library
#   This project is part of CRITERIA-3D distribution
#
#   It requires Qwt library
#   https://qwt.sourceforge.io/index.html
#   Windows: set QWT_ROOT in environment variables
#
#----------------------------------------------------

QT  += widgets sql

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

INCLUDEPATH += ../crit3dDate ../mathFunctions ../soil ../utilities

unix:{
    INCLUDEPATH += /usr/include/qwt/
}
macx:{
    INCLUDEPATH += /usr/local/opt/qwt/lib/qwt.framework/Headers/
}

SOURCES += \
    barHorizon.cpp \
    curvePicker.cpp \
    soilTable.cpp \
    soilWidget.cpp \
    tabHorizons.cpp \
    tabWaterRetentionCurve.cpp \
    tabWaterRetentionData.cpp \
    tabHydraulicConductivityCurve.cpp \
    tableDelegate.cpp \
    tableDelegateWaterRetention.cpp \
    curvePanner.cpp \
    dialogNewSoil.cpp \
    tableWaterRetention.cpp

HEADERS += \
    barHorizon.h \
    curvePicker.h \
    soilTable.h \
    soilWidget.h \
    tabHorizons.h \
    tabWaterRetentionCurve.h \
    tabWaterRetentionData.h \
    tabHydraulicConductivityCurve.h \
    tableDelegate.h \
    tableDelegateWaterRetention.h \
    curvePanner.h \
    tableWidgetItem.h \
    dialogNewSoil.h \
    tableWaterRetention.h

win32:{
    include($$(QWT_ROOT)\features\qwt.prf)
}
unix:{
    include(/usr/lib/x86_64-linux-gnu/qt5/mkspecs/features/qwt.prf)
}
macx:{
    include(/usr/local/opt/qwt/features/qwt.prf)
}
