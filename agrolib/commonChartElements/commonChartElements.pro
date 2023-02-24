#------------------------------------------------------
#
#   common elements used with Qt Charts library
#
#   This project is part of agrolib distribution
#
#------------------------------------------------------

QT  += gui widgets charts

TEMPLATE = lib
CONFIG += staticlib
CONFIG += debug_and_release
CONFIG += c++14 c++17

DEFINES += _CRT_SECURE_NO_WARNINGS

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/commonChartElements
    } else {
        TARGET = release/commonChartElements
    }
}
win32:{
    TARGET = commonChartElements
}

INCLUDEPATH += ../mathFunctions

SOURCES += callout.cpp \
    dialogChangeAxis.cpp

HEADERS += callout.h \
    dialogChangeAxis.h

FORMS += \
 
