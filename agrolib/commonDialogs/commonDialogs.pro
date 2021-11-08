#------------------------------------------------------
#
#   common dialogs library
#   contains dialogs for generic purpose
#
#   This project is part of agrolib distribution
#
#------------------------------------------------------

QT  += gui widgets

TEMPLATE = lib
CONFIG += staticlib
CONFIG += debug_and_release

DEFINES += _CRT_SECURE_NO_WARNINGS

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/commonDialogs
    } else {
        TARGET = release/commonDialogs
    }
}
win32:{
    TARGET = commonDialogs
}

INCLUDEPATH += ../mathFunctions

SOURCES += \
    formInfo.cpp \
    formTimePeriod.cpp

HEADERS += \
    formInfo.h \
    formTimePeriod.h

FORMS += \
    formTimePeriod.ui
