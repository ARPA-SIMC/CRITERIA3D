#------------------------------------------------------
#
#   common dialogs library
#   contains dialogs for generic purpose
#
#   This project is part of agrolib distribution
#
#------------------------------------------------------

QT  += widgets

TEMPLATE = lib
CONFIG += staticlib
CONFIG += debug_and_release
CONFIG += c++14 c++17

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
    formSelection.cpp \
    formText.cpp \
    formTimePeriod.cpp \
    formSelectionSource.cpp

HEADERS += \
    formInfo.h \
    formSelection.h \
    formText.h \
    formTimePeriod.h \
    formSelectionSource.h

FORMS += \
    formTimePeriod.ui
