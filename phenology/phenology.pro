#------------------------------------------------------
#
#   phenology (library)
#   this project is part of ARPAE agrolib distribution
#
#------------------------------------------------------

QT       -= core gui

TEMPLATE = lib
CONFIG += staticlib

DEFINES += PHENOLOGY_LIBRARY
DEFINES += _CRT_SECURE_NO_WARNINGS

INCLUDEPATH +=  ../mathFunctions ../crit3dDate

CONFIG += debug_and_release
CONFIG += c++14 c++17

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/phenology
    } else {
        TARGET = release/phenology
    }
}
win32:{
    TARGET = phenology
}

SOURCES += \
    bietola.cpp \
    cereali.cpp \
    console.cpp \
    fenologia.cpp \
    girasole.cpp \
    mais.cpp \
    olivo.cpp \
    pomodoro.cpp \
    soia.cpp \
    stazione.cpp \
    vite.cpp

HEADERS += \
    bietola.h \
    cereali.h \
    coltura.h \
    console.h \
    fenologia.h \
    girasole.h \
    mais.h \
    olivo.h \
    pomodoro.h \
    soia.h \
    stazione.h \
    vite.h
