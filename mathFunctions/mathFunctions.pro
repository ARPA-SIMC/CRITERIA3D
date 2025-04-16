#-------------------------------------------------------
#
#   mathFunctions library
#   contains common constants, math, physics and statistic functions
#
#   This project is part of ARPAE agrolib distribution
#
#--------------------------------------------------------

QT       -= core gui

TEMPLATE = lib
CONFIG += staticlib

CONFIG += debug_and_release
CONFIG += c++11 c++14 c++17

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/mathFunctions
    } else {
        TARGET = release/mathFunctions
    }
}
win32:{
    TARGET = mathFunctions
}


HEADERS += \
    commonConstants.h \
    basicMath.h \
    furtherMathFunctions.h \
    statistics.h \
    physics.h \
    gammaFunction.h

SOURCES += \
    basicMath.cpp \
    furtherMathFunctions.cpp \
    statistics.cpp \
    physics.cpp \
    gammaFunction.cpp

