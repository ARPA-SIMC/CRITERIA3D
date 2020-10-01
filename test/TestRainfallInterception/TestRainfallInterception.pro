QT -= gui
QT += xml

TEMPLATE = app

CONFIG += c++11 console
CONFIG -= app_bundle

CONFIG += console
CONFIG -= app_bundle


# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS
DEFINES += _CRT_SECURE_NO_WARNINGS

unix:{
    CONFIG(debug, debug|release) {
        TARGET = debug/TestRainfallInterception
    } else {
        TARGET = release/TestRainfallInterception
    }
}
win32:{
    TARGET = TestRainfallInterception
}


# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG(release, debug|release) {
    LIBS += -L../../agrolib/crop/release -lcrop
    LIBS += -L../../agrolib/mathFunctions/release -lmathFunctions

} else {
    LIBS += -L../../agrolib/crop/debug -lcrop
    LIBS += -L../../agrolib/mathFunctions/release -lmathFunctions
}


INCLUDEPATH += ../../agrolib/crop
INCLUDEPATH += ../../agrolib/mathFunctions



SOURCES += \
        main.cpp \
        readWeatherMonticolo.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

HEADERS += \
    readWeatherMonticolo.h
