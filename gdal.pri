# GDAL LIBRARY

unix:!macx {
    LIBS += -lgdal

    INCLUDEPATH += /usr/include/gdal
    DEPENDPATH += /usr/include/gdal
}

win32-msvc {
    LIBS += $$(GDAL_PATH)/lib/gdal_i.lib

    INCLUDEPATH += $$(GDAL_PATH)/include
    DEPENDPATH += $$(GDAL_PATH)/include

    PRE_TARGETDEPS += $$(GDAL_PATH)/lib/gdal_i.lib
}

win32-g++ {
# TODO
}

mac {
    LIBS += -framework IOKit
    LIBS += -framework CoreFoundation

    QMAKE_LFLAGS += -F/Library/Frameworks/

    LIBS += -framework GDAL
    LIBS += -framework PROJ
    LIBS += -framework GEOS

    INCLUDEPATH += /Library/Frameworks/GDAL.framework/Headers
    INCLUDEPATH += /Library/Frameworks/PROJ.framework/Headers
    INCLUDEPATH += /Library/Frameworks/GEOS.framework/Headers
}

