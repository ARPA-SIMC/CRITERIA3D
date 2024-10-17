#-----------------------------------------------------------------------------------------
#   GDAL library
#
#   How to compile on windows:
#
#   win32-msvc (Microsoft Visual C)
#   - install GDAL from OSGeo4W https://trac.osgeo.org/osgeo4w/
#   - add GDAL_PATH to the system variables - example: GDAL_PATH = C:\OSGeo4W
#   - add GDAL_DATA to the system variables - example: GDAL_DATA = C:\OSGeo4W\share\epsg_csv
#   - add PROJ_LIB to the system variables  - example: PROJ_LIB  = C:\OSGeo4W\share\proj
#   - add OSGeo4W\bin to the system path
#
#------------------------------------------------------------------------------------------

unix:!macx {
    LIBS += -L/usr/lib -lgdal

    INCLUDEPATH += /usr/include/gdal
    DEPENDPATH += /usr/include/gdal
}

win32-msvc {
    LIBS += -L$$(GDAL_PATH)/lib/ -lgdal_i

    INCLUDEPATH += $$(GDAL_PATH)/include
    DEPENDPATH += $$(GDAL_PATH)/include
}

win32-g++ {
    LIBS += -L$$(GDAL_PATH)/lib/ -lgdal

    INCLUDEPATH += $$(GDAL_PATH)/include
    DEPENDPATH += $$(GDAL_PATH)/include

    PRE_TARGETDEPS += $$(MSYS_PATH)/lib/libgdal.a
}

mac {
    LIBS += -framework IOKit
    LIBS += -framework CoreFoundation
    LIBS += -framework GDAL

    QMAKE_LFLAGS += -F/Library/Frameworks/

    INCLUDEPATH += /Library/Frameworks/GDAL.framework/Headers
}


