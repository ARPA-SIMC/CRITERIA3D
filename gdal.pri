#-----------------------------------------------------------------------------------------
#   GDAL library
#
#   How to compile on windows:
#
#   win32-msvc (Microsoft Visual C)
#   - install GDAL (32 or 64 bit) from OSGeo4W https://trac.osgeo.org/osgeo4w/
#   - add GDAL_PATH to the system variables - example: GDAL_PATH = C:\OSGeo4W64
#   - add GDAL_DATA to the system variables - example: GDAL_DATA = C:\OSGeo4W64\share\epsg_csv
#   - add PROJ_LIB to the system variables  - example: PROJ_LIB  = C:\OSGeo4W64\share\proj
#   - add OSGeo4W\bin to the system path
#
#   win32-g++ (MinGW)
#   Unfortunately it doesn't seem to work at the moment
#   - install and update MSYS2 from https://www.msys2.org/
#   - run MSYS2 shell and install GDAL package  - example: pacman -S mingw-w64-x86_64-gdal
#   - add MSYS_PATH to system variables         - example: MSYS_PATH = C:\msys64\mingw64
#   - add msys\mingw\bin to the system path
#
#------------------------------------------------------------------------------------------

unix:!macx {
    LIBS += -L/usr/lib -lgdal
    LIBS += -L/usr/lib/x86_64-linux-gnu -lgeos_c

    INCLUDEPATH += /usr/include/gdal
    DEPENDPATH += /usr/include/gdal
}

win32-msvc {
    LIBS += -L$$(GDAL_PATH)/lib/ -lgdal_i -lgeos_c

    INCLUDEPATH += $$(GDAL_PATH)/include
    DEPENDPATH += $$(GDAL_PATH)/include
}

win32-g++ {
    LIBS += -L$$(MSYS_PATH)/lib/ -lgdal -lgeos -lgeos_c

    INCLUDEPATH += $$(MSYS_PATH)/include
    DEPENDPATH += $$(MSYS_PATH)/include

    PRE_TARGETDEPS += $$(MSYS_PATH)/lib/libgdal.a
    PRE_TARGETDEPS += $$(MSYS_PATH)/lib/libgeos.dll.a
    PRE_TARGETDEPS += $$(MSYS_PATH)/lib/libgeos_c.dll.a
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


