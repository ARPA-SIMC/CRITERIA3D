   ## GDAL library on windows

   #### win32-msvc (Microsoft Visual C)
   - install GDAL (32 or 64 bit) from OSGeo4W https://trac.osgeo.org/osgeo4w/
   - add GDAL_PATH to the system variables - example: GDAL_PATH = C:\OSGeo4W64
   - add GDAL_DATA to the system variables - example: GDAL_DATA = C:\OSGeo4W64\share\epsg_csv
   - add PROJ_LIB to the system variables  - example: PROJ_LIB  = C:\OSGeo4W64\share\proj
   - add OSGeo4W\bin to the system path

   #### win32-g++ (MinGW)
   Unfortunately it doesn't seem to work at the moment
   - install and update MSYS2 from https://www.msys2.org/
   - run MSYS2 shell and install GDAL package - example: pacman -S mingw-w64-x86_64-gdal
   - add MSYS_PATH to system variables     - example: MSYS_PATH = C:\msys64\mingw64
   - add msys\mingw\bin to the system path - example: add C:\msys64\mingw64\bin
