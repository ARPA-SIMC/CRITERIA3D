TEMPLATE = subdirs

SUBDIRS =       ../../agrolib/soilFluxes3D  \
                ../../agrolib/crit3dDate ../../agrolib/mathFunctions  \
                ../../agrolib/gis ../../agrolib/meteo ../../agrolib/soil ../../agrolib/crop \
                ../../agrolib/interpolation ../../agrolib/solarRadiation  ../../agrolib/snow \
                ../../agrolib/utilities ../../agrolib/dbMeteoPoints ../../agrolib/dbMeteoGrid  \
                ../../agrolib/importDataXML ../../agrolib/project  \
                ../../agrolib/meteoWidget ../../agrolib/soilWidget \
                ../../agrolib/commonDialogs ../../agrolib/graphics \
                ../CRITERIA3D

CONFIG += ordered

