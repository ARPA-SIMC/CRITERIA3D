TEMPLATE = subdirs

SUBDIRS =       ../../agrolib/soilFluxes3D  \
                ../../agrolib/crit3dDate ../../agrolib/mathFunctions \
                ../../agrolib/gis ../../agrolib/meteo \
                ../../agrolib/utilities ../../agrolib/soil ../../agrolib/crop \
                ../../agrolib/grapevine ../../agrolib/outputPoints \
                ../../agrolib/interpolation ../../agrolib/solarRadiation  \
                ../../agrolib/dbMeteoPoints ../../agrolib/dbMeteoGrid  \
                ../../agrolib/commonDialogs ../../agrolib/importDataXML \
                ../../agrolib/proxyWidget ../../agrolib/project  \
                ../../agrolib/meteoWidget  ../../agrolib/graphics \
                ../VINE3D

CONFIG += ordered

