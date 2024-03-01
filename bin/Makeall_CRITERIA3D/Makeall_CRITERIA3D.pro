TEMPLATE = subdirs

SUBDIRS =       ../../agrolib/soilFluxes3D  \
                ../../agrolib/crit3dDate ../../agrolib/mathFunctions  \
                ../../agrolib/gis ../../agrolib/meteo \
                ../../agrolib/utilities ../../agrolib/soil ../../agrolib/crop \
                ../../agrolib/interpolation ../../agrolib/solarRadiation  ../../agrolib/snow \
                ../../agrolib/dbMeteoPoints ../../agrolib/outputPoints \
                ../../agrolib/dbMeteoGrid  ../../agrolib/inOutDataXML \
                ../../agrolib/proxyWidget ../../agrolib/project  \
                ../../agrolib/meteoWidget ../../agrolib/soilWidget \
                ../../agrolib/commonDialogs ../../agrolib/commonChartElements ../../agrolib/graphics \
                ../CRITERIA3D

CONFIG += ordered

