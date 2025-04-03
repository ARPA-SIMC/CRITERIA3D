#ifndef RADIATIONDEFINITIONS_H
#define RADIATIONDEFINITIONS_H

    #ifndef _STRING_
        #include <string>
    #endif

    #ifndef _MAP_
        #include <map>
    #endif

    #ifndef GIS_H
        #include "gis.h"
    #endif

    /*! Surface pressure at sea level (millibars) (used for refraction correction and optical air mass) */
    #define PRESSURE_SEALEVEL 1013
    /*! Ambient default dry-bulb temperature (degrees C) (used for refraction correction) */
    #define TEMPERATURE_DEFAULT 15
    /*! scale height of Rayleigh atmosphere near the Earth surface */
    #define RAYLEIGH_Z0 8434.5

    /*! Eppley shadow band width (cm) */
    #define SBWID 7.6f
    /*! Eppley shadow band radius (cm) */
    #define SBRAD 31.7f
    /*! Drummond factor for partly cloudy skies */
    #define SBSKY 0.04f

    #define CLEAR_SKY_TRANSMISSIVITY_DEFAULT     0.75f
    #define SHADOW_FACTOR 1

    enum TradiationAlgorithm{RADIATION_ALGORITHM_RSUN = 0};
    enum TradiationRealSkyAlgorithm{RADIATION_REALSKY_TOTALTRANSMISSIVITY, RADIATION_REALSKY_LINKE};
    enum TparameterMode {PARAM_MODE_FIXED = 0, PARAM_MODE_MAP = 1, PARAM_MODE_MONTHLY = 2} ;
    enum TlandUse {LAND_USE_MOUNTAIN = 0, LAND_USE_RURAL = 1, LAND_USE_CITY = 2, LAND_USE_INDUSTRIAL = 3};
    enum TtiltMode{TILT_TYPE_FIXED=1, TILT_TYPE_DEM=2};
    //enum TtransmissivityAlgorithm{TRANSMISSIVITY_MODEL_HOURLY = 0, TRANSMISSIVITY_MODEL_DAILY = 1, TRANSMISSIVITY_MODEL_SAMANI = 2};
    //enum TtransmissivityComputationPeriod{TRANSMISSIVITY_COMPUTATION_DYNAMIC = 0,TRANSMISSIVITY_COMPUTATION_DAILY = 1};

    const std::map<std::string, TradiationAlgorithm> radAlgorithmToString = {
      { "r.sun", RADIATION_ALGORITHM_RSUN }
    };

    const std::map<std::string, TradiationRealSkyAlgorithm> realSkyAlgorithmToString = {
      { "Linke turbidity factor", RADIATION_REALSKY_LINKE },
      { "Total transmissivity", RADIATION_REALSKY_TOTALTRANSMISSIVITY }
    };

    const std::map<std::string, TparameterMode> paramModeToString = {
      { "fixed", PARAM_MODE_FIXED },
      { "map", PARAM_MODE_MAP },
      { "monthly", PARAM_MODE_MONTHLY}
    };

    const std::map<std::string, TtiltMode> tiltModeToString = {
      { "fixed", TILT_TYPE_FIXED },
      { "dem", TILT_TYPE_DEM }
    };

    const std::map<std::string, TlandUse> landUseToString = {
        { "industrial", LAND_USE_INDUSTRIAL },
        { "urban", LAND_USE_CITY },
        { "country", LAND_USE_RURAL },
        { "mountain", LAND_USE_MOUNTAIN }
    };

    struct TsunPosition
    {
        float hourDecimal;
        float rise;                     /*!<  Sunrise time, from midnight, local, WITHOUT refraction [s] */
        float set;                      /*!<  Sunset time, from midnight, local, WITHOUT refraction [s] */
        float azimuth;                  /*!<  Solar azimuth angle [degrees, N=0, E=90, S=180, W = 270] */
        float elevation;                /*!<  Solar elevation, no atmospheric correction */
        float elevationRefr;            /*!<  Solar elevation angle refracted [deg.From horizon] */
        float incidence;                /*!<  Solar incidence angle on Panel [deg] */
        float relOptAirMass;            /*!<  Relative optical airmass [] */
        float relOptAirMassCorr;        /*!<  Pressure-corrected airmass [] */
        float extraIrradianceNormal;    /*!<  Extraterrestrial (top-of-atmosphere) direct normal solar irradiance [W m-2] */
        float extraIrradianceHorizontal;/*!<  Extraterrestrial (top-of-atmosphere) global horizontal solar irradiance [W m-2] */
        bool shadow;                    /*!<  Boolean for "Sun is not visible" */
    };

    struct TradPoint
    {
        double x;
        double y;
        double height;
        double lat;
        double lon;
        double slope;
        double aspect;
        double beam;
        double diffuse;
        double reflected;
        double global;
        double transmissivity;
    };

    struct TelabRadPoint
    {
        TradPoint radPoint;
        std::string fileName;
        Crit3DDate iniDate, endDate;
        int iniHour, endHour;
    };


#endif // RADIATIONDEFINITIONS_H
