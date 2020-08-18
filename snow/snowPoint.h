#ifndef SNOWPOINT_H
#define SNOWPOINT_H

    #ifndef RADIATIONDEFINITIONS_H
        #include "radiationDefinitions.h"
    #endif

    /*!
     * \brief
     * Junsei Kondo, Hiromi Yamazawa, Measurement of snow surface emissivity
    */
    #define SNOW_EMISSIVITY 0.97                /*!<  [-] */
    #define SOIL_EMISSIVITY 0.92                /*!<  [-] soil (average) */

    /*!
     * specific gas constant of water vapor
    */
    #define THERMO_WATER_VAPOR 0.4615           /*!<  [kJ/(kg °K)] */

    /*!
     * heat of fusion for ice at 0 °C
    */
    #define LATENT_HEAT_FUSION  335             /*!<  [kJ/kg] */
    #define LATENT_HEAT_VAPORIZATION 2500       /*!<  [kJ/kg] */

    #define SOIL_SPECIFIC_HEAT 2.1              /*!<  [KJ/kg/°C] */
    #define DEFAULT_BULK_DENSITY 1300           /*!<  [kg/m^3] */
    #define SOIL_DAMPING_DEPTH 0.3              /*!<  [m] */
    #define SNOW_DAMPING_DEPTH 0.05             /*!<  [m] */
    #define SNOW_MINIMUM_HEIGHT 2               /*!<  [mm] */


    struct snowParameters {
        double snowSkinThickness;              /*!<  [m] */
        double soilAlbedo;                     /*!<  [-] bare soil */
        double snowVegetationHeight;           /*!<  [m] height of vegetation */
        double snowWaterHoldingCapacity;       /*!<  [-] percentuale di acqua libera che il manto nevoso può trattenere */
        double snowMaxWaterContent;            /*!<  [m] acqua libera (torrenti, laghetti) */
        double tempMaxWithSnow;                /*!<  [°C] */
        double tempMinWithRain;                /*!<  [°C] */
    };

    class Crit3DSnowPoint
    {
    public:
        Crit3DSnowPoint(struct TradPoint* radpoint, double temp, double prec, double relHum, double windInt, double clearSkyTransmissivity);
        ~Crit3DSnowPoint();

        bool checkValidPoint();
        void computeSnowFall();
        void computeSnowBrooksModel();

        double getSnowFall();
        double getSnowMelt();
        double getSnowWaterEquivalent();
        double getIceContent();
        double getLWContent();
        double getInternalEnergy();
        double getSurfaceInternalEnergy();
        double getSnowSurfaceTemp();
        double getAgeOfSnow();

        double getSnowSkinThickness();
        double getSoilAlbedo();
        double getSnowVegetationHeight();
        double getSnowWaterHoldingCapacity();
        double getSnowMaxWaterContent();
        double getTempMaxWithSnow();
        double getTempMinWithRain();

        static double aerodynamicResistanceCampbell77(bool isSnow , double zRefWind, double myWindSpeed, double vegetativeHeight);

    private:
        /*! input */
        TradPoint* _radpoint;
        double _clearSkyTransmissivity;      /*!<   [-] */
        double _prec;                        /*!<   [mm] */
        double _airT;                        /*!<   [°C] */
        double _airRH;                       /*!<   [%] */
        double _windInt;                     /*!<   [m/s] */
        double _waterContent;
        double _evaporation;
        struct snowParameters* _parameters;

        /*! output */
        double _snowFall;
        double _snowMelt;
        double _snowWaterEquivalent;
        double _iceContent;
        double _lWContent;
        double _internalEnergy;
        double _surfaceInternalEnergy;
        double _snowSurfaceTemp;
        double _ageOfSnow;
    };

#endif // SNOWPOINT_H
