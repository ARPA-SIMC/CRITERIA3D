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
        float snowSkinThickness;              /*!<  [m] */
        float soilAlbedo;                     /*!<  [-] bare soil */
        float snowVegetationHeight;           /*!<  [m] height of vegetation */
        float snowWaterHoldingCapacity;       /*!<  [-] percentuale di acqua libera che il manto nevoso può trattenere */
        float snowMaxWaterContent;            /*!<  [m] acqua libera (torrenti, laghetti) */
        float tempMaxWithSnow;                /*!<  [°C] */
        float tempMinWithRain;                /*!<  [°C] */
    };

    class Crit3DSnowPoint
    {
    public:
        Crit3DSnowPoint(struct TradPoint* radpoint, float temp, float prec, float relHum, float windInt, float clearSkyTransmissivity);
        ~Crit3DSnowPoint();

        bool checkValidPoint();
        void computeSnowFall();
        void computeSnowBrooksModel();

        float getSnowFall();
        float getSnowMelt();
        float getSnowWaterEquivalent();
        float getIceContent();
        float getLWContent();
        float getInternalEnergy();
        float getSurfaceInternalEnergy();
        float getSnowSurfaceTemp();
        float getAgeOfSnow();

        float getSnowSkinThickness();
        float getSoilAlbedo();
        float getSnowVegetationHeight();
        float getSnowWaterHoldingCapacity();
        float getSnowMaxWaterContent();
        float getTempMaxWithSnow();
        float getTempMinWithRain();

        static float aerodynamicResistanceCampbell77(bool isSnow , float zRefWind, float myWindSpeed, float vegetativeHeight);

    private:
        /*! input */
        TradPoint* _radpoint;
        float _clearSkyTransmissivity;      /*!<   [-] */
        float _prec;                        /*!<   [mm] */
        float _airT;                        /*!<   [°C] */
        float _airRH;                       /*!<   [%] */
        float _windInt;                     /*!<   [m/s] */
        float _waterContent;
        float _evaporation;
        struct snowParameters* _parameters;

        /*! output */
        float _snowFall;
        float _snowMelt;
        float _snowWaterEquivalent;
        float _iceContent;
        float _lWContent;
        float _internalEnergy;
        float _surfaceInternalEnergy;
        float _snowSurfaceTemp;
        float _ageOfSnow;
    };

#endif // SNOWPOINT_H
