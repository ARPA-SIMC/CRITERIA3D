#ifndef SNOW_H
#define SNOW_H

    /*!
     * Junsei Kondo, Hiromi Yamazawa, Measurement of snow surface emissivity
    */
    #define SNOW_EMISSIVITY 0.97                /*!<  [-] */
    #define SOIL_EMISSIVITY 0.92                /*!<  [-] soil (average) */

    /*!
     * specific gas constant of water vapor
    */
    #define THERMO_WATER_VAPOR 0.4615           /*!<  [kJ kg-1 °K-1] */

    /*!
     * heat of fusion for ice at 0 °C
    */
    #define LATENT_HEAT_FUSION_KJ  335.            /*!<  [kJ kg-1] */
    #define LATENT_HEAT_VAPORIZATION_KJ 2500.      /*!<  [kJ kg-1] (at 0°C) */

    #define SNOW_SPECIFIC_HEAT 2.1              /*!<  [KJ kg-1 °C-1] */
    #define SOIL_SPECIFIC_HEAT 1.4              /*!<  [KJ kg-1 °C-1] wet soil */
    #define DEFAULT_BULK_DENSITY 1350           /*!<  [kg m-3] */
    #define SOIL_DAMPING_DEPTH 0.3              /*!<  [m] */
    #define SNOW_MINIMUM_HEIGHT 1.              /*!<  [mm] */

    #include "commonConstants.h"

    class Crit3DSnowParameters
    {
    public:
        double skinThickness;                   /*!<  [m] */
        double soilAlbedo;                      /*!<  [-] bare soil */
        double snowVegetationHeight;            /*!<  [m] height of vegetation */
        double snowWaterHoldingCapacity;        /*!<  [-] percentuale di acqua libera che il manto nevoso può trattenere */
        double tempMaxWithSnow;                 /*!<  [°C] */
        double tempMinWithRain;                 /*!<  [°C] */
        double snowSurfaceDampingDepth;         /*!<  [m] */

        Crit3DSnowParameters();

        void initializeSnowParameters();
    };


    class Crit3DSnow
    {
    public:
        Crit3DSnowParameters snowParameters;

        Crit3DSnow();

        void initializeSnow();

        void setSnowInputData(double temp, double prec, double relHum, double windInt, double globalRad,
                          double beamRad, double transmissivity, double clearSkyTransmissivity, double waterContent);

        bool isSnowPointValid();
        void computeSnowFall();
        void computeSnowBrooksModel();

        double getSnowFall() { return _precSnow; }
        double getSnowMelt() { return MAXVALUE(_snowMelt, 0); }
        double getDeltaSWE() { return _deltaSWE; }
        double getSensibleHeat() { return _sensibleHeat; }
        double getLatentHeat() { return _latentHeat; }

        double getSnowWaterEquivalent() { return _snowWaterEquivalent; }
        void setSnowWaterEquivalent(double value) { _snowWaterEquivalent = value; }

        double getIceContent() { return _iceContent; }
        void setIceContent(double value) { _iceContent = value; }

        double getLiquidWaterContent() { return _liquidWaterContent; }
        void setLiquidWaterContent(double value) { _liquidWaterContent = value; }

        double getInternalEnergy() { return _internalEnergy; }
        void setInternalEnergy(double value) { _internalEnergy = value; }

        double getSurfaceEnergy() { return _surfaceEnergy; }
        void setSurfaceEnergy(double value) { _surfaceEnergy = value; }

        double getSnowSurfaceTemp() { return _surfaceTemp; }
        void setSnowSurfaceTemp(float value) { _surfaceTemp = double(value); }

        double getAgeOfSnow() { return _ageOfSnow; }
        void setAgeOfSnow(float value) { _ageOfSnow = double(value); }

    private:
        // input
        double _clearSkyTransmissivity;     /*!<   [-] */
        double _transmissivity;             /*!<   [-] */
        double _globalRadiation;            /*!<   [W m-2] */
        double _beamRadiation;              /*!<   [W m-2] */
        double _prec;                       /*!<   [mm] */
        double _airT;                       /*!<   [°C] */
        double _airRH;                      /*!<   [%] */
        double _windInt;                    /*!<   [m/s] */
        double _surfaceWaterContent;        /*!<   [mm] */

        // output
        double _evaporation;                /*!<   [mm] */
        double _precRain;                   /*!<   [mm] */
        double _precSnow;                   /*!<   [mm] */
        double _snowMelt;                   /*!<   [mm] */
        double _deltaSWE;                   /*!<   [mm] */
        double _sensibleHeat;               /*!<   [kJ m-2] */
        double _latentHeat;                 /*!<   [kJ m-2] */

        // state variables
        double _snowWaterEquivalent;        /*!<   [mm] */
        double _iceContent;                 /*!<   [mm] */
        double _liquidWaterContent;         /*!<   [mm] */
        double _internalEnergy;             /*!<   [kJ m-2] */
        double _surfaceEnergy;              /*!<   [kJ m-2] */
        double _surfaceTemp;                /*!<   [°C] */
        double _ageOfSnow;                  /*!<   [days] */
    };


    double aerodynamicResistanceCampbell77(bool isSnow , double zRefWind, double windSpeed, double vegetativeHeight);

    double computeInternalEnergy(double soilTemperature,int bulkDensity, double swe);
    double computeInternalEnergySoil(double soilTemperature, int bulkDensity);

    double computeSurfaceEnergySnow(double surfaceTemperature, double skinThickness);
    double computeSurfaceEnergySoil(double surfaceTemperature, double skinThickness);


#endif // SNOW_H

