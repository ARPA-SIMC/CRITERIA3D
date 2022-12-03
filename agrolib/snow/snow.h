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
    #define LATENT_HEAT_FUSION  335.             /*!<  [kJ kg-1] */
    #define LATENT_HEAT_VAPORIZATION 2500.       /*!<  [kJ kg-1] */

    #define SNOW_SPECIFIC_HEAT 2.1              /*!<  [KJ kg-1 °C-1] */
    #define SOIL_SPECIFIC_HEAT 1.4              /*!<  [KJ kg-1 °C-1] wet soil */
    #define DEFAULT_BULK_DENSITY 1350.          /*!<  [kg m-3] */
    #define SOIL_DAMPING_DEPTH 0.3              /*!<  [m] */
    #define SNOW_DAMPING_DEPTH 0.05             /*!<  [m] */
    #define SNOW_MINIMUM_HEIGHT 2.              /*!<  [mm] */


    class Crit3DSnowParameters
    {
    public:
        double snowSkinThickness;              /*!<  [m] */
        double soilAlbedo;                     /*!<  [-] bare soil */
        double snowVegetationHeight;           /*!<  [m] height of vegetation */
        double snowWaterHoldingCapacity;       /*!<  [-] percentuale di acqua libera che il manto nevoso può trattenere */
        double snowMaxWaterContent;            /*!<  [m] acqua libera (torrenti, laghetti) */
        double tempMaxWithSnow;                /*!<  [°C] */
        double tempMinWithRain;                /*!<  [°C] */

        Crit3DSnowParameters();

        void initialize();
    };


    class Crit3DSnow
    {
    public:
        Crit3DSnowParameters snowParameters;

        Crit3DSnow();

        void initialize();

        void setInputData(double temp, double prec, double relHum, double windInt, double globalRad,
                          double beamRad, double transmissivity, double clearSkyTransmissivity, double waterContent);

        bool checkValidPoint();
        void computeSnowFall();
        void computeSnowBrooksModel();

        double getSnowFall();
        double getSnowMelt();
        double getSensibleHeat();
        double getLatentHeat();

        double getSnowWaterEquivalent();
        double getIceContent();
        double getLiquidWaterContent();
        double getInternalEnergy();
        double getSurfaceEnergy();
        double getSnowSurfaceTemp();
        double getAgeOfSnow();

        void setSnowWaterEquivalent(float value);
        void setIceContent(float value);
        void setLiquidWaterContent(float value);
        void setInternalEnergy(float value);
        void setSurfaceEnergy(float value);
        void setSnowSurfaceTemp(float value);
        void setAgeOfSnow(float value);

    private:
        // input
        double _clearSkyTransmissivity;      /*!<   [-] */
        double _transmissivity;              /*!<   [-] */
        double _globalRadiation;             /*!<   [W m-2] */
        double _beamRadiation;               /*!<   [W m-2] */
        double _prec;                        /*!<   [mm] */
        double _airT;                        /*!<   [°C] */
        double _airRH;                       /*!<   [%] */
        double _windInt;                     /*!<   [m/s] */
        double _surfaceWaterContent;         /*!<   [mm] */

        // output
        double _evaporation;                /*!<   [mm] */
        double _precRain;                   /*!<   [mm] */
        double _precSnow;                   /*!<   [mm] */
        double _snowMelt;                   /*!<   [mm] */
        double _sensibleHeat;               /*!<   [kJ m-2] */
        double _latentHeat;                 /*!<   [kJ m-2] */

        // state variables
        double _snowWaterEquivalent;        /*!<   [mm] */
        double _iceContent;                 /*!<   [mm] */
        double _liquidWaterContent;         /*!<   [mm] */
        double _internalEnergy;             /*!<   [kJ m-2] */
        double _surfaceEnergy;              /*!<   [kJ m-2] */
        double _snowSurfaceTemp;            /*!<   [°C] */
        double _ageOfSnow;                  /*!<   [days] */
    };


    double aerodynamicResistanceCampbell77(bool isSnow , double zRefWind, double windSpeed, double vegetativeHeight);
    double computeInternalEnergy(double initSoilPackTemp,int bulkDensity, double initSWE);
    double computeSurfaceEnergy(double initSnowSurfaceTemp, double snowSkinThickness);


#endif // SNOW_H

