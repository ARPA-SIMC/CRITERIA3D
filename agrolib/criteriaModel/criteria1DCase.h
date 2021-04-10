#ifndef CRITERIA1DCASE_H
#define CRITERIA1DCASE_H

    #ifndef SOIL_H
        #include "soil.h"
    #endif
    #ifndef CROP_H
        #include "crop.h"
    #endif
    #ifndef METEOPOINT_H
        #include "meteoPoint.h"
    #endif

    #include <QString>
    #include <vector>

    /*!
    * \brief daily output of Criteria1D
    * \note all variables are in [mm]
    */
    class Crit1DOutput
    {
    public:
        double dailyPrec;
        double dailySurfaceRunoff;
        double dailySoilWaterContent;
        double dailySurfaceWaterContent;
        double dailyLateralDrainage;
        double dailyDrainage;
        double dailyIrrigation;
        double dailyEt0;
        double dailyMaxEvaporation;
        double dailyEvaporation;
        double dailyMaxTranspiration;
        double dailyTranspiration;
        double dailyAvailableWater;
        double dailyFractionAW;
        double dailyReadilyAW;
        double dailyWaterTable;
        double dailyCapillaryRise;

        Crit1DOutput();
        void initialize();
    };


    class Crit1DCase
    {
    public:
        QString idCase;

        // SOIL
        soil::Crit3DSoil mySoil;
        std::vector<soil::Crit3DLayer> soilLayers;

        double minLayerThickness;       // [m]
        double geometricFactor;         // [-]

        bool isGeometricLayers;
        bool isNumericalInfiltration;

        // CROP
        Crit3DCrop myCrop;
        bool optimizeIrrigation;

        // WHEATER
        Crit3DMeteoPoint meteoPoint;

        // OUTPUT
        Crit1DOutput output;

        Crit1DCase();

        bool initializeSoil(std::string &myError);
        bool computeDailyModel(Crit3DDate myDate, std::string &myError);
        double getWaterContent(double depth);
        double getWaterPotential(double depth);
        double getSoilWaterDeficit(double depth);

    private:
        bool initializeNumericalFluxes(std::string &myError);

    };


    bool dailyModel(Crit3DDate myDate, Crit3DMeteoPoint &meteoPoint, Crit3DCrop &myCrop, std::vector<soil::Crit3DLayer> &soilLayers,
                    Crit1DOutput &myOutput, bool optimizeIrrigation, std::string &myError);


#endif // CRITERIA1DCASE_H
