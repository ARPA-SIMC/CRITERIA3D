#ifndef CRITERIA1DCASE_H
#define CRITERIA1DCASE_H

    #ifndef SOIL_H
        #include "soil.h"
    #endif
    #ifndef CARBON_H
        #include "carbonNitrogen.h"
    #endif
    #ifndef CROP_H
        #include "crop.h"
    #endif
    #ifndef METEOPOINT_H
        #include "meteoPoint.h"
    #endif
    #ifndef COMPUTATIONUNITSDB_H
        #include "computationUnitsDb.h"
    #endif

    #include <QString>
    #include <vector>

    /*!
    * \brief daily output of Criteria1D
    * \note all variables are in [mm] (except dailyWaterTable [m])
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
        double dailyWaterTable;             // [m]
        double dailyCapillaryRise;
        double dailyBalance;

        Crit1DOutput();
        void initialize();
    };


    class Crit1DCase
    {
    public:
        Crit1DCompUnit unit;
        bool computeFactorOfSafety;

        // SOIL
        soil::Crit3DSoil mySoil;
        std::vector<soil::Crit3DLayer> soilLayers;
        std::vector<Crit3DCarbonNitrogenLayer> carbonNitrogenLayers;
        soil::Crit3DFittingOptions fittingOptions;

        // CROP
        Crit3DCrop crop;

        // WHEATER
        Crit3DMeteoPoint meteoPoint;

        // OUTPUT
        Crit1DOutput output;

        Crit1DCase();

        bool initializeSoil(std::string &error);
        void initializeWaterContent(Crit3DDate myDate);
        bool computeDailyModel(Crit3DDate &myDate, std::string &error);

        double getWaterContent(double computationDepth);
        double getWaterPotential(double computationDepth);
        double getFractionAW(double computationDepth);
        double getSlopeStability(double computationDepth);

        double getWaterDeficitSum(double computationDepth);
        double getWaterCapacitySum(double computationDepth);
        double getAvailableWaterSum(double computationDepth);

    private:
        double minLayerThickness;       // [m]
        double geometricFactor;         // [-]

        double lx, ly;                  // [m]
        double area;                    // [m2]


        bool initializeNumericalFluxes(std::string &error);
        bool computeNumericalFluxes(const Crit3DDate &myDate, std::string &error);
        bool computeWaterFluxes(const Crit3DDate &myDate, std::string &error);
        double checkIrrigationDemand(int doy, double currentPrec, double nextPrec, double maxTranspiration);
        void saveWaterContent();
        void restoreWaterContent();
        double getTotalWaterContent();


     public:
        std::vector<double> prevWaterContent;
        double ploughedSoilDepth;       // [m]

    };



#endif // CRITERIA1DCASE_H
