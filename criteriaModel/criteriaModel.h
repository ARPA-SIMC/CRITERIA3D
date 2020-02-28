#ifndef CRITERIAMODEL_H
#define CRITERIAMODEL_H

    #ifndef SOIL_H
        #include "soil.h"
    #endif
    #ifndef CROP_H
        #include "crop.h"
    #endif
    #ifndef METEOPOINT_H
        #include "meteoPoint.h"
    #endif
    #ifndef QSTRING_H
        #include <QString>
    #endif
    #ifndef QSQLDATABASE_H
        #include <QSqlDatabase>
    #endif
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
        double dailyCropAvailableWater;
        double dailyWaterDeficit;
        double dailyWaterTable;
        double dailyCapillaryRise;

        Crit1DOutput();
        void initialize();
    };


    class Crit1DUnit
    {
        public:
        QString idCase;
        QString idCrop;
        QString idSoil;
        QString idMeteo;
        QString idForecast;
        QString idCropClass;
        int idCropNumber;
        int idSoilNumber;

        Crit1DUnit();
    };


    class Crit1DCase
    {
    public:
        QString idCase;

        // SOIL
        soil::Crit3DSoil mySoil;
        std::vector<soil::Crit3DLayer> soilLayers;

        double layerThickness;                  /*!<  [m]  */
        double maxSimulationDepth;              /*!<  [m]  */
        bool isGeometricLayer;

        // CROP
        Crit3DCrop myCrop;
        bool optimizeIrrigation;

        // WHEATER
        Crit3DMeteoPoint meteoPoint;

        // OUTPUT
        Crit1DOutput output;

        Crit1DCase();

        void initializeSoil();

    };


    class Crit1DIrrigationForecast
    {
    public:
        // DATABASE
        QSqlDatabase dbCrop;
        QSqlDatabase dbSoil;
        QSqlDatabase dbMeteo;
        QSqlDatabase dbForecast;
        QSqlDatabase dbOutput;

        soil::Crit3DTextureClass soilTexture[13];
        soil::Crit3DFittingOptions fittingOptions;

        // seasonal forecast
        bool isSeasonalForecast;
        int firstSeasonMonth;
        double* seasonalForecasts;
        int nrSeasonalForecasts;

        // short term forecast
        bool isShortTermForecast;
        int daysOfForecast;

        Crit1DCase myCase;
        QString outputString;

        Crit1DIrrigationForecast();

        bool setSoil(QString soilCode, QString *myError);
        bool loadMeteo(QString idMeteo, QString idForecast, QString *myError);
        bool createOutputTable(QString* myError);
        void prepareOutput(Crit3DDate myDate, bool isFirst);
        bool saveOutput(QString* myError);
        void initializeSeasonalForecast(const Crit3DDate& firstDate, const Crit3DDate& lastDate);
    };


#endif // CRITERIAMODEL_H
