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
     * \brief The CriteriaModelOutput class
     * \note all variables are in [mm]
     */
    class CriteriaModelOutput
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

        CriteriaModelOutput();
        void initializeDaily();
    };


    class CriteriaUnit
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

        CriteriaUnit();
    };


    class CriteriaModel
    {
    public:
        QString idCase;

        // DATABASE
        QSqlDatabase dbCrop;
        QSqlDatabase dbSoil;
        QSqlDatabase dbMeteo;
        QSqlDatabase dbForecast;
        QSqlDatabase dbOutput;

        // IRRIGATION seasonal forecast
        bool isSeasonalForecast;
        int firstSeasonMonth;
        double* seasonalForecasts;
        int nrSeasonalForecasts;

        // IRRIGATION short term forecast
        bool isShortTermForecast;
        int daysOfForecast;

        // SOIL
        soil::Crit3DSoil mySoil;
        soil::Crit3DTextureClass soilTexture[13];
        std::vector<soil::Crit3DLayer> layers;
        soil::Crit3DFittingOptions fittingOptions;
        unsigned int nrLayers;
        double layerThickness;                  /*!<  [m]  */
        double maxSimulationDepth;              /*!<  [m]  */
        bool isGeometricLayer;

        // CROP
        Crit3DCrop myCrop;
        bool optimizeIrrigation;

        // WHEATER
        Crit3DMeteoPoint meteoPoint;

        // FIELD
        double depthPloughedSoil;               /*!< [m] depth of ploughed soil (working layer) */
        double initialAW[2];                    /*!< [-] fraction of available water (between wilting point and field capacity) */

        // OUTPUT
        CriteriaModelOutput output;
        QString outputString;

        CriteriaModel();

        bool loadMeteo(QString idMeteo, QString idForecast, QString *myError);
        bool setSoil(QString soilCode, QString *myError);
        bool createOutputTable(QString* myError);
        void prepareOutput(Crit3DDate myDate, bool isFirst);
        bool saveOutput(QString* myError);
        void initializeSeasonalForecast(const Crit3DDate& firstDate, const Crit3DDate& lastDate);
    };


#endif // CRITERIAMODEL_H
