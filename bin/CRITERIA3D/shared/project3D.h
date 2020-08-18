#ifndef PROJECT3D_H
#define PROJECT3D_H

    #ifndef PROJECT_H
        #include "project.h"
    #endif
    #ifndef METEOMAPS_H
        #include "meteoMaps.h"
    #endif
    #ifndef SOIL_H
        #include "soil.h"
    #endif
    class QString;


    enum criteria3DVariable {waterContent, waterTotalPotential, waterMatricPotential,
                        availableWaterContent, degreeOfSaturation, soilTemperature,
                        soilSurfaceMoisture, bottomDrainage, waterDeficit, waterInflow, waterOutflow};


    class Project3D : public Project
    {
    private:
        void computeNrLayers();
        void setLayersDepth();
        bool setIndexMaps();
        bool setLateralBoundary();
        bool setCrit3DSurfaces();
        bool setCrit3DSoils();
        bool setCrit3DTopography();
        bool setCrit3DNodeSoil();

        bool interpolateAndSaveHourlyMeteo(meteoVariable myVar, const QDateTime& myTime,
                                           const QString& outputPath, bool saveOutput);

    public:
        bool isCriteria3DInitialized;

        QString soilDbFileName;
        QString cropDbFileName;
        QString soilMapFileName;

        unsigned long nrNodes;
        unsigned int nrLayers;
        int nrLateralLink;

        // 3D soil fluxes maps
        gis::Crit3DRasterGrid soilIndexMap;
        gis::Crit3DRasterGrid boundaryMap;
        std::vector <gis::Crit3DRasterGrid> indexMap;

        // soil properties
        unsigned int nrSoils;
        double soilDepth;                       // [m]

        std::vector <soil::Crit3DSoil> soilList;
        soil::Crit3DTextureClass texturalClassList[13];
        soil::Crit3DFittingOptions fittingOptions;

        // layers
        double minThickness;                    // [m]
        double maxThickness;                    // [m]
        double thickFactor;                     // [m]

        std::vector <double> layerDepth;        // [m]
        std::vector <double> layerThickness;    // [m]

        // sink/source
        std::vector <double> waterSinkSource;   // [m^3/sec]
        double totalPrecipitation, totalEvaporation, totalTranspiration;

        Project3D();

        void initializeProject3D();

        void clearProject3D();
        void clearWaterBalance3D();

        bool initializeWaterBalance3D();

        bool loadSoilDatabase(QString dbName);
        bool loadCropDatabase(QString dbName);

        double getSoilLayerTop(unsigned int i);
        double getSoilLayerBottom(unsigned int i);
        int getSoilLayerIndex(double depth);

        bool initializeSoilMoisture(int month);

        int getSoilIndex(long row, long col);
        bool isWithinSoil(int soilIndex, double depth);

        bool saveHourlyMeteoOutput(meteoVariable myVar, const QString& myPath, QDateTime myTime, const QString& myArea);
        bool aggregateAndSaveDailyMap(meteoVariable myVar, aggregationMethod myAggregation, const Crit3DDate& myDate,
                                      const QString& dailyPath, const QString& hourlyPath, const QString& myArea);

        bool interpolateHourlyMeteoVar(meteoVariable myVar, const QDateTime& myTime, bool showInfo);
        double computeEvaporation(int row, int col, double lai);
        bool computeWaterSinkSource();
        void computeWaterBalance3D(double timeStep);
        bool computeCrop(QDateTime myTime);
        bool modelHourlyCycle(bool isInitialState, QDateTime myTime, const QString& outputPath, bool saveOutput);

    };

    bool isCrit3dError(int result, QString* error);
    double getCriteria3DVar(criteria3DVariable myVar, long nodeIndex);
    bool setCriteria3DVar(criteria3DVariable myVar, long nodeIndex, double myValue);

    QString getOutputNameDaily(QString varName, QString strArea, QString notes, QDate myDate);
    QString getOutputNameHourly(meteoVariable myVar, QDateTime myTime, QString myArea);
    QString getDailyPrefixFromVar(QDate myDate, QString myArea, criteria3DVariable myVar);

    float readDataHourly(meteoVariable myVar, QString hourlyPath, QDateTime myTime, QString myArea, int row, int col);
    bool readHourlyMap(meteoVariable myVar, QString hourlyPath, QDateTime myTime, QString myArea, gis::Crit3DRasterGrid* myGrid);

    double getMaxEvaporation(double ET0, double LAI);


#endif // PROJECT3D_H
