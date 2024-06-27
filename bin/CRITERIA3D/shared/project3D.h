#ifndef PROJECT3D_H
#define PROJECT3D_H

    #ifndef PROJECT_H
        #include "project.h"
    #endif
    #ifndef SOIL_H
        #include "soil.h"
    #endif
    #ifndef LANDUNIT_H
        #include "landUnit.h"
    #endif
    #ifndef CROP_H
        #include "crop.h"
    #endif

    class QString;
    #include <QList>

    class WaterFluxesParameters
    {
    public:

        double initialWaterPotential;            // [m]
        double imposedComputationDepth;          // [m]
        double horizVertRatioConductivity;       // [-]

        bool freeCatchmentRunoff;
        bool freeLateralDrainage;
        bool freeBottomDrainage;
        bool computeOnlySurface;
        bool computeAllSoilDepth;

        WaterFluxesParameters();
        void initialize();
    };


    class Crit3DProcesses
    {
    public:

        bool computeMeteo, computeRadiation, computeWater, computeSlopeStability;
        bool computeCrop, computeSnow, computeSolutes;
        bool computeHeat, computeAdvectiveHeat, computeLatentHeat;

        Crit3DProcesses();
        void initialize();

        void setComputeCrop(bool value);
        void setComputeSnow(bool value);
        void setComputeWater(bool value);

    };


    class Project3D : public Project
    {
        Q_OBJECT

    signals:
        void updateOutputSignal();

    private:
        void setSoilLayers();
        void setLayersDepth();
        void setIndexMaps();
        bool setLateralBoundary();
        bool setCrit3DSurfaces();
        bool setCrit3DSoils();
        bool setCrit3DTopography();
        bool setCrit3DNodeSoil();
        bool initializeMatricPotential(float psi);

    public:
        bool isCriteria3DInitialized;
        bool isCropInitialized;
        bool showEachTimeStep;

        Crit3DProcesses processes;
        WaterFluxesParameters waterFluxesParameters;

        QString soilDbFileName;
        QString cropDbFileName;
        QString soilMapFileName;
        QString landUseMapFileName;

        unsigned long nrNodes;
        unsigned int nrLayers;
        int nrLateralLink;
        double currentSeconds;

        // soil and land use
        gis::Crit3DRasterGrid soilMap;
        gis::Crit3DRasterGrid landUseMap;
        std::vector <Crit3DLandUnit> landUnitList;
        // same index of landUnitsList
        std::vector <Crit3DCrop> cropList;

        // 3D soil fluxes maps
        gis::Crit3DRasterGrid soilIndexMap;
        QList<int> soilIndexList;
        gis::Crit3DRasterGrid boundaryMap;
        gis::Crit3DRasterGrid criteria3DMap;
        std::vector <gis::Crit3DRasterGrid> indexMap;

        // soil properties
        unsigned int nrSoils;
        double computationSoilDepth;            // [m]

        std::vector <soil::Crit3DTextureClass> texturalClassList;
        std::vector<soil::Crit3DGeotechnicsClass> geotechnicsClassList;
        std::vector <soil::Crit3DSoil> soilList;

        soil::Crit3DFittingOptions fittingOptions;

        // layers
        double minThickness;                    // [m]
        double maxThickness;                    // [m]
        double thickFactor;                     // [m]

        std::vector <double> layerDepth;        // [m]
        std::vector <double> layerThickness;    // [m]

        // sink/source
        std::vector <double> waterSinkSource;   // [m3 s-1]
        double totalPrecipitation;              // [m3 h-1]
        double totalEvaporation;                // [m3 h-1]
        double totalTranspiration;              // [m3 h-1]

        Project3D();

        void initializeProject3D();
        bool loadProject3DSettings();

        void clearProject3D();
        void clearWaterBalance3D();

        bool setSoilIndexMap();
        bool initializeWaterBalance3D();

        bool loadSoilDatabase(QString dbName);
        bool loadCropDatabase(QString dbName);
        bool loadSoilMap(QString fileName);

        double getSoilLayerTop(unsigned int i);
        double getSoilLayerBottom(unsigned int i);
        int getSoilLayerIndex(double depth);
        int getLandUnitIdUTM(double x, double y);
        int getLandUnitIdGeo(double lat, double lon);
        int getLandUnitIndexRowCol(int row, int col);

        bool initializeSoilMoisture(int month);

        int getSoilId(double x, double y);
        int getSoilListIndex(double x, double y);
        QString getSoilCode(double x, double y);

        int getSoilIndex(long row, long col);
        bool isWithinSoil(int soilIndex, double depth);

        bool interpolateAndSaveHourlyMeteo(meteoVariable myVar, const QDateTime& myTime,
                                           const QString& outputPath, bool isSaveOutputRaster);

        bool saveHourlyMeteoOutput(meteoVariable myVar, const QString& myPath, QDateTime myTime);
        bool aggregateAndSaveDailyMap(meteoVariable myVar, aggregationMethod myAggregation, const Crit3DDate& myDate,
                                      const QString& dailyPath, const QString& hourlyPath);

        bool interpolateHourlyMeteoVar(meteoVariable myVar, const QDateTime& myTime);

        double assignEvaporation(int row, int col, double lai, int soilIndex);
        double assignTranspiration(int row, int col, double currentLai, double currentDegreeDays);

        bool setSinkSource();
        void computeWaterBalance3D(double totalTimeStep);
        bool updateCrop(QDateTime myTime);

        bool setCriteria3DMap(criteria3DVariable var, int layerIndex);

        float computeFactorOfSafety(int row, int col, int layerIndex, int nodeIndex);
    };

    bool isCrit3dError(int result, QString &error);
    double getCriteria3DVar(criteria3DVariable myVar, long nodeIndex);
    bool setCriteria3DVar(criteria3DVariable myVar, long nodeIndex, double myValue);

    QString getOutputNameDaily(QString varName, QString notes, QDate myDate);
    QString getOutputNameHourly(meteoVariable myVar, QDateTime myTime);
    QString getDailyPrefixFromVar(QDate myDate, criteria3DVariable myVar);

    float readDataHourly(meteoVariable myVar, QString hourlyPath, QDateTime myTime, QString myArea, int row, int col);
    bool readHourlyMap(meteoVariable myVar, QString hourlyPath, QDateTime myTime, QString myArea, gis::Crit3DRasterGrid* myGrid);


#endif // PROJECT3D_H
