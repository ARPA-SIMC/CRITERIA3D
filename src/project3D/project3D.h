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

    #define ERROR_STR_INITIALIZE_3D "Initialize 3D model before."

    class WaterFluxesParameters
    {
    public:
        bool computeOnlySurface;
        bool computeAllSoilDepth;

        bool isInitialWaterPotential;
        double initialWaterPotential;               // [m]
        double initialDegreeOfSaturation;           // [-]

        double imposedComputationDepth;             // [m]
        double conductivityHorizVertRatio;          // [-]

        int modelAccuracy;                          // [-]
        int numberOfThreads;                        // [-]

        bool freeCatchmentRunoff;
        bool freeLateralDrainage;
        bool freeBottomDrainage;

        double minSoilLayerThickness;               // [m] minimum thickness of soil layers
        double maxSoilLayerThickness;               // [m] maximum thickness of soil layers
        double maxSoilLayerThicknessDepth;          // [m] depth at which the layers must have maximum thickness

        WaterFluxesParameters();

        void initialize();
    };


    class Crit3DProcesses
    {
    public:

        bool computeMeteo, computeRadiation, computeWater;
        bool computeCrop, computeSnow, computeSolutes, computeHydrall;
        bool computeHeat, computeAdvectiveHeat, computeLatentHeat;

        Crit3DProcesses();
        void initialize();

        void setComputeHydrall(bool value);
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
        bool setLayersDepth();
        void setIndexMaps();
        bool setLateralBoundary();
        bool setCrit3DSurfaces();
        bool setCrit3DSoils();
        bool setCrit3DTopography();
        bool setCrit3DNodeSoil();
        bool initializeWaterContent();

    public:
        bool isCriteria3DInitialized;
        bool isCropInitialized;
        bool isSnowInitialized;

        bool showEachTimeStep;
        bool increaseSlope;

        bool isModelRunning, isModelPaused, isModelStopped;

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

        // same header of DTM
        gis::Crit3DRasterGrid soilMap;
        gis::Crit3DRasterGrid landUseMap;
        gis::Crit3DRasterGrid laiMap;

        // same indexes
        std::vector <Crit3DLandUnit> landUnitList;
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
        std::vector <double> layerDepth;        // [m]
        std::vector <double> layerThickness;    // [m]
        double soilLayerThicknessGrowthFactor;  // [-] progressive growth factor of layer thicknesses

        double previousTotalWaterContent;       // [m3]

        // sink/source
        std::vector <double> waterSinkSource;   // [m3 s-1]
        double totalPrecipitation;              // [m3 h-1]
        double totalEvaporation;                // [m3 h-1]
        double totalTranspiration;              // [m3 h-1]

        // specific outputs
        std::vector<int> waterContentDepth;
        std::vector<int> degreeOfSaturationDepth;
        std::vector<int> waterPotentialDepth;
        std::vector<int> factorOfSafetyDepth;

        Project3D();

        void initializeProject3D();
        bool loadProject3DSettings();

        void clearProject3D();
        void clearWaterBalance3D();

        bool setSoilIndexMap();
        bool initialize3DModel();

        bool loadLandUseMap(const QString &fileName);
        bool loadSoilDatabase(const QString &dbName);
        bool loadCropDatabase(const QString &dbName);
        bool loadSoilMap(const QString &fileName);

        double getSoilLayerTop(unsigned int i);
        double getSoilLayerBottom(unsigned int i);
        int getSoilLayerIndex(double depth);
        int getLandUnitFromUtm(double x, double y);
        int getLandUnitIdGeo(double lat, double lon);
        int getLandUnitIndexRowCol(int row, int col);

        bool initializeSoilMoisture(int month);

        int getSoilMapId(double x, double y);
        int getSoilListIndex(double x, double y);
        QString getSoilCode(double x, double y);

        int getLandUnitListIndex(int id);
        bool isCrop(int unitIndex);

        float computeCurrentPond(int row, int col);
        bool dailyUpdatePond();

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
        bool setAccuracy();

        void runWaterFluxes3DModel(double totalTimeStep, bool isRestart = false);
        bool updateCrop(QDateTime myTime);

        bool computeCriteria3DMap(gis::Crit3DRasterGrid &outputRaster, criteria3DVariable var, int layerIndex);
        bool computeMinimumFoS(gis::Crit3DRasterGrid &outputRaster);

        float computeFactorOfSafety(int row, int col, unsigned int layerIndex);

        bool getTotalSurfaceWaterContent(double &wcSum, long &nrVoxels);
        bool getTotalSoilWaterContent(double &wcSum, long &nrVoxels);
    };

    bool isCrit3dError(int result, QString &error);
    double getCriteria3DVar(criteria3DVariable myVar, long nodeIndex);
    bool setCriteria3DVar(criteria3DVariable myVar, long nodeIndex, double myValue);

    QString getOutputNameDaily(QString varName, QString notes, QDate myDate);
    QString getOutputNameHourly(meteoVariable myVar, QDateTime myTime);
    QString getDailyPrefixFromVar(QDate myDate, criteria3DVariable myVar);

    float readDataHourly(meteoVariable myVar, QString hourlyPath, QDateTime myTime, QString myArea, int row, int col);
    bool readHourlyMap(meteoVariable myVar, QString hourlyPath, QDateTime myTime, QString myArea, gis::Crit3DRasterGrid* myGrid);

    bool setVariableDepth(const QList<QString> &depthList, std::vector<int> &variableDepth);

#endif // PROJECT3D_H
