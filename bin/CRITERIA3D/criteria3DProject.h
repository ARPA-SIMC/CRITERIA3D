#ifndef CRITERIA3DPROJECT_H
#define CRITERIA3DPROJECT_H

    #ifndef SOIL_H
        #include "soil.h"
    #endif
    #ifndef GIS_H
        #include "gis.h"
    #endif
    #ifndef SNOWMAPS_H
        #include "snowMaps.h"
    #endif
    #ifndef PROJECT_H
        #include "project.h"
    #endif
    #ifndef PROJECT3D_H
        #include "project3D.h"
    #endif
    #ifndef GEOMETRY_H
        #include "geometry.h"
    #endif
    #ifndef HYDRALL_H
        #include "hydrall.h"
    #endif
    #ifndef ROTHCPLUSPLUS_H
        #include "rothCplusplus.h"
    #endif

    #include <QString>

    #define CRITERIA3D_VERSION "V1.0.7"


    class Crit3DProject : public Project3D
    {

    private:
        bool _saveOutputRaster, _saveOutputPoints, _saveDailyState, _saveEndOfRunState;

        void clear3DProject();
        bool check3DProject();
        bool updateDailyTemperatures();
        bool updateLast30DaysTavg();
        void updateHydrallLAI();

        bool saveSnowModelState(const QString &currentStatePath);
        bool saveSoilWaterState(const QString &currentStatePath);

        void appendCriteria3DOutputValue(criteria3DVariable myVar, int row, int col,
                                         const std::vector<int> &depthList, std::vector<float> &outputList);

    public:
        Crit3DGeometry* openGlGeometry;

        // same header of DEM
        Crit3DSnowMaps snowMaps;
        gis::Crit3DRasterGrid degreeDaysMap;
        gis::Crit3DRasterGrid dailyTminMap;
        gis::Crit3DRasterGrid dailyTmaxMap;
        gis::Crit3DRasterGrid monthlyPrec;
        gis::Crit3DRasterGrid monthlyET0;

        Crit3DHydrallMaps hydrallMaps;

        Crit3DSnow snowModel;
        Crit3DHydrall hydrallModel;
        Crit3DRothCplusplus rothCModel;

        QDateTime modelFirstTime, modelLastTime;
        QDateTime meteoPointsDbFirstTime;

        Crit3DProject();

        bool initializeCriteria3DModel();

        void clearCropMaps();
        bool initializeCropMaps();
        bool initializeHydrall();
        bool initializeRothC();
        void updateETAndPrecMonthlyMaps();
        void dailyUpdateCropMaps(const QDate &myDate);

        bool initializeCropWithClimateData();
        bool initializeCropFromDegreeDays(gis::Crit3DRasterGrid &myDegreeMap);

        void assignETreal();
        void assignPrecipitation();
        float checkSoilCracking(int row, int col, float precipitation);

        bool checkProcesses();
        bool runModels(QDateTime firstTime, QDateTime lastTime, bool isRestart = false);

        void setSaveDailyState(bool isSave) { _saveDailyState = isSave; }
        bool isSaveDailyState() { return _saveDailyState; }

        void setSaveEndOfRunState(bool isSave) { _saveEndOfRunState = isSave; }
        bool isSaveEndOfRunState() { return _saveEndOfRunState; }

        void setSaveOutputRaster(bool isSave);
        bool isSaveOutputRaster();

        void setSaveOutputPoints(bool isSave);
        bool isSaveOutputPoints();

        bool loadCriteria3DProject(const QString &fileName);
        bool loadCriteria3DParameters();
        bool writeCriteria3DParameters(bool isSnow, bool isWater, bool isSoilCrack);

        double getSoilVar(int soilIndex, int layerIndex, soil::soilVariable myVar);
        double* getSoilVarProfile(int row, int col, soil::soilVariable myVar);

        bool computeAllMeteoMaps(const QDateTime& myTime, bool showInfo);

        bool initializeSnowModel();

        bool computeHydrallModel(int row, int col);
        void dailyUpdateHydrallMaps();
        bool dailyUpdateHydrall(const QDate &myDate);
        void setHydrallVariables(int row, int col);

        bool computeRothCModel();
        bool updateRothC();

        bool computeSnowModel();
        void computeSnowPoint(int row, int col);

        bool runModelHour(const QString& hourlyOutputPath, bool isRestart = false);

        void setAllHourlyMeteoMapsComputed(bool value);

        bool saveDailyOutput(QDate myDate, const QString& outputPathHourly);

        bool saveModelsState();

        bool loadModelState(QString statePath);
        bool loadWaterPotentialState(QString waterPath);

        bool getAllSavedState(QList<QString> &stateList);

        bool writeOutputPointsTables();
        bool writeOutputPointsData();
        bool writeMeteoPointsProperties(const QList<QString> &joinedPropertiesList,
                                        const QList<QString> &csvFields, const QList<QList<QString>> &csvData);

        void clearGeometry();
        bool initializeGeometry();
        void shadowColor(const Crit3DColor &colorIn, Crit3DColor &colorOut, int row, int col);
        bool update3DColors(gis::Crit3DRasterGrid *rasterPointer = nullptr);

        // SHELL
        int criteria3DShell();
        int criteria3DBatch(const QString &scriptFileName);

        int executeScript(const QString &scriptFileName);
        int executeCommand(const QList<QString> &argumentList);
        int executeCriteria3DCommand(const QList<QString> &argumentList, bool &isCommandFound);

        int cmdList(const QList<QString> &argumentList);
        int cmdOpenCriteria3DProject(const QList<QString> &argumentList);
        int cmdLoadState(const QList<QString> &argumentList);
        int cmdRunModels(const QList<QString> &argumentList);

        int printCriteria3DVersion();
        int printCriteria3DCommandList();

    };


#endif // CRITERIA3DPROJECT_H
