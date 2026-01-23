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


    #define CRITERIA3D_VERSION "v1.3.0 (2026)"


    class Crit3DProject : public Project3D
    {

    private:
        bool _saveOutputRaster, _saveOutputPoints, _saveDailyState, _saveEndOfRunState,
             _saveYearlyState, _saveMonthlyState;

        void clear3DProject();
        bool check3DProject();
        bool updateDailyTemperatures();
        bool updateLast30DaysTavg();


        bool saveSnowModelState(const QString &currentStatePath);
        bool saveSoilWaterState(const QString &currentStatePath);
        bool saveRothCState(const QString &currentStatePath);
        bool saveHydrallState(const QString &currentStatePath);

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
        gis::Crit3DRasterGrid mapLast30DaysTAvg;

        Crit3DHydrallMaps hydrallMaps;

        Crit3DSnow snowModel;
        Crit3DHydrall hydrallModel;
        Crit3DRothCplusplus rothCModel;

        QDateTime modelFirstTime, modelLastTime;
        QDateTime meteoPointsDbFirstTime;

        Crit3DProject();

        void clearCropMaps();
        bool initializeCropMaps();
        bool initializeHydrall();
        bool initializeHydrallConversionVector();
        bool initializeRothC();
        bool initializeRothCSoilCarbonContent();
        bool loadRothCTempMaps();
        bool loadRothCBICMaps();
        double getRothCClayContent(int soilIndex);
        double getRothCYield(int row, int col);
        void updateETAndPrecMaps();
        void dailyUpdateCropMaps(const QDate &myDate);

        void clearHydrallMaps();
        void clearRothCMaps();

        bool initializeCropWithClimateData();
        bool initializeCropFromDegreeDays(gis::Crit3DRasterGrid &myDegreeMap);

        void assignETreal();
        void assignPrecipitation();
        float computeSoilCracking(int row, int col, float precipitation);

        void setSaveDailyState(bool isSave) { _saveDailyState = isSave; }
        bool isSaveDailyState() { return _saveDailyState; }

        void setSaveYearlyState(bool isSave) { _saveYearlyState = isSave; }
        bool isSaveYearlyState() { return _saveYearlyState; }

        void setSaveEndOfRunState(bool isSave) { _saveEndOfRunState = isSave; }
        bool isSaveEndOfRunState() { return _saveEndOfRunState; }

        void setSaveMonthlyState(bool isSave) {_saveMonthlyState = isSave;}
        bool isSaveMonthlyState() {return _saveMonthlyState;}

        void setSaveOutputRaster(bool isSave);
        bool isSaveOutputRaster();

        void setSaveOutputPoints(bool isSave);
        bool isSaveOutputPoints();

        bool loadCriteria3DProject(const QString &fileName);
        bool loadCriteria3DParameters();
        bool writeCriteria3DParameters(bool isSnow, bool isWater);

        double getSoilVar(int soilIndex, int layerIndex, soil::soilVariable myVar);
        double* getSoilVarProfile(int row, int col, soil::soilVariable myVar);

        bool computeAllMeteoMaps(const QDateTime& myTime, bool showInfo);

        bool initializeSnowModel();

        bool computeHydrallModel(Crit3DHydrall &myHydrallModel, int row, int col, int forestIndex);
        bool setHydrallVariables(Crit3DHydrall &myHydrallModel, int row, int col, int forestIndex);

        bool dailyUpdateHydrall(const QDate &myDate);

        bool computeRothCModel();
        bool updateRothC(const QDate &myDate);
        void setRothCVariables(int row, int col, int month);

        bool computeSnowModel();
        void computeSnowPoint(Crit3DSnow &snowPoint, int row, int col);

        bool checkProcesses();

        bool startModels(const QDateTime &firstTime, const QDateTime &lastTime);
        bool runModels(const QDateTime &firstTime, const QDateTime &lastTime, bool isRestart = false);
        bool runModelHour(const QString& hourlyOutputPath, bool isRestart = false);

        void setAllHourlyMeteoMapsComputed(bool value);

        bool saveDailyOutput(QDate myDate, const QString& outputPathHourly);

        bool saveModelsState(QString &dirName);

        bool loadModelState(QString statePath);
        bool loadWaterPotentialState(QString waterPath);

        bool getAllSavedState(QList<QString> &stateList);

        bool writeOutputPointsTables();
        bool writeOutputPointsData();
        bool writeMeteoPointsProperties(const QList<QString> &joinedPropertiesList,
                                        const QList<QString> &csvFields, const QList<QList<QString>> &csvData);

        void clearGeometry();
        bool initializeGeometry();
        void shadowDtmColor(const Crit3DColor &colorIn, Crit3DColor &colorOut, int row, int col);
        bool update3DColors(gis::Crit3DRasterGrid *rasterPointer = nullptr);
        void getMixedColor(gis::Crit3DRasterGrid *rasterPointer, int row, int col,
                           double variableRange, const Crit3DColor& dtmColor, Crit3DColor& otutColor);

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
        int cmdSaveCurrentState();

        int printCriteria3DVersion();
        int printCriteria3DCommandList();
        int cmdSetThreadsNr(const QList<QString> &argumentList);

    };


#endif // CRITERIA3DPROJECT_H
