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

    #include <QString>


    class Crit3DProject : public Project3D
    {

    private:
        bool _saveOutputRaster, _saveOutputPoints, _saveDailyState, _saveEndOfRunState;

        void clear3DProject();
        bool check3DProject();
        bool updateDailyTemperatures();

        bool saveSnowModelState(const QString &currentStatePath);
        bool saveSoilWaterState(const QString &currentStatePath);

    public:
        Crit3DGeometry* openGlGeometry;

        // same header of DEM
        Crit3DSnowMaps snowMaps;
        gis::Crit3DRasterGrid degreeDaysMap;
        gis::Crit3DRasterGrid laiMap;
        gis::Crit3DRasterGrid dailyTminMap;
        gis::Crit3DRasterGrid dailyTmaxMap;

        Crit3DSnow snowModel;

        QDateTime modelFirstTime, modelLastTime;
        QDateTime meteoPointsDbFirstTime;

        Crit3DProject();

        bool initializeCriteria3DModel();

        bool initializeCrop();
        bool initializeCropWithClimateData();
        bool initializeCropFromDegreeDays(gis::Crit3DRasterGrid &myDegreeMap);
        void dailyUpdateCrop();

        void assignETreal();
        void assignPrecipitation();

        bool runModels(QDateTime firstTime, QDateTime lastTime, bool isRestart = false);

        void setSaveDailyState(bool isSave) { _saveDailyState = isSave; }
        bool isSaveDailyState() { return _saveDailyState; }

        void setSaveEndOfRunState(bool isSave) { _saveEndOfRunState = isSave; }
        bool isSaveEndOfRunState() { return _saveEndOfRunState; }

        void setSaveOutputRaster(bool isSave);
        bool isSaveOutputRaster();

        void setSaveOutputPoints(bool isSave);
        bool isSaveOutputPoints();

        bool loadCriteria3DProject(QString myFileName);
        bool loadCriteria3DParameters();
        bool writeCriteria3DParameters();

        bool loadLandUseMap(QString fileName);

        double getSoilVar(int soilIndex, int layerIndex, soil::soilVariable myVar);
        double* getSoilVarProfile(int row, int col, soil::soilVariable myVar);

        bool computeAllMeteoMaps(const QDateTime& myTime, bool showInfo);

        bool initializeSnowModel();
        bool computeSnowModel();
        void computeSnowPoint(int row, int col);

        bool runModelHour(const QDateTime &myDateTime, const QString& hourlyOutputPath, bool isRestart = false);

        void setAllHourlyMeteoMapsComputed(bool value);

        bool saveDailyOutput(QDate myDate, const QString& outputPathHourly);

        bool saveModelsState();

        bool loadModelState(QString statePath);
        bool loadWaterPotentialState(QString waterPath);

        QList<QString> getAllSavedState();

        bool writeOutputPointsTables();
        bool writeOutputPointsData();
        bool writeMeteoPointsProperties(const QList<QString> &joinedPropertiesList,
                                        const QList<QString> &csvFields, const QList<QList<QString>> &csvData);

        void clearGeometry();
        bool initializeGeometry();
        void shadowColor(const Crit3DColor &colorIn, Crit3DColor &colorOut, int row, int col);
        bool update3DColors(gis::Crit3DRasterGrid *rasterPointer = nullptr);

    };


#endif // CRITERIA3DPROJECT_H
