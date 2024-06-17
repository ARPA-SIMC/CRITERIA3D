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
        bool _saveOutputRaster, _saveOutputPoints, _saveDailyState;

        void clear3DProject();
        bool setSoilIndexMap();
        bool check3DProject();
        bool updateDailyTemperatures();


    public:
        Crit3DGeometry* openGlGeometry;

        // same header of DEM
        Crit3DSnowMaps snowMaps;
        gis::Crit3DRasterGrid degreeDaysMap;
        gis::Crit3DRasterGrid laiMap;
        gis::Crit3DRasterGrid dailyTminMap;
        gis::Crit3DRasterGrid dailyTmaxMap;

        Crit3DSnow snowModel;

        bool modelPause, modelStop;

        QDateTime modelFirstTime, modelLastTime;
        QDateTime meteoPointsDbFirstTime;

        Crit3DProject();

        bool initializeCriteria3DModel();

        void initializeCrop();
        void initializeCropWithClimateData();
        bool initializeCropFromDegreeDays(gis::Crit3DRasterGrid &myDegreeMap);
        void dailyUpdateCrop();

        void assignETreal();
        void assignPrecipitation();

        bool runModels(QDateTime firstTime, QDateTime lastTime);

        void setSaveDailyState(bool isSave);
        bool isSaveDailyState();

        void setSaveOutputRaster(bool isSave);
        bool isSaveOutputRaster();

        void setSaveOutputPoints(bool isSave);
        bool isSaveOutputPoints();

        bool loadCriteria3DProject(QString myFileName);
        bool loadCriteria3DParameters();
        bool writeCriteria3DParameters();

        bool loadSoilMap(QString fileName);
        bool loadLandUseMap(QString fileName);

        double getSoilVar(int soilIndex, int layerIndex, soil::soilVariable myVar);
        double* getSoilVarProfile(int row, int col, soil::soilVariable myVar);

        int getCrit3DSoilId(double x, double y);
        int getCrit3DSoilIndex(double x, double y);
        QString getCrit3DSoilCode(double x, double y);

        bool computeAllMeteoMaps(const QDateTime& myTime, bool showInfo);

        bool initializeSnowModel();
        bool computeSnowModel();
        void computeSnowPoint(int row, int col);

        bool modelHourlyCycle(QDateTime myTime, const QString& hourlyOutputPath);

        void setAllHourlyMeteoMapsComputed(bool value);

        bool saveDailyOutput(QDate myDate, const QString& outputPathHourly);
        bool saveSnowModelState();
        bool saveWaterModelState();

        bool loadModelState(QString statePath);
        QList<QString> getAllSavedState();

        bool writeOutputPointsTables();
        bool writeOutputPointsData();
        bool writeMeteoPointsProperties(const QList<QString> &joinedPropertiesList,
                                        const QList<QString> &csvFields, const QList<QList<QString>> &csvData);

        void clearGeometry();
        bool initializeGeometry();
        void shadowColor(const Crit3DColor &colorIn, Crit3DColor &colorOut, int row, int col);
        bool update3DColors();

    };


#endif // CRITERIA3DPROJECT_H
