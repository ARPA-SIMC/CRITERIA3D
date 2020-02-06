#ifndef PROJECT_H
#define PROJECT_H

    #ifndef QUALITY_H
        #include "quality.h"
    #endif
    #ifndef GIS_H
        #include "gis.h"
    #endif
    #ifndef DBMETEOPOINTS_H
        #include "dbMeteoPointsHandler.h"
    #endif
    #ifndef DBAGGREGATIONSHANDLER_H
        #include "dbAggregationsHandler.h"
    #endif
    #ifndef DBMETEOGRID_H
        #include "dbMeteoGrid.h"
    #endif
    #ifndef SOLARRADIATION_H
        #include "solarRadiation.h"
    #endif
    #ifndef INTERPOLATIONCMD_H
        #include "interpolationCmd.h"
    #endif
    #ifndef METEOMAPS_H
        #include "meteoMaps.h"
    #endif

    #ifndef QSETTINGS_H
        #include <QSettings>
    #endif
    #ifndef _FSTREAM_
        #include <fstream>
    #endif

    class Project {
    private:
        QString appPath;
        QString defaultPath;
        QString projectPath;

        void clearMeteoPoints();
        bool createDefaultProject(QString fileName);
        bool searchDefaultPath(QString* path);

    protected:
        frequencyType currentFrequency;
        meteoVariable currentVariable;
        QDate previousDate, currentDate;
        int currentHour;

    public:
        QString projectName = "";
        bool isProjectLoaded;
        int modality;
        QString currentTileMap;

        bool requestedExit;
        QString errorString;

        QString logFileName;
        QString demFileName;
        QString dbPointsFileName;
        QString dbGridXMLFileName;
        QString parametersFileName;
        std::ofstream logFile;

        QSettings* parameters;
        QSettings* projectSettings;

        bool meteoPointsLoaded;
        int nrMeteoPoints;
        Crit3DMeteoPoint* meteoPoints;
        Crit3DMeteoPointsDbHandler* meteoPointsDbHandler;
        Crit3DAggregationsDbHandler* aggregationDbHandler;
        QList<gis::Crit3DGeoPoint> meteoPointsSelected;
        Crit3DColorScale* meteoPointsColorScale;

        bool meteoGridLoaded;
        Crit3DMeteoGridDbHandler* meteoGridDbHandler;
        bool loadGridDataAtStart;

        Crit3DQuality* quality;
        bool checkSpatialQuality;

        Crit3DMeteoSettings* meteoSettings;

        gis::Crit3DGisSettings gisSettings;
        Crit3DRadiationSettings radSettings;

        Crit3DRadiationMaps *radiationMaps;
        Crit3DHourlyMeteoMaps *hourlyMeteoMaps;

        gis::Crit3DRasterGrid DEM;

        Crit3DInterpolationSettings interpolationSettings;
        Crit3DInterpolationSettings qualityInterpolationSettings;

        std::vector <Crit3DProxyGridSeries> proxyGridSeries;

        Crit3DClimateParameters climateParameters;

        Project();

        void initializeProject();
        void clearProject();

        void createProject(QString path_, QString name_, QString description);
        void saveProject();
        void saveProjectSettings();
        void saveAllParameters();
        void saveGenericParameters();
        void saveInterpolationParameters();
        void saveRadiationParameters();
        void saveProxies();

        bool start(QString appPath);
        bool loadProject();
        bool loadProjectSettings(QString settingsFileName);
        bool loadParameters(QString parametersFileName);
        bool loadRadiationGrids();

        void setProxyDEM();
        void clearProxyDEM();
        bool checkProxy(QString name_, QString gridName_, QString table_, QString field_, QString *error);
        void addProxyToProject(QString name_, QString gridName_, QString table_, QString field_, bool isForQuality_, bool isActive_);
        bool addProxyGridSeries(QString name_, std::vector <QString> gridNames, std::vector <unsigned> gridYears);
        void setCurrentDate(QDate myDate);
        void setCurrentHour(int myHour);
        void setCurrentVariable(meteoVariable variable);
        int getCurrentHour();
        QDate getCurrentDate();
        Crit3DTime getCrit3DCurrentTime();
        QDateTime getCurrentTime();
        meteoVariable getCurrentVariable();

        void setApplicationPath(QString myPath);
        QString getApplicationPath();
        void setDefaultPath(QString myPath);
        QString getDefaultPath();
        void setProjectPath(QString myPath);
        QString getProjectPath();
        QString getRelativePath(QString fileName);
        QString getCompleteFileName(QString fileName, QString secondaryPath);

        bool setLogFile(QString fileNameWithPath);
        void logError(QString myStr);
        void logInfo(QString myStr);
        void logInfoGUI(QString myStr);
        void logError();

        void closeMeteoPointsDB();
        void closeMeteoGridDB();

        bool loadDEM(QString myFileName);
        void closeDEM();
        bool loadMeteoPointsData(QDate firstDate, QDate lastDate, bool loadHourly, bool loadDaily, bool showInfo);
        bool loadMeteoPointsDB(QString dbName);
        bool loadMeteoGridDB(QString xmlName);
        bool loadAggregationdDB(QString dbName);
        bool loadMeteoGridDailyData(QDate firstDate, QDate lastDate, bool showInfo);
        bool loadMeteoGridHourlyData(QDateTime firstDate, QDateTime lastDate, bool showInfo);
        void loadMeteoGridData(QDate firstDate, QDate lastDate, bool showInfo);
        QDateTime findDbPointLastTime();
        QDateTime findDbPointFirstTime();

        void getMeteoPointsRange(float *minimum, float *maximum);
        bool getMeteoPointSelected(int i);

        float meteoDataConsistency(meteoVariable myVar, const Crit3DTime& timeIni, const Crit3DTime& timeFin);

        bool loadProxyGrids();
        bool readPointProxyValues(Crit3DMeteoPoint* myPoint, QSqlDatabase* myDb);
        bool readProxyValues();
        bool updateProxy();
        void checkMeteoPointsDEM();
        bool writeTopographicDistanceMaps(bool onlyWithData);
        bool loadTopographicDistanceMaps(bool showInfo);
        void passInterpolatedTemperatureToHumidityPoints(Crit3DTime myTime);
        bool interpolationDemMain(meteoVariable myVar, const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster, bool showInfo);
        bool interpolationDem(meteoVariable myVar, const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster, bool showInfo);
        bool interpolateDemRadiation(const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster, bool showInfo);
        frequencyType getCurrentFrequency() const;
        void setCurrentFrequency(const frequencyType &value);

        bool checkMeteoGridForExport();
        void importHourlyMeteoData(const QString& fileName, bool importAllFiles, bool deletePreviousData);

        gis::Crit3DRasterGrid* getHourlyMeteoRaster(meteoVariable myVar);

    };


#endif // PROJECT_H
