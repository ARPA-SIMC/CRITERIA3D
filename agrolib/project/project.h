#ifndef PROJECT_H
#define PROJECT_H

#include <deque>
#ifndef QUALITY_H
        #include "quality.h"
    #endif
    #ifndef GIS_H
        #include "gis.h"
    #endif
    #ifndef OUTPUTPOINTS_H
        #include "outputPoints.h"
    #endif
    #ifndef WELL_H
        #include "well.h"
    #endif
    #ifndef DBOUTPUTPOINTSHANDLER_H
        #include "dbOutputPointsHandler.h"
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
    #ifndef PROXYWIDGET_H
        #include "proxyWidget.h"
    #endif
    #ifndef LOCALPROXYWIDGET_H
        #include "localProxyWidget.h"
    #endif
    #ifndef WATERTABLE_H
        #include "waterTable.h"
    #endif

    #ifndef _FSTREAM_
        #include <fstream>
    #endif

    #define ERROR_NONE 0
    #define ERROR_SETTINGS 1
    #define ERROR_DEM 2
    #define ERROR_DBPOINT 3
    #define ERROR_DBGRID 4
    #define ERROR_OUTPUTPOINTLIST 5

    #define ERROR_STR_MISSING_DB "Load a meteo points database before."
    #define ERROR_STR_MISSING_DEM "Load a Digital Elevation Model (DEM) before."
    #define ERROR_STR_MISSING_PROJECT "Open a project before."
    #define ERROR_STR_MISSING_GRID "Load a meteo grid DB before."
    #define ERROR_STR_MISSING_NETCDF "Load a NetCDF file before."
    #define ERROR_STR_MISSING_POINT_GRID "Load a meteo points DB or a meteo grid before."

    class Crit3DMeteoWidget;
    class FormInfo;

    class Project : public QObject {
        Q_OBJECT

    private:
        QString appPath;
        QString defaultPath;
        QString projectPath;
        bool computeOnlyPoints;
        FormInfo* formLog;

        void clearMeteoPoints();
        bool createDefaultProject(QString fileName);
        bool searchDefaultPath(QString* defaultPath);

        bool verboseStdoutLogging;

    protected:
        frequencyType _currentFrequency;
        meteoVariable _currentVariable;
        QDate _currentDate;
        int _currentHour;

    public:
        QString projectName = "";
        bool isProjectLoaded;
        int modality;
        QString currentTileMap;

        bool requestedExit;
        QString errorString;
        int errorType;

        QString logFileName;
        QString demFileName;
        QString dbPointsFileName;
        QString dbAggregationFileName;
        QString aggregationPath;
        QString dbGridXMLFileName;
        QString parametersFileName;
        QString glocalMapName;
        QString glocalPointsName;
        std::ofstream logFile;

        // output points
        QString outputPointsFileName;
        QString currentDbOutputFileName;

        QSettings* parametersSettings;
        QSettings* projectSettings;

        bool meteoPointsLoaded;
        int nrMeteoPoints;
        Crit3DMeteoPoint* meteoPoints;
        std::vector<gis::Crit3DOutputPoint> outputPoints;
        std::vector<Well> wellPoints;

        Crit3DMeteoPointsDbHandler* meteoPointsDbHandler;
        Crit3DOutputPointsDbHandler* outputPointsDbHandler;
        Crit3DAggregationsDbHandler* aggregationDbHandler;
        QDateTime meteoPointsDbFirstTime;
        QDateTime meteoPointsDbLastTime;

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
        Crit3DCrossValidationStatistics crossValidationStatistics;
        std::vector<Crit3DCrossValidationStatistics> glocalCrossValidationStatistics;

        std::vector <Crit3DProxyGridSeries> proxyGridSeries;

        Crit3DClimateParameters climateParameters;

        QList<Crit3DMeteoWidget*> meteoWidgetPointList;
        QList<Crit3DMeteoWidget*> meteoWidgetGridList;

        Crit3DProxyWidget* proxyWidget;
        Crit3DLocalProxyWidget* localProxyWidget;

        QList<WaterTable> waterTableList;

        Project();

        void initializeProject();
        void clearProject();

        void createProject(QString path_, QString name_, QString description);
        void saveProject();
        void saveProjectLocation();
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
        bool checkProxy(Crit3DProxy &myProxy, QString *error, bool isActive);
        bool addProxyToProject(std::vector <Crit3DProxy> proxyList, std::deque <bool> proxyActive, std::vector <int> proxyOrder);
        void addProxyGridSeries(QString name_, std::vector <QString> gridNames, std::vector <unsigned> gridYears);
        void checkProxyForMultipleDetrending(Crit3DProxy &myProxy, bool isHeight);
        void setCurrentDate(QDate myDate);
        void setCurrentHour(int myHour);
        void setCurrentVariable(meteoVariable variable);
        int getCurrentHour();
        QDate getCurrentDate();
        Crit3DTime getCrit3DCurrentTime();
        QDateTime getCurrentTime();
        meteoVariable getCurrentVariable() const;

        void setApplicationPath(QString myPath);
        QString getApplicationPath();
        void setDefaultPath(QString myPath);
        QString getDefaultPath();
        void setProjectPath(QString myPath);
        QString getProjectPath();
        QString getRelativePath(QString fileName);
        QString getCompleteFileName(QString fileName, QString secondaryPath);

        bool setLogFile(QString myFileName);

        void logInfo(QString myStr);
        void logInfoGUI(QString myStr);
        void closeLogInfo();

        void logError(QString myStr);
        void logError();

        void logWarning(QString myStr);
        void logWarning();

        int setProgressBar(QString myStr, int nrValues);
        void updateProgressBar(int value);
        void updateProgressBarText(QString myStr);
        void closeProgressBar();

        void closeMeteoPointsDB();
        void closeMeteoGridDB();
        void cleanMeteoPointsData();

        void closeOutputPointsDB();

        bool loadDEM(const QString & fileName);
        void closeDEM();
        bool loadMeteoPointsData(const QDate &firstDate, const QDate &lastDate, bool loadHourly, bool loadDaily, bool showInfo);
        bool loadMeteoPointsData(const QDate &firstDate, const QDate &lastDate, bool loadHourly, bool loadDaily, const QString &dataset, bool showInfo);
        bool loadMeteoPointsDB(QString dbName);
        bool loadMeteoGridDB(QString xmlName);
        bool newMeteoGridDB(QString xmlName);
        bool deleteMeteoGridDB();
        bool loadAggregationdDB(QString dbName);
        bool loadAggregationDBAsMeteoPoints(QString fileName);
        bool loadOutputPointsDB(QString dbName);
        bool newOutputPointsDB(QString dbName);
        bool loadMeteoGridDailyData(QDate firstDate, QDate lastDate, bool showInfo);
        bool loadMeteoGridHourlyData(QDateTime firstDate, QDateTime lastDate, bool showInfo);
        bool loadMeteoGridMonthlyData(QDate firstDate, QDate lastDate, bool showInfo);
        void loadMeteoGridData(QDate firstDate, QDate lastDate, bool showInfo);

        QDateTime findDbPointLastTime();
        QDateTime findDbPointFirstTime();

        void getMeteoPointsRange(float& minimum, float& maximum, bool useNotActivePoints);
        float meteoDataConsistency(meteoVariable myVar, const Crit3DTime& timeIni, const Crit3DTime& timeFin);

        bool loadProxyGrids();
        bool readPointProxyValues(Crit3DMeteoPoint* myPoint, QSqlDatabase* myDb);
        bool readProxyValues();
        bool updateProxy();
        void checkMeteoPointsDEM();
        bool writeTopographicDistanceMaps(bool onlyWithData, bool showInfo);
        bool writeTopographicDistanceMap(int pointIndex, const gis::Crit3DRasterGrid& demMap, QString pathTd);
        bool loadTopographicDistanceMaps(bool onlyWithData, bool showInfo);
        void passInterpolatedTemperatureToHumidityPoints(Crit3DTime myTime, Crit3DMeteoSettings *meteoSettings);
        void passGridTemperatureToHumidityPoints(Crit3DTime myTime, Crit3DMeteoSettings* meteoSettings);
        bool loadGlocalAreasMap();
        bool loadGlocalStationsAndCells(bool isGrid, QString fileNameStations);
        bool loadGlocalWeightMaps(std::vector<Crit3DMacroArea> &myAreas, bool isGrid);
        bool loadGlocalStationsCsv(QString fileName, std::vector<std::vector<std::string> > &areaPoints);
        bool groupCellsInArea(std::vector<int> &areaPoints, int index, bool isGrid);
        bool writeGlocalWeightsMaps(float windowWidth);


        bool checkInterpolation(meteoVariable myVar);
        bool checkInterpolationGrid(meteoVariable myVar);
        bool interpolationGrid(meteoVariable myVar, const Crit3DTime& myTime);
        bool interpolationDemMain(meteoVariable myVar, const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster);
        bool interpolationDem(meteoVariable myVar, const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster);
        bool interpolationDemLocalDetrending(meteoVariable myVar, const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster);
        bool interpolationDemGlocalDetrending(meteoVariable myVar, const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster);
        bool interpolateDemRadiation(const Crit3DTime& myTime, gis::Crit3DRasterGrid *myRaster);
        bool interpolationOutputPoints(std::vector <Crit3DInterpolationDataPoint> &interpolationPoints,
                                       gis::Crit3DRasterGrid *outputGrid, meteoVariable myVar);
        bool interpolationCv(meteoVariable myVar, const Crit3DTime& myTime);
        bool computeResidualsAndStatisticsGlocalDetrending(meteoVariable myVar, std::vector<Crit3DInterpolationDataPoint> &interpolationPoints);

        bool computeStatisticsCrossValidation();
        bool computeStatisticsGlocalCrossValidation(Crit3DMacroArea myArea);
        bool meteoGridAggregateProxy(std::vector<gis::Crit3DRasterGrid *> &myGrids);

        frequencyType getCurrentFrequency() const;
        void setCurrentFrequency(const frequencyType &value);

        bool checkMeteoGridForExport();
        void importHourlyMeteoData(const QString& fileName, bool importAllFiles, bool deletePreviousData);

        gis::Crit3DRasterGrid* getHourlyMeteoRaster(meteoVariable myVar);
        void showMeteoWidgetPoint(std::string idMeteoPoint, std::string namePoint, std::string dataset,
                                  double altitude, std::string lapseRateCode, bool isAppend);
        void showMeteoWidgetGrid(const std::string &idCell, const std::string &dataset, bool isAppend);
        void showProxyGraph(int macroAreaNumber);
        void showLocalProxyGraph(gis::Crit3DGeoPoint myPoint);

        void clearSelectedPoints();
        void clearSelectedOutputPoints();
        bool setActiveStateSelectedPoints(bool isActive);
        bool setActiveStatePointList(QString fileName, bool isActive);
        bool setSelectedStatePointList(QString fileName);
        bool selectPointList(const QList<QString> &pointList);
        bool setActiveStateWithCriteria(bool isActive);
        bool setSelectedStateWithCriteria();
        bool setMarkedFromPointList(QString fileName);
        bool setMarkedPointsOfMacroArea(int areaNumber, bool viewNotActivePoints);
        bool deleteMeteoPoints(const QList<QString>& pointList);
        bool deleteMeteoPointsData(const QList<QString>& pointList);
        bool loadOutputPointList(QString fileName);
        bool writeOutputPointList(QString fileName);

        bool exportMeteoGridToCsv(QString fileName);
        bool exportMeteoGridToRasterFlt(QString fileName, double cellSize);
        bool exportMeteoPointsDailyDataCsv(bool isTPrec, QDate firstDate, QDate lastDate, QString idListFileName, QString outputPath);

        bool loadAndExportMeteoGridToRasterFlt(QString fileName, double cellSize, meteoVariable myVar, QDate dateIni, QDate dateFin);
        int computeDefaultCellSizeFromMeteoGrid(float resolutionRatio);

        void setComputeOnlyPoints(bool value);
        bool getComputeOnlyPoints();

        bool waterTableImportLocation(const QString &csvFileName);
        bool waterTableImportDepths(const QString &csvDepthsFileName);
        bool waterTableComputeSingleWell(int indexWell);
        void waterTableShowSingleWell(const WaterTable &waterTable, const QString &idWell);
        bool waterTableAssignNearestMeteoPoint(bool isMeteoGridLoaded, double wellUtmX, double wellUtmY, QDate firstMeteoDate, Crit3DMeteoPoint* linkedMeteoPoint);
        bool waterTableAssignMeteoData(Crit3DMeteoPoint* linkedMeteoPoint, QDate firstMeteoDate);

        bool assignAltitudeToAggregationPoints();

        Crit3DCrossValidationStatistics getCrossValidationStatistics() const;
        void setCrossValidationStatistics(const Crit3DCrossValidationStatistics &newCrossValidationStatistics);
        void getMeteoPointsCurrentValues(std::vector<float> &validValues);

        bool readVmArkimetData(const QList<QString> &vmFileList, frequencyType frequency, bool isPrec0024);

    private slots:
        void deleteMeteoWidgetPoint(int id);
        void deleteMeteoWidgetGrid(int id);
        void deleteProxyWidget();

    };


#endif // PROJECT_H
