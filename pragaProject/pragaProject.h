#ifndef PRAGAPROJECT_H
#define PRAGAPROJECT_H

    #ifndef CRIT3DCLIMATE_H
        #include "crit3dClimate.h"
    #endif

    #ifndef PROJECT_H
        #include "project.h"
    #endif

    #ifndef METEOMAPS_H
        #include "meteoMaps.h"
    #endif

    #ifndef PRAGAMETEOMAPS_H
        #include "pragaMeteoMaps.h"
    #endif

    #ifdef NETCDF
        #include "netcdfHandler.h"
    #endif

    #ifndef INOUTDATAXML_H
        #include "inOutDataXML.h"
    #endif

    #ifndef DROUGHT_H
        #include "drought.h"
    #endif

    #ifndef POINTSTATISTICSWIDGET_H
        #include "pointStatisticsWidget.h"
    #endif

    #ifndef HOMOGENEITYWIDGET_H
        #include "homogeneityWidget.h"
    #endif

    #ifndef SYNCHRONICITYWIDGET_H
        #include "synchronicityWidget.h"
    #endif

    class PragaProject : public Project
    {
    private:

    private slots:
            void deleteSynchWidget();

    public:
        QString projectPragaFolder;
        QList<QString> users;

        gis::Crit3DRasterGrid dataRaster;
        Crit3DDailyMeteoMaps* pragaDailyMaps;
        PragaHourlyMeteoMaps* pragaHourlyMaps;

        aggregationMethod grdAggrMethod;

        Crit3DClimate* clima;
        Crit3DClimate* climaFromDb;
        Crit3DClimate* referenceClima;
        bool lastElabTargetisGrid;

        bool isElabMeteoPointsValue;
        QString climateIndex;

        QSettings* pragaDefaultSettings;
        std::map<QString, QList<int> > idArkimetHourlyMap;
        std::map<QString, QList<int> > idArkimetDailyMap;

        Crit3DPointStatisticsWidget* pointStatisticsWidget;
        Crit3DHomogeneityWidget* homogeneityWidget;
        Crit3DSynchronicityWidget* synchronicityWidget;

        std::string synchReferencePoint;

        InOutDataXML* inOutData;

        Crit3DMeteoPointsDbHandler* outputMeteoPointsDbHandler;
        bool outputMeteoPointsLoaded;

        #ifdef NETCDF
            NetCDFHandler netCDF;
        #endif

        PragaProject();

        void initializePragaProject();
        void clearPragaProject();

        void createPragaProject(QString path_, QString name_, QString description_);
        void savePragaProject();
        void savePragaParameters();

        bool loadPragaProject(QString myFileName);
        bool loadPragaSettings();

        void closeOutputMeteoPointsDB();
        bool loadOutputMeteoPointsDB(const QString &fileName);
        bool writeMeteoPointsProperties(const QList<QString> &joinedPropertiesList, const QList<QString> &csvFields,
                                        const QList<QList<QString>> &csvData, bool isOutputPoints);

        gis::Crit3DRasterGrid* getPragaMapFromVar(meteoVariable myVar);

        bool downloadDailyDataArkimet(QList<QString> variables, bool prec0024, QDate startDate, QDate endDate, bool showInfo);
        bool downloadHourlyDataArkimet(QList<QString> variables, QDate startDate, QDate endDate, bool showInfo);

        bool interpolationOutputPointsPeriod(QDate dateIni, QDate lastDate, QList <meteoVariable> variables);

        bool deriveVariableMeteoGrid(meteoVariable myVar, frequencyType myFrequency, const Crit3DTime& myTime);
        bool interpolationMeteoGrid(meteoVariable myVar, frequencyType myFrequency, const Crit3DTime& myTime);
        bool interpolationMeteoGridPeriod(QDate dateIni, QDate dateFin, QList <meteoVariable> variables, QList <meteoVariable> aggrVariables,
                                          QList<meteoVariable> derivedVariables, int nrDaysLoading, int nrDaysSaving);
        bool interpolationCrossValidationPeriod(QDate dateIni, QDate dateFin, meteoVariable myVar, QString filename);
        bool saveGrid(meteoVariable myVar, frequencyType myFrequency, const Crit3DTime& myTime, bool showInfo);
        bool timeAggregateGridVarHourlyInDaily(meteoVariable dailyVar, Crit3DDate dateIni, Crit3DDate dateFin);
        bool timeAggregateGrid(QDate dateIni, QDate dateFin, QList <meteoVariable> variables, bool loadData, bool saveData);
        bool computeDailyVariablesPoint(Crit3DMeteoPoint *meteoPoint, QDate first, QDate last, QList <meteoVariable> variables);
        bool derivedVariablesMeteoGridPeriod(QDate first, QDate last, QList<meteoVariable> variables, bool useNetRad);
        bool elaborationCyclePoints(bool isAnomaly, bool showInfo);
        bool elaborationCycleGrid(bool isAnomaly, bool showInfo);
        bool elaborationCheck(bool isMeteoGrid, bool isAnomaly);
        bool computeElaboration(bool isMeteoGrid, bool isAnomaly, bool isClimate, bool showInfo);
        bool showClimateFields(bool isMeteoGrid, QList<QString> *climateDbElab, QList<QString> *climateDbVarList);
        void readClimate(bool isMeteoGrid, QString climateSelected, int climateIndex, bool showInfo);
        bool deleteClimate(bool isMeteoGrid, QString climaSelected);
        bool climateCyclePoints(bool showInfo);
        bool climateCycleGrid(bool showInfo);
        bool averageSeriesOnZonesMeteoGrid(meteoVariable variable, meteoComputation elab1MeteoComp,
                                           QString aggregationString, float threshold, gis::Crit3DRasterGrid* zoneGrid,
                                           QDate startDate, QDate endDate, bool showInfo);
        bool getIsElabMeteoPointsValue() const;
        void setIsElabMeteoPointsValue(bool value);
        bool dbMeteoPointDataCount(QDate myFirstDate, QDate myLastDate, meteoVariable myVar, QString dataset, std::vector<int> &myCounter);
        bool dbMeteoGridMissingData(QDate myFirstDate, QDate myLastDate, meteoVariable myVar, QList<QDate> &dateList, QList<QString> &idList);

        int executePragaCommand(QList<QString> argumentList, bool* isCommandFound);
        bool parserXMLImportExportData(QString xmlName, bool isGrid);
        bool loadXMLImportData(QString fileName);
        bool loadXMLExportData(QString code, QDateTime myFirstTime, QDateTime myLastTime);
        bool loadXMLExportDataGrid(QString code, QDateTime myFirstTime, QDateTime myLastTime);
        bool monthlyAggregateVariablesGrid(const QDate &firstDate, const QDate &lastDate, QList <meteoVariable> &variablesList);
        bool computeDroughtIndexGrid(droughtIndex index, int firstYear, int lastYear, QDate date, int timescale, meteoVariable myVar);
        bool computeDroughtIndexPoint(droughtIndex index, int timescale, int refYearStart, int refYearEnd);
        bool computeDroughtIndexPointGUI(droughtIndex index, int timescale, int refYearStart, int refYearEnd, QDate myDate);
        void showPointStatisticsWidgetPoint(std::string idMeteoPoint);
        void showHomogeneityTestWidgetPoint(std::string idMeteoPoint);
        void showSynchronicityTestWidgetPoint(std::string idMeteoPoint);
        void setSynchronicityReferencePoint(std::string idMeteoPoint);
        void showPointStatisticsWidgetGrid(std::string id);
        bool activeMeteoGridCellsWithDEM();
        bool planGriddingTask(QDate dateIni, QDate dateFin, QString user, QString notes);
        bool getGriddingTasks(std::vector<QDateTime> &timeCreation, std::vector<QDate> &dateStart, std::vector<QDate> &dateEnd,
                                                        std::vector<QString> &users, std::vector<QString> &notes);
        bool removeGriddingTask(QDateTime dateCreation, QString user, QDate dateStart, QDate dateEnd);
        bool computeClimatePointXML(QString xmlName);
        bool cleanClimatePoint();
        bool saveLogProceduresGrid(QString nameProc, QDate date);

        bool dailyZoneAggregationMeteoGrid(meteoVariable variable, const QString& aggregationString, float threshold,
                                           gis::Crit3DRasterGrid* zoneGrid, std::vector<std::vector<float>> &zoneValues, std::vector<float> &outputSeries,
                                           std::vector<std::vector<int>> &indexRowCol,
                                           std::vector<std::vector<int>> &meteoGridRow, std::vector<std::vector<int>> &meteoGridcol,
                                           const Crit3DDate& startDate, int nrDays, bool showInfo);

        bool hourlyZoneAggregationMeteoGrid(meteoVariable variable, const QString& aggregationString, float threshold,
                                           gis::Crit3DRasterGrid* zoneRaster, std::vector<std::vector<float>> &zoneValues, std::vector<float> &outputSeries,
                                           std::vector<std::vector<int>> &indexRowCol,
                                           std::vector<std::vector<int>> &meteoGridRow, std::vector<std::vector<int>> &meteoGridcol,
                                           const Crit3DDate& startDate, int nrDays, bool showInfo);

        #ifdef NETCDF
                bool exportMeteoGridToNetCDF(QString fileName, QString title, QString variableName, std::string variableUnit, Crit3DDate myDate, int nDays, int refYearStart, int refYearEnd);
                bool exportXMLElabGridToNetcdf(QString xmlName);
        #endif
    };


#endif // PRAGAPROJECT_H
