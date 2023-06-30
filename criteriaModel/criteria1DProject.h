#ifndef CRITERIA1DPROJECT_H
#define CRITERIA1DPROJECT_H

    #ifndef LOGGER_H
        #include "logger.h"
    #endif
    #ifndef DBMETEOGRID_H
        #include "dbMeteoGrid.h"
    #endif
    #ifndef CRITERIA1DCASE_H
        #include "criteria1DCase.h"
    #endif
    #ifndef CARBON_NITROGEN_MODEL_H
        #include "carbonNitrogenModel.h"
    #endif

    #include <fstream>

    class Crit1DProject
    {

    public:
        bool isProjectLoaded;
        QString projectError;
        Logger logger;

        QString path;

        // database
        QString dbCropName;
        QString dbSoilName;
        QString dbOutputName;
        QString dbMeteoName;
        QString dbForecastName;
        QString dbComputationUnitsName;

        QSqlDatabase dbCrop;
        QSqlDatabase dbSoil;
        QSqlDatabase dbMeteo;
        Crit3DMeteoGridDbHandler* observedMeteoGrid;

        // dates
        QDate firstSimulationDate;
        QDate lastSimulationDate;

        bool isXmlMeteoGrid;

        Crit1DCase myCase;
        Crit1DCarbonNitrogenProfile myCarbonNitrogenProfile;

        // soil
        std::vector<soil::Crit3DTextureClass> texturalClassList;
        std::vector<soil::Crit3DGeotechnicsClass> geotechnicsClassList;

        std::vector<Crit1DCompUnit> compUnitList;

        Crit1DProject();

        void initialize();
        int initializeProject(QString settingsFileName);
        int computeAllUnits();
        bool computeUnit(const Crit1DCompUnit& myUnit);

    private:
        QString projectName;
        QString configFileName;

        // save/restart
        bool isSaveState;
        bool isRestart;

        // forecast period
        bool isYearlyStatistics;
        bool isSeasonalForecast;
        bool isMonthlyForecast;
        bool isShortTermForecast;

        int firstSeasonMonth;
        int daysOfForecast;
        int nrYears;
        std::vector<float> irriSeries;
        std::vector<float> precSeries;

        QString outputString;
        QString logFileName;
        QString outputCsvFileName;
        std::ofstream outputCsvFile;

        bool addDateTimeLogFile;

        // specific output
        std::vector<int> waterContentDepth;
        std::vector<int> waterPotentialDepth;
        std::vector<int> waterDeficitDepth;
        std::vector<int> awcDepth;
        std::vector<int> availableWaterDepth;
        std::vector<int> fractionAvailableWaterDepth;
        std::vector<int> factorOfSafetyDepth;

        // DATABASE
        QSqlDatabase dbForecast;
        QSqlDatabase dbOutput;
        QSqlDatabase dbState;

        Crit3DMeteoGridDbHandler* forecastMeteoGrid;

        void closeProject();
        bool readSettings();
        void closeAllDatabase();
        int openAllDatabase();
        void checkSimulationDates();

        bool setSoil(QString soilCode, QString &errorStr);

        bool setMeteoSqlite(QString idMeteo, QString idForecast);
        bool setMeteoXmlGrid(QString idMeteo, QString idForecast, unsigned int memberNr);

        bool setPercentileOutputCsv();
        void updateMonthlyForecastOutput(Crit3DDate myDate, unsigned int memberNr);
        void initializeIrrigationStatistics(const Crit3DDate &firstDate, const Crit3DDate &lastDate);
        void updateIrrigationStatistics(Crit3DDate myDate, int &index);
        bool computeIrrigationStatistics(unsigned int index, float irriRatio);
        bool computeMonthlyForecast(unsigned int unitIndex, float irriRatio);

        bool computeCase(unsigned int memberNr);
        bool computeUnit(unsigned int unitIndex, unsigned int memberNr);

        bool createOutputTable(QString &myError);
        bool createDbState(QString &myError);
        bool saveState(QString &myError);
        bool restoreState(QString dbStateToRestoreName, QString &myError);
        void updateOutput(Crit3DDate myDate, bool isFirst);
        bool saveOutput(QString &errorStr);

    };


    QString getOutputStringNullZero(double value);
    bool setVariableDepth(const QList<QString> &depthList, std::vector<int> &variableDepth);


#endif // CRITERIA1DPROJECT_H
