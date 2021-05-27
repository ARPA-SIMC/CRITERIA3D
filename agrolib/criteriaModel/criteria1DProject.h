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

    #include <QDate>
    #include <fstream>

    class Crit1DProject
    {

    public:
        QString projectError;
        Logger logger;

        // dates
        QDate firstSimulationDate;
        QDate lastSimulationDate;

        Crit1DProject();

        int initializeProject(QString settingsFileName);
        int computeAllUnits();

    private:
        bool isProjectLoaded;

        QString path;
        QString projectName;
        QString configFileName;

        // save/restart
        bool isSaveState;
        bool isRestart;

        // seasonal forecast
        bool isSeasonalForecast;
        int firstSeasonMonth;
        std::vector<float> seasonalForecasts;
        int nrSeasonalForecasts;

        // short term forecast
        bool isShortTermForecast;
        unsigned int daysOfForecast;

        // monthly forecast
        bool isMonthlyForecast;

        QString dbCropName;
        QString dbSoilName;
        QString dbOutputName;
        QString dbMeteoName;
        QString dbForecastName;
        QString dbUnitsName;

        QString outputString;

        QString logFileName;
        std::ofstream logFile;

        bool isXmlGrid;
        bool addDateTimeLogFile;

        QString outputCsvFileName;
        std::ofstream outputCsvFile;

        // specific output
        std::vector<int> waterDeficitDepth;
        std::vector<int> waterContentDepth;
        std::vector<int> waterPotentialDepth;

        // DATABASE
        QSqlDatabase dbCrop;
        QSqlDatabase dbSoil;
        QSqlDatabase dbMeteo;
        QSqlDatabase dbForecast;
        QSqlDatabase dbOutput;
        QSqlDatabase dbState;

        Crit3DMeteoGridDbHandler* observedMeteoGrid;
        Crit3DMeteoGridDbHandler* forecastMeteoGrid;

        std::vector<Crit1DUnit> unitList;

        // soil
        soil::Crit3DTextureClass soilTexture[13];

        Crit1DCase myCase;

        void initialize();
        void closeProject();
        bool readSettings();
        void closeAllDatabase();
        int openAllDatabase();
        void checkSimulationDates();

        bool setSoil(QString soilCode, QString &myError);

        bool setMeteoSqlite(QString idMeteo, QString idForecast);
        bool setMeteoXmlGrid(QString idMeteo, QString idForecast);

        bool setPercentileOutputCsv();
        void updateSeasonalForecastOutput(Crit3DDate myDate, int &index);
        void initializeSeasonalForecast(const Crit3DDate &firstDate, const Crit3DDate &lastDate);
        bool computeSeasonalForecast(unsigned int index, double irriRatio);
        bool computeMonthlyForecast(unsigned int index, double irriRatio);

        bool computeUnit(unsigned int unitIndex);

        bool createOutputTable(QString &myError);
        bool createState(QString &myError);
        bool saveState(QString &myError);
        bool restoreState(QString dbStateToRestoreName, QString &myError);
        void prepareOutput(Crit3DDate myDate, bool isFirst);
        bool saveOutput(QString &myError);

    };


    QString getOutputStringNullZero(double value);
    bool setVariableDepth(QStringList &depthList, std::vector<int> &variableDepth);


#endif // CRITERIA1DPROJECT_H
