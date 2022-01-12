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
        QString dbUnitsName;

        QSqlDatabase dbCrop;
        QSqlDatabase dbSoil;
        QSqlDatabase dbMeteo;

        // dates
        QDate firstSimulationDate;
        QDate lastSimulationDate;

        bool isXmlMeteoGrid;

        // soil
        soil::Crit3DTextureClass soilTexture[13];

        std::vector<Crit1DUnit> unitList;

        Crit1DProject();

        void initialize();
        int initializeProject(QString settingsFileName);
        int computeAllUnits();
        bool computeUnit(const Crit1DUnit& myUnit);

    private:
        QString projectName;
        QString configFileName;

        // save/restart
        bool isSaveState;
        bool isRestart;

        // seasonal forecast
        bool isSeasonalForecast;
        bool isMonthlyForecast;
        bool isShortTermForecast;

        int firstSeasonMonth;
        int daysOfForecast;
        int nrForecasts;
        std::vector<float> forecastIrr;
        std::vector<float> forecastPrec;

        QString outputString;

        QString logFileName;
        std::ofstream logFile;

        bool addDateTimeLogFile;

        QString outputCsvFileName;
        std::ofstream outputCsvFile;

        // specific output
        std::vector<int> waterContentDepth;
        std::vector<int> waterPotentialDepth;
        std::vector<int> waterDeficitDepth;
        std::vector<int> awcDepth;

        // DATABASE
        QSqlDatabase dbForecast;
        QSqlDatabase dbOutput;
        QSqlDatabase dbState;

        Crit3DMeteoGridDbHandler* observedMeteoGrid;
        Crit3DMeteoGridDbHandler* forecastMeteoGrid;

        Crit1DCase myCase;

        void closeProject();
        bool readSettings();
        void closeAllDatabase();
        int openAllDatabase();
        void checkSimulationDates();

        bool setSoil(QString soilCode, QString &myError);

        bool setMeteoSqlite(QString idMeteo, QString idForecast);
        bool setMeteoXmlGrid(QString idMeteo, QString idForecast, int memberNr);

        bool setPercentileOutputCsv();
        void updateSeasonalForecastOutput(Crit3DDate myDate, int &index);
        void updateMonthlyForecastOutput(Crit3DDate myDate, unsigned int memberNr);
        void initializeSeasonalForecast(const Crit3DDate &firstDate, const Crit3DDate &lastDate);
        bool computeSeasonalForecast(unsigned int index, float irriRatio);
        bool computeMonthlyForecast(unsigned int unitIndex, float irriRatio);

        bool computeCase(unsigned int memberNr);
        bool computeUnit(unsigned int unitIndex, unsigned int memberNr);

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
