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

        bool setSoil(QString soilCode, QString &errorStr);

    private:
        QString projectName;
        QString configFileName;

        // save/restart
        bool isSaveState;
        bool isRestart;

        // forecast/climate type
        bool isYearlyStatistics;
        bool isMonthlyStatistics;
        bool isSeasonalForecast;
        bool isEnsembleForecast;
        bool isShortTermForecast;

        int firstMonth;
        int daysOfForecast;
        int nrYears;
        std::vector<float> irriSeries;                  // [mm]
        std::vector<float> precSeries;                  // [mm]

        QString outputString;
        QString outputCsvFileName;
        std::ofstream outputCsvFile;

        bool addDateTimeLogFile;
        bool computeAllSoilDepth;
        double computationSoilDepth;                    // [m]

        // specific output
        bool isClimateOutput;
        std::vector<int> waterContentDepth;             // [cm]
        std::vector<int> degreeOfSaturationDepth;       // [cm]
        std::vector<int> waterPotentialDepth;           // [cm]
        std::vector<int> waterDeficitDepth;             // [cm]
        std::vector<int> awcDepth;                      // [cm]
        std::vector<int> availableWaterDepth;           // [cm]
        std::vector<int> fractionAvailableWaterDepth;   // [cm]
        std::vector<int> factorOfSafetyDepth;           // [cm]

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

        bool setMeteoSqlite(QString idMeteo, QString idForecast);
        bool setMeteoXmlGrid(QString idMeteo, QString idForecast, unsigned int memberNr);

        bool setPercentileOutputCsv();
        void updateMediumTermForecastOutput(Crit3DDate myDate, unsigned int memberNr);
        void initializeIrrigationStatistics(const Crit3DDate &firstDate, const Crit3DDate &lastDate);
        void updateIrrigationStatistics(Crit3DDate myDate, int &currentIndex);
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
