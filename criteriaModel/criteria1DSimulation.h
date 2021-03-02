#ifndef CRITERIA1DSIMULATION_H
#define CRITERIA1DSIMULATION_H

    #ifndef CRITERIA1DCASE_H
        #include "criteria1DCase.h"
    #endif
    #ifndef COMPUTATIONUNITSDB_H
        #include "computationUnitsDb.h"
    #endif
    #ifndef QDATETIME_H
        #include <QDate>
    #endif
    #ifndef DBMETEOGRID_H
        #include "dbMeteoGrid.h"
    #endif


class Crit1DSimulation
    {

    public:
        // DATABASE
        QSqlDatabase dbCrop;
        QSqlDatabase dbSoil;
        QSqlDatabase dbMeteo;
        QSqlDatabase dbForecast;
        QSqlDatabase dbOutput;
        QSqlDatabase dbState;

        Crit3DMeteoGridDbHandler* observedMeteoGrid;
        Crit3DMeteoGridDbHandler* forecastMeteoGrid;

        bool isXmlGrid;

        Crit1DCase myCase;
        QString outputString;

        // soil
        soil::Crit3DTextureClass soilTexture[13];
        soil::Crit3DFittingOptions fittingOptions;

        // dates
        QDate firstSimulationDate;
        QDate lastSimulationDate;

        // seasonal forecast
        bool isSaveState;
        bool isRestart;
        bool isSeasonalForecast;
        int firstSeasonMonth;
        std::vector<float> seasonalForecasts;
        int nrSeasonalForecasts;

        // short term forecast
        bool isShortTermForecast;
        int daysOfForecast;

        Crit1DSimulation();

        bool runModel(const Crit1DUnit &myUnit, QString &myError);
        bool createState(QString &myError);

    private:

        bool setSoil(QString soilCode, QString &myError);

        bool setMeteoSqlite(QString idMeteo, QString idForecast, QString *myError);
        bool setMeteoXmlGrid(QString idMeteo, QString idForecast, QString *myError);

        void initializeSeasonalForecast(const Crit3DDate& firstDate, const Crit3DDate& lastDate);

        void updateSeasonalForecast(Crit3DDate myDate, int *index);

        bool createOutputTable(QString &myError);

        bool saveState(QString &myError);
        bool restoreState(QString dbStateToRestoreName, QString &myError);
        void prepareOutput(Crit3DDate myDate, bool isFirst);
        bool saveOutput(QString &myError);

    };


    QString getOutputStringNullZero(double value);
    QString getId5Char(QString id);


#endif // CRITERIA1DSIMULATION_H
