#ifndef IRRIGATIONFORECAST_H
#define IRRIGATIONFORECAST_H

    #ifndef CRITERIA1DCASE_H
        #include "criteria1DCase.h"
    #endif
    #ifndef CRITERIA1DUNIT_H
        #include "criteria1DUnit.h"
    #endif
    #ifndef QSQLDATABASE_H
        #include <QSqlDatabase>
    #endif
    #ifndef QDATETIME_H
        #include <QDate>
    #endif


    class Crit1DIrrigationForecast
    {

    public:
        // DATABASE
        QSqlDatabase dbCrop;
        QSqlDatabase dbSoil;
        QSqlDatabase dbMeteo;
        QSqlDatabase dbForecast;
        QSqlDatabase dbOutput;

        bool isXmlGrid;

        Crit1DCase myCase;
        QString outputString;

        // soil
        soil::Crit3DTextureClass soilTexture[13];
        soil::Crit3DFittingOptions fittingOptions;

        // dates
        bool useAllMeteoData;
        QDate firstDate;
        QDate lastDate;

        // seasonal forecast
        bool isSeasonalForecast;
        int firstSeasonMonth;
        std::vector<float> seasonalForecasts;
        int nrSeasonalForecasts;

        // short term forecast
        bool isShortTermForecast;
        int daysOfForecast;

        Crit1DIrrigationForecast();

        bool runModel(const Crit1DUnit &myUnit, QString &myError);

    private:

        bool setSoil(QString soilCode, QString &myError);

        bool setMeteoSqlite(QString idMeteo, QString idForecast, QString *myError);
        bool setMeteoXmlGrid(QString idMeteo, QString idForecast, QString *myError);

        void initializeSeasonalForecast(const Crit3DDate& firstDate, const Crit3DDate& lastDate);

        void updateSeasonalForecast(Crit3DDate myDate, int *index);

        bool createOutputTable(QString &myError);
        void prepareOutput(Crit3DDate myDate, bool isFirst);
        bool saveOutput(QString &myError);

    };


    QString getOutputStringNullZero(double value);
    QString getId5Char(QString id);


#endif // IRRIGATIONFORECAST_H
