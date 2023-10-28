#ifndef DBMETEOGRID_H
#define DBMETEOGRID_H

    #ifndef METEO_H
        #include "meteo.h"
    #endif
    #ifndef METEOGRID_H
        #include "meteoGrid.h"
    #endif

    #ifndef QSQLDATABASE_H
        #include <QSqlDatabase>
    #endif
    #ifndef QMAP_H
        #include <QMap>
    #endif
    #ifndef QDOM_H
        #include <QDomElement>
    #endif
    #ifndef QDATETIME_H
        #include <QDate>
    #endif


    struct TXMLConnection
    {
        QString provider;
        QString server;
        QString name;
        QString user;
        QString password;
    };


    struct TXMLvar
    {
        QString varField;
        int varCode;
        QString varPragaName;
    };

    struct TXMLTable
    {
         bool exists;
         QString fieldTime;
         QString fieldVarCode;
         QString fieldValue;
         QString prefix;
         QString postFix;
         std::vector<TXMLvar> varcode;
    };


    class Crit3DMeteoGridDbHandler
    {
    public:

        Crit3DMeteoGridDbHandler();
        ~Crit3DMeteoGridDbHandler();

        bool openDatabase(QString *myError);
        bool openDatabase(QString *myError, QString connectionName);
        bool newDatabase(QString *myError);
        bool newDatabase(QString *myError, QString connectionName);
        bool deleteDatabase(QString *myError);
        void closeDatabase();
        bool parseXMLFile(QString xmlFileName, QDomDocument* xmlDoc, QString *error);
        bool checkXML(QString *myError);
        bool parseXMLGrid(QString xmlFileName, QString *myError);
        void initMapMySqlVarType();

        QSqlDatabase db() const;
        QString fileName() const;
        TXMLConnection connection() const;
        Crit3DMeteoGridStructure gridStructure() const;
        Crit3DMeteoGrid *meteoGrid() const;
        QDate firstDate() const;
        QDate lastDate() const;
        TXMLTable tableDaily() const;
        TXMLTable tableHourly() const;
        TXMLTable tableMonthly() const;
        QString tableDailyModel() const;
        QString tableHourlyModel() const;

        void setMeteoGrid(Crit3DMeteoGrid *meteoGrid);
        void setDb(const QSqlDatabase &db);
        void setFirstDate(const QDate &firstDate);
        void setLastDate(const QDate &lastDate);

        int getDailyVarCode(meteoVariable meteoGridDailyVar);
        QString getDailyVarField(meteoVariable meteoGridDailyVar);
        meteoVariable getDailyVarEnum(int varCode);
        meteoVariable getDailyVarFieldEnum(QString varField);

        int getHourlyVarCode(meteoVariable meteoGridHourlyVar);
        QString getHourlyVarField(meteoVariable meteoGridHourlyVar);
        meteoVariable getHourlyVarEnum(int varCode);
        meteoVariable getHourlyVarFieldEnum(QString varField);

        int getMonthlyVarCode(meteoVariable meteoGridMonthlyVar);
        QString getMonthlyVarField(meteoVariable meteoGridMonthlyVar);
        meteoVariable getMonthlyVarEnum(int varCode);
        meteoVariable getMonthlyVarFieldEnum(QString varField);

        std::string getDailyPragaName(meteoVariable meteoVar);
        std::string getHourlyPragaName(meteoVariable meteoVar);
        std::string getMonthlyPragaName(meteoVariable meteoVar);

        bool loadCellProperties(QString *myError);
        bool newCellProperties(QString *myError);
        bool writeCellProperties(QString *myError, int nRow, int nCol);
        bool loadIdMeteoProperties(QString *myError, QString idMeteo);
        bool updateGridDate(QString *myError);

        bool loadGridDailyData(QString &myError, QString meteoPoint, QDate first, QDate last);
        bool loadGridDailyDataFixedFields(QString &myError, QString meteoPoint, QDate first, QDate last);
        bool loadGridDailyDataEnsemble(QString &myError, QString meteoPoint, int memberNr, QDate first, QDate last);
        bool loadGridHourlyData(QString &myError, QString meteoPoint, QDateTime first, QDateTime last);
        bool loadGridHourlyDataFixedFields(QString &myError, QString meteoPoint, QDateTime first, QDateTime last);
        bool loadGridHourlyDataEnsemble(QString &myError, QString meteoPoint, int memberNr, QDateTime first, QDateTime last);
        bool loadGridMonthlyData(QString &myError, QString meteoPoint, QDate first, QDate last);

        std::vector<float> loadGridDailyVar(QString *myError, QString meteoPoint, meteoVariable variable, QDate first, QDate last, QDate *firstDateDB);
        std::vector<float> loadGridDailyVarFixedFields(QString *myError, QString meteoPoint, meteoVariable variable, QDate first, QDate last, QDate* firstDateDB);
        std::vector<float> loadGridHourlyVar(QString *myError, QString meteoPoint, meteoVariable variable, QDateTime first, QDateTime last, QDateTime* firstDateDB);
        std::vector<float> loadGridHourlyVarFixedFields(QString *myError, QString meteoPoint, meteoVariable variable, QDateTime first, QDateTime last, QDateTime* firstDateDB);
        bool getYearList(QString *myError, QString meteoPoint, QList<QString>* yearList);
        bool idDailyList(QString *myError, QList<QString>* idMeteoList);

        bool saveGridData(QString *myError, QDateTime firstTime, QDateTime lastTime, QList<meteoVariable> meteoVariableList, Crit3DMeteoSettings *meteoSettings);
        bool saveGridHourlyData(QString *myError, QDateTime firstDate, QDateTime lastDate, QList<meteoVariable> meteoVariableList);
        bool saveGridDailyData(QString *myError, QDateTime firstDate, QDateTime lastDate, QList<meteoVariable> meteoVariableList, Crit3DMeteoSettings *meteoSettings);
        bool deleteAndWriteCellGridDailyData(QString& myError, QString meteoPointID, int row, int col, QDate firstDate, QDate lastDate,
                                             QList<meteoVariable> meteoVariableList, Crit3DMeteoSettings* meteoSettings);
        bool saveCellGridDailyData(QString *myError, QString meteoPointID, int row, int col, QDate firstDate, QDate lastDate, QList<meteoVariable> meteoVariableList, Crit3DMeteoSettings *meteoSettings);
        bool saveCellGridDailyDataFF(QString *myError, QString meteoPointID, int row, int col, QDate firstDate, QDate lastDate, Crit3DMeteoSettings *meteoSettings);
        bool saveCellGridDailyDataEnsemble(QString *myError, QString meteoPointID, int row, int col, QDate firstDate, QDate lastDate,
                                                             QList<meteoVariable> meteoVariableList, int memberNr, Crit3DMeteoSettings *meteoSettings);
        bool saveCellGridMonthlyData(QString *myError, QString meteoPointID, int row, int col, QDate firstDate, QDate lastDate,
                                                             QList<meteoVariable> meteoVariableList);
        bool saveListDailyDataEnsemble(QString *myError, QString meteoPointID, QDate date, meteoVariable meteoVar, QList<float> values);
        bool saveListDailyData(QString *myError, QString meteoPointID, QDate firstDate, meteoVariable meteoVar, QList<float> values, bool reverseOrder);
        bool cleanDailyOldData(QString *myError, QDate date);
        bool saveListHourlyData(QString *myError, QString meteoPointID, QDateTime firstDateTime, meteoVariable meteoVar, QList<float> values);
        bool saveCellCurrentGridDaily(QString *myError, QString meteoPointID, QDate date, int varCode, float value);

        bool saveCellCurrentGridDailyList(QString meteoPointID, QList<QString> listEntries, QString &errorStr);
        bool saveCellCurrentGridHourlyList(QString meteoPointID, QList<QString> listEntries, QString &errorStr);

        bool saveCellCurrentGridDailyFF(QString &errorStr, QString meteoPointID, QDate date, QString varPragaName, float value);
        bool saveCellGridHourlyData(QString *myError, QString meteoPointID, int row, int col, QDateTime firstTime, QDateTime lastTime, QList<meteoVariable> meteoVariableList);
        bool saveCellGridHourlyDataFF(QString *myError, QString meteoPointID, int row, int col, QDateTime firstTime, QDateTime lastTime);
        bool saveCellGridHourlyDataEnsemble(QString *myError, QString meteoPointID, int row, int col,
                                                              QDateTime firstTime, QDateTime lastTime, QList<meteoVariable> meteoVariableList, int memberNr);
        bool saveCellCurrentGridHourly(QString& errorStr, QString meteoPointID, QDateTime dateTime, int varCode, float value);
        bool saveCellCurrentGridHourlyFF(QString &errorStr, QString meteoPointID, QDateTime dateTime, QString varPragaName, float value);
        bool activeAllCells(QString *myError);
        bool setActiveStateCellsInList(QString *myError, QList<QString> idList, bool activeState);

        bool exportDailyDataCsv(QString &errorStr, bool isTPrec, QDate firstDate, QDate lastDate, QString idListFileName, QString outputPath);
        bool MeteoGridToRasterFlt(double cellSize, const gis::Crit3DGisSettings &gisSettings, gis::Crit3DRasterGrid& myGrid);

        QDate getFirstDailyDate() const;
        QDate getLastDailyDate() const;
        QDate getFirstHourlyDate() const;
        QDate getLastHourlyDate() const;
        QDate getFirsMonthlytDate() const;
        QDate getLastMonthlyDate() const;

        bool saveLogProcedures(QString *myError, QString nameProc, QDate date);

    private:

        QString _fileName;
        QSqlDatabase _db;
        TXMLConnection _connection;
        Crit3DMeteoGridStructure _gridStructure;
        Crit3DMeteoGrid* _meteoGrid;

        QDate _firstDate;
        QDate _lastDate;

        QDate _firstDailyDate;
        QDate _lastDailyDate;
        QDate _firstHourlyDate;
        QDate _lastHourlyDate;
        QDate _firsMonthlytDate;
        QDate _lastMonthlyDate;

        TXMLTable _tableDaily;
        TXMLTable _tableHourly;
        TXMLTable _tableMonthly;

        QMap<meteoVariable, int> _gridDailyVar;
        QMap<meteoVariable, int> _gridHourlyVar;
        QMap<meteoVariable, int> _gridMonthlyVar;

        QMap<meteoVariable, QString> _gridDailyVarField;
        QMap<meteoVariable, QString> _gridHourlyVarField;
        QMap<meteoVariable, QString> _gridMonthlyVarField;

        QString _tableDailyModel;
        QString _tableHourlyModel;

        QMap<meteoVariable, QString> _mapDailyMySqlVarType;
        QMap<meteoVariable, QString> _mapHourlyMySqlVarType;

    };


#endif // DBMETEOGRID_H
