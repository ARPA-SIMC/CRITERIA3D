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

        QString fileName() const { return _fileName; }
        TXMLConnection connection() const { return _connection; }

        QSqlDatabase db() const { return _db; }
        void setDb(const QSqlDatabase &db) { _db = db; }

        QDate firstDate() const { return _firstDate; }
        QDate lastDate() const { return _lastDate; }

        void setFirstDate(const QDate &firstDate) { _firstDate = firstDate; }
        void setLastDate(const QDate &lastDate) { _lastDate = lastDate; }

        Crit3DMeteoGridStructure gridStructure() const { return _gridStructure; }

        Crit3DMeteoGrid *meteoGrid() const { return _meteoGrid; }
        void setMeteoGrid(Crit3DMeteoGrid *meteoGrid) { _meteoGrid = meteoGrid; }

        TXMLTable tableHourly() const { return _tableHourly; }
        TXMLTable tableDaily() const { return _tableDaily; }
        TXMLTable tableMonthly() const { return _tableMonthly; }

        QString tableDailyModel() const { return _tableDailyModel; }
        QString tableHourlyModel() const { return _tableHourlyModel; }

        bool openDatabase(QString *errorStr);
        bool openDatabase(QString *errorStr, QString connectionName);
        bool newDatabase(QString *errorStr);
        bool newDatabase(QString *errorStr, QString connectionName);
        bool deleteDatabase(QString *errorStr);
        void closeDatabase();
        bool parseXMLFile(QString xmlFileName, QDomDocument* xmlDoc, QString *error);
        bool checkXML(QString *errorStr);
        bool parseXMLGrid(QString xmlFileName, QString *errorStr);
        void initMapMySqlVarType();

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

        bool loadCellProperties(QString *errorStr);
        bool newCellProperties(QString *errorStr);
        bool writeCellProperties(QString *errorStr, int nRow, int nCol);
        bool loadIdMeteoProperties(QString *errorStr, QString idMeteo);
        bool updateMeteoGridDate(QString &errorStr);

        bool loadGridDailyData(QString &errorStr, const QString &meteoPointId, const QDate &firstDate, const QDate &lastDate);
        bool loadGridDailyDataFixedFields(QString &errorStr, QString meteoPoint, QDate first, QDate last);
        bool loadGridDailyDataEnsemble(QString &errorStr, QString meteoPoint, int memberNr, QDate first, QDate last);
        bool loadGridDailyMeteoPrec(QString &errorStr, const QString &meteoPointId, const QDate &firstDate, const QDate &lastDate);
        bool loadGridHourlyData(QString &errorStr, QString meteoPoint, QDateTime firstDate, QDateTime lastDate);
        bool loadGridHourlyDataFixedFields(QString &errorStr, QString meteoPoint, QDateTime first, QDateTime last);
        bool loadGridHourlyDataEnsemble(QString &errorStr, QString meteoPoint, int memberNr, QDateTime first, QDateTime last);
        bool loadGridMonthlyData(QString &errorStr, QString meteoPoint, QDate firstDate, QDate lastDate);
        bool loadGridAllMonthlyData(QString &errorStr, QDate firstDate, QDate lastDate);
        bool loadGridMonthlySingleDate(QString &errorStr, const QString &meteoPoint, const QDate &myDate);

        std::vector<float> loadGridDailyVar(QString *errorStr, QString meteoPoint, meteoVariable variable, QDate first, QDate last, QDate *firstDateDB);
        std::vector<float> loadGridDailyVarFixedFields(QString *errorStr, QString meteoPoint, meteoVariable variable, QDate first, QDate last, QDate* firstDateDB);
        std::vector<float> loadGridHourlyVar(QString *errorStr, QString meteoPoint, meteoVariable variable, QDateTime first, QDateTime last, QDateTime* firstDateDB);
        std::vector<float> loadGridHourlyVarFixedFields(QString *errorStr, QString meteoPoint, meteoVariable variable, QDateTime first, QDateTime last, QDateTime* firstDateDB);
        std::vector<float> exportAllDataVar(QString *errorStr, frequencyType freq, meteoVariable variable, QString id, QDateTime myFirstTime, QDateTime myLastTime, std::vector<QString> &dateStr);
        bool getYearList(QString *errorStr, QString meteoPoint, QList<QString>* yearList);
        bool idDailyList(QString *errorStr, QList<QString>* idMeteoList);

        bool saveGridData(QString *errorStr, QDateTime firstTime, QDateTime lastTime, QList<meteoVariable> meteoVariableList, Crit3DMeteoSettings *meteoSettings);
        bool saveGridHourlyData(QString *errorStr, QDateTime firstDate, QDateTime lastDate, QList<meteoVariable> meteoVariableList);
        bool saveGridDailyData(QString *errorStr, QDateTime firstDate, QDateTime lastDate, QList<meteoVariable> meteoVariableList, Crit3DMeteoSettings *meteoSettings);
        bool deleteAndWriteCellGridDailyData(QString& errorStr, QString meteoPointID, int row, int col, QDate firstDate, QDate lastDate,
                                             QList<meteoVariable> meteoVariableList, Crit3DMeteoSettings* meteoSettings);
        bool saveCellGridDailyData(QString *errorStr, QString meteoPointID, int row, int col, QDate firstDate, QDate lastDate, QList<meteoVariable> meteoVariableList, Crit3DMeteoSettings *meteoSettings);
        bool saveCellGridDailyDataFF(QString *errorStr, QString meteoPointID, int row, int col, QDate firstDate, QDate lastDate, Crit3DMeteoSettings *meteoSettings);
        bool saveCellGridDailyDataEnsemble(QString *errorStr, QString meteoPointID, int row, int col, QDate firstDate, QDate lastDate,
                                                             QList<meteoVariable> meteoVariableList, int memberNr, Crit3DMeteoSettings *meteoSettings);
        bool saveCellGridMonthlyData(QString *errorStr, QString meteoPointID, int row, int col, QDate firstDate, QDate lastDate,
                                                             QList<meteoVariable> meteoVariableList);
        bool saveListDailyDataEnsemble(QString *errorStr, QString meteoPointID, QDate date, meteoVariable meteoVar, QList<float> values);
        bool saveListDailyData(QString *errorStr, QString meteoPointID, QDate firstDate, meteoVariable meteoVar, QList<float> values, bool reverseOrder);
        bool cleanDailyOldData(QString *errorStr, QDate date);
        bool saveListHourlyData(QString *errorStr, QString meteoPointID, QDateTime firstDateTime, meteoVariable meteoVar, QList<float> values);
        bool saveCellCurrentGridDaily(QString *errorStr, QString meteoPointID, QDate date, int varCode, float value);

        bool saveCellCurrentGridDailyList(QString meteoPointID, QList<QString> listEntries, QString &errorStr);
        bool saveCellCurrentGridHourlyList(QString meteoPointID, QList<QString> listEntries, QString &errorStr);

        bool saveCellCurrentGridDailyFF(QString &errorStr, QString meteoPointID, QDate date, QString varPragaName, float value);
        bool saveCellGridHourlyData(QString *errorStr, QString meteoPointID, int row, int col, QDateTime firstTime, QDateTime lastTime, QList<meteoVariable> meteoVariableList);
        bool saveCellGridHourlyDataFF(QString *errorStr, QString meteoPointID, int row, int col, QDateTime firstTime, QDateTime lastTime);
        bool saveCellGridHourlyDataEnsemble(QString *errorStr, QString meteoPointID, int row, int col,
                                                              QDateTime firstTime, QDateTime lastTime, QList<meteoVariable> meteoVariableList, int memberNr);
        bool saveCellCurrentGridHourly(QString& errorStr, QString meteoPointID, QDateTime dateTime, int varCode, float value);
        bool saveCellCurrentGridHourlyFF(QString &errorStr, QString meteoPointID, QDateTime dateTime, QString varPragaName, float value);
        bool activeAllCells(QString *errorStr);
        bool setActiveStateCellsInList(QString *errorStr, QList<QString> idList, bool activeState);

        bool saveDailyDataCsv(const QString &csvFileName, const QList<meteoVariable> &variableList,
                              const QDate &firstDate, const QDate &lastDate, unsigned row, unsigned col, QString &errorStr);

        bool exportDailyDataCsv(const QList<meteoVariable> &variableList, const QDate &firstDate,
                                const QDate &lastDate, const QString &idListFileName, QString &outputPath, QString &errorStr);

        bool MeteoGridToRasterFlt(double cellSize, const gis::Crit3DGisSettings &gisSettings, gis::Crit3DRasterGrid& myGrid);

        QDate getFirstDailyDate() const;
        QDate getLastDailyDate() const;
        QDate getFirstHourlyDate() const;
        QDate getLastHourlyDate() const;
        QDate getFirstMonthlytDate() const;
        QDate getLastMonthlyDate() const;

        bool isDaily();
        bool isHourly();
        bool isMonthly();

        bool saveLogProcedures(QString *errorStr, QString nameProc, QDate date);

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
        QDate _firstMonthlyDate;
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
