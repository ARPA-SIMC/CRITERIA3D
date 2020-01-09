#ifndef DBMETEOGRID_H
#define DBMETEOGRID_H

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

        QSqlDatabase db() const;

        void setDb(const QSqlDatabase &db);

        QString fileName() const;

        TXMLConnection connection() const;

        Crit3DMeteoGridStructure gridStructure() const;

        Crit3DMeteoGrid *meteoGrid() const;

        void setMeteoGrid(Crit3DMeteoGrid *meteoGrid);

        QDate firstDate() const;

        void setFirstDate(const QDate &firstDate);

        QDate lastDate() const;

        void setLastDate(const QDate &lastDate);

        TXMLTable tableDaily() const;

        TXMLTable tableHourly() const;

        QString tableDailyModel() const;

        QString tableHourlyModel() const;

        bool openDatabase(QString *myError);

        void closeDatabase();

        bool parseXMLFile(QString xmlFileName, QDomDocument* xmlDoc, QString *error);

        void initMapMySqlVarType();

        bool checkXML(QString *myError);

        bool parseXMLGrid(QString xmlFileName, QString *myError);

        int getDailyVarCode(meteoVariable meteoGridDailyVar);

        QString getDailyVarField(meteoVariable meteoGridDailyVar);

        meteoVariable getDailyVarEnum(int varCode);

        meteoVariable getDailyVarFieldEnum(QString varField);

        int getHourlyVarCode(meteoVariable meteoGridHourlyVar);

        QString getHourlyVarField(meteoVariable meteoGridHourlyVar);

        meteoVariable getHourlyVarEnum(int varCode);

        meteoVariable getHourlyVarFieldEnum(QString varField);

        std::string getDailyPragaName(meteoVariable meteoVar);

        std::string getHourlyPragaName(meteoVariable meteoVar);

        bool loadCellProperties(QString *myError);

        bool updateGridDate(QString *myError);

        bool loadGridDailyData(QString *myError, QString meteoPoint, QDate first, QDate last);

        bool loadGridDailyDataFixedFields(QString *myError, QString meteoPoint, QDate first, QDate last);

        bool loadGridHourlyData(QString *myError, QString meteoPoint, QDateTime first, QDateTime last);

        bool loadGridHourlyDataFixedFields(QString *myError, QString meteoPoint, QDateTime first, QDateTime last);

        std::vector<float> loadGridDailyVar(QString *myError, QString meteoPoint, meteoVariable variable, QDate first, QDate last, QDate *firstDateDB);

        std::vector<float> loadGridDailyVarFixedFields(QString *myError, QString meteoPoint, meteoVariable variable, QDate first, QDate last, QDate* firstDateDB);

        std::vector<float> loadGridHourlyVar(QString *myError, QString meteoPoint, meteoVariable variable, QDateTime first, QDateTime last, QDateTime* firstDateDB);

        std::vector<float> loadGridHourlyVarFixedFields(QString *myError, QString meteoPoint, meteoVariable variable, QDateTime first, QDateTime last, QDateTime* firstDateDB);

        bool saveGridData(QString *myError, QDateTime firstTime, QDateTime lastTime, QList<meteoVariable> meteoVariableList);
        bool saveGridHourlyData(QString *myError, QDateTime firstDate, QDateTime lastDate, QList<meteoVariable> meteoVariableList);
        bool saveGridDailyData(QString *myError, QDateTime firstDate, QDateTime lastDate, QList<meteoVariable> meteoVariableList);

        bool saveCellGridDailyData(QString *myError, QString meteoPointID, int row, int col, QDate firstDate, QDate lastDate, QList<meteoVariable> meteoVariableList);
        bool saveCellGridDailyDataFF(QString *myError, QString meteoPointID, int row, int col, QDate firstDate, QDate lastDate);

        bool saveCellCurrentGridDaily(QString *myError, QString meteoPointID, QDate date, int varCode, float value);
        bool saveCellCurrentGridDailyFF(QString *myError, QString meteoPointID, QDate date, QString varPragaName, float value);

        bool saveCellGridHourlyData(QString *myError, QString meteoPointID, int row, int col, QDateTime firstDate, QDateTime lastDate, QList<meteoVariable> meteoVariableList);
        bool saveCellGridHourlyDataFF(QString *myError, QString meteoPointID, int row, int col, QDateTime firstDate, QDateTime lastDate);

        bool saveCellCurrentGridHourly(QString *myError, QString meteoPointID, QDateTime dateTime, int varCode, float value);
        bool saveCellCurrentGridHourlyFF(QString *myError, QString meteoPointID, QDateTime dateTime, QString varPragaName, float value);

    private:

        QString _fileName;
        QSqlDatabase _db;
        TXMLConnection _connection;
        Crit3DMeteoGridStructure _gridStructure;
        Crit3DMeteoGrid* _meteoGrid;

        QDate _firstDate;
        QDate _lastDate;

        TXMLTable _tableDaily;
        TXMLTable _tableHourly;

        QMap<meteoVariable, int> _gridDailyVar;
        QMap<meteoVariable, int> _gridHourlyVar;

        QMap<meteoVariable, QString> _gridDailyVarField;
        QMap<meteoVariable, QString> _gridHourlyVarField;

        QString _tableDailyModel;
        QString _tableHourlyModel;

        QMap<QString, QString> _mapDailyMySqlVarType;
        QMap<QString, QString> _mapHourlyMySqlVarType;

    };


#endif // DBMETEOGRID_H
