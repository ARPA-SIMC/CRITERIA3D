#ifndef DBMETEOPOINTS_H
#define DBMETEOPOINTS_H

    #ifndef CRIT3DDATE_H
        #include "crit3dDate.h"
    #endif
    #ifndef METEOPOINT_H
        #include "meteoPoint.h"
    #endif
    #ifndef INTERPOLATIONSETTINGS_H
        #include "interpolationSettings.h"
    #endif

    #ifndef QSQLDATABASE_H
        #include <QSqlDatabase>
    #endif
    #ifndef QOBJECT_H
        #include <QObject>
    #endif


    class Crit3DMeteoPointsDbHandler : public QObject
    {
        Q_OBJECT

    private:
        QString _errorStr;

    public:
        explicit Crit3DMeteoPointsDbHandler();
        explicit Crit3DMeteoPointsDbHandler(QString dbname_);
        explicit Crit3DMeteoPointsDbHandler(QString provider_, QString host_, QString dbname_, int port_, QString user_, QString pass_);

        ~Crit3DMeteoPointsDbHandler();

        QString getDbName() { return _db.databaseName(); }
        QSqlDatabase getDb() const { return _db; }
        void setDb(const QSqlDatabase &db) { _db = db; }
        QString getErrorString() { return _errorStr; }
        void setErrorString(QString str) { _errorStr = str; }

        QString getDatasetURL(QString dataset, bool &isOk);
        bool setAndOpenDb(QString dbname_);

        QList<QString> getAllDatasetsList();
        QList<QString> getDatasetsActive();
        void setDatasetsActive(QString active);
        QString getDatasetFromId(const QString& idPoint);

        QDateTime getFirstDate(frequencyType frequency);
        QDateTime getLastDate(frequencyType frequency);
        QDateTime getFirstDate(frequencyType frequency, std::string idMeteoPoint);
        QDateTime getLastDate(frequencyType frequency, std::string idMeteoPoint);

        bool existData(const Crit3DMeteoPoint &meteoPoint, frequencyType myFreq);
        bool deleteData(QString pointCode, frequencyType myFreq, QDate first, QDate last);
        bool deleteData(QString pointCode, frequencyType myFreq, QList<meteoVariable> varList, QDate first, QDate last);
        bool deleteAllData(frequencyType myFreq);
        bool writePointProperties(const Crit3DMeteoPoint &pointProp);
        bool updatePointProperties(const QList<QString> &columnList, const QList<QString> &valueList);
        bool updatePointPropertiesGivenId(QString id, QList<QString> columnList, QList<QString> valueList);
        bool getPropertiesFromDb(QList<Crit3DMeteoPoint>& meteoPointsList,
                                 const gis::Crit3DGisSettings& gisSettings, QString& errorString);
        bool getPropertiesGivenId(const QString &id, Crit3DMeteoPoint &meteoPoint,
                                  const gis::Crit3DGisSettings& gisSettings, QString& errorString);

        bool loadDailyData(const Crit3DDate &firstDate, const Crit3DDate &lastDate, Crit3DMeteoPoint &meteoPoint);

        std::vector<float> loadDailyVar(meteoVariable variable, const Crit3DDate &dateStart, const Crit3DDate &dateEnd,
                                        const Crit3DMeteoPoint &meteoPoint, QDate &firstDateDB);

        std::vector<float> exportAllDataVar(QString *myError, frequencyType freq, meteoVariable variable, QString id, QDateTime myFirstTime, QDateTime myLastTime, std::vector<QString> &dateStr);

        bool loadHourlyData(const Crit3DDate &firstDate, const Crit3DDate &lastDate, Crit3DMeteoPoint &meteoPoint);

        std::vector<float> loadHourlyVar(meteoVariable variable, const QString& meteoPointId, const QDateTime& startTime,
                                         const QDateTime& endTime, QDateTime &firstDateDB, QString &myError);

        bool loadVariableProperties();
        bool getFieldList(const QString &tableName, QList<QString> &fieldList);
        int getIdfromMeteoVar(meteoVariable meteoVar);
        int getArkIdFromVar(const QString& variable);
        std::map<int, meteoVariable> getMapIdMeteoVar() const;

        bool existIdPoint(const QString& idPoint);
        bool createTable(const QString& tableName, bool deletePrevious);
        QString getNewDataEntry(int pos, const QList<QString>& dataStr, const QString& dateTimeStr,
                            const QString& idVarStr, meteoVariable myVar,
                            int* nrMissingData, int* nrWrongData, Crit3DQuality* dataQuality);
        bool importHourlyMeteoData(const QString &fileNameComplete, bool deletePreviousData, QString &log);

        bool writeDailyDataList(const QString &pointCode, const QList<QString> &listEntries, QString& log);
        bool writeHourlyDataList(const QString &pointCode, const QList<QString> &listEntries, QString& log);

        bool setAllPointsActive();
        bool setAllPointsNotActive();
        bool setActiveStatePointList(const QList<QString> &pointList, bool activeState);

        bool deleteAllPointsFromIdList(const QList<QString> &pointList);
        bool deleteAllPointsFromGeoPointList(const QList<gis::Crit3DGeoPoint>& pointList);
        bool deleteAllPointsFromDataset(QList<QString> datasets);

        QList<QString> getIdList();
        QList<QString> getIdListGivenDataset(QList<QString> datasets);
        QList<QString> getMunicipalityList();
        QList<QString> getProvinceList();
        QList<QString> getRegionList();
        QList<QString> getStateList();
        QList<QString> getDatasetList();
        bool setActiveStateIfCondition(bool activeState, const QString &condition);
        bool getPointListWithCriteria(QList<QString> &selectedPointsList, const QString &condition);
        bool setOrogCode(QString id, int orogCode);
        QList<QString> getJointStations(const QString& idPoint);
        bool setJointStations(const QString& idPoint, const QList<QString> &stationsList);
        QString getNameGivenId(QString id);
        double getAltitudeGivenId(QString id);

    protected:

        QSqlDatabase _db;
        std::map<int, meteoVariable> _mapIdMeteoVar;
    signals:

    protected slots:
    };


#endif // DBMETEOPOINTS_H
