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

    public:
        QString error;

        explicit Crit3DMeteoPointsDbHandler();
        explicit Crit3DMeteoPointsDbHandler(QString dbname_);
        explicit Crit3DMeteoPointsDbHandler(QString provider_, QString host_, QString dbname_, int port_, QString user_, QString pass_);

        ~Crit3DMeteoPointsDbHandler();
        void dbManager();
        QString getDatasetURL(QString dataset);
        QString getDbName();

        QSqlDatabase getDb() const;
        void setDb(const QSqlDatabase &db);
        bool setAndOpenDb(QString dbname_);

        QList<QString> getAllDatasetsList();
        QList<QString> getDatasetsActive();
        void setDatasetsActive(QString active);
        QString getDatasetFromId(const QString& idPoint);

        QDateTime getFirstDate(frequencyType frequency);
        QDateTime getLastDate(frequencyType frequency);
        QDateTime getFirstDate(frequencyType frequency, std::string idMeteoPoint);
        QDateTime getLastDate(frequencyType frequency, std::string idMeteoPoint);

        bool existData(Crit3DMeteoPoint *meteoPoint, frequencyType myFreq);
        bool deleteData(QString pointCode, frequencyType myFreq, QDate first, QDate last);
        bool deleteData(QString pointCode, frequencyType myFreq, QList<meteoVariable> varList, QDate first, QDate last);
        bool deleteAllData(frequencyType myFreq);
        bool writePointProperties(Crit3DMeteoPoint* pointProp);
        bool updatePointProperties(QList<QString> columnList, QList<QString> valueList);
        bool updatePointPropertiesGivenId(QString id, QList<QString> columnList, QList<QString> valueList);
        bool getPropertiesFromDb(QList<Crit3DMeteoPoint>& meteoPointsList,
                                 const gis::Crit3DGisSettings& gisSettings, QString& errorString);
        bool getPropertiesGivenId(QString id, Crit3DMeteoPoint* meteoPoint,
                                                const gis::Crit3DGisSettings& gisSettings, QString& errorString);
        bool loadDailyData(Crit3DDate dateStart, Crit3DDate dateEnd, Crit3DMeteoPoint *meteoPoint);
        std::vector<float> loadDailyVar(QString *myError, meteoVariable variable,
                                        Crit3DDate dateStart, Crit3DDate dateEnd,
                                        QDate* firstDateDB, Crit3DMeteoPoint *meteoPoint);
        bool loadHourlyData(Crit3DDate dateStart, Crit3DDate dateEnd, Crit3DMeteoPoint *meteoPoint);
        std::vector<float> loadHourlyVar(QString *myError, meteoVariable variable,
                                         Crit3DDate dateStart, Crit3DDate dateEnd,
                                         QDateTime* firstDateDB, Crit3DMeteoPoint *meteoPoint);

        bool loadVariableProperties();
        bool getNameColumn(QString tableName, QList<QString>* columnList);
        int getIdfromMeteoVar(meteoVariable meteoVar);
        int getArkIdFromVar(const QString& variable);
        std::map<int, meteoVariable> getMapIdMeteoVar() const;

        bool existIdPoint(const QString& idPoint);
        bool createTable(const QString& tableName, bool deletePrevious);
        QString getNewDataEntry(int pos, const QList<QString>& dataStr, const QString& dateTimeStr,
                            const QString& idVarStr, meteoVariable myVar,
                            int* nrMissingData, int* nrWrongData, Crit3DQuality* dataQuality);
        bool importHourlyMeteoData(QString fileNameComplete, bool deletePreviousData, QString *log);
        bool writeDailyDataList(QString pointCode, QList<QString> listEntries, QString* log);
        bool writeHourlyDataList(QString pointCode, QList<QString> listEntries, QString* log);
        bool writeDroughtDataList(QList<QString> listEntries, QString* log);

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
        bool setActiveStateIfCondition(bool activeState, QString condition);
        bool setOrogCode(QString id, int orogCode);
        QList<QString> getJointStations(const QString& idPoint);
        bool setJointStations(const QString& idPoint, QList<QString> stationsList);
        QString getNameGivenId(QString id);
        double getAltitudeGivenId(QString id);

    protected:

        QSqlDatabase _db;
        std::map<int, meteoVariable> _mapIdMeteoVar;
    signals:

    protected slots:
    };


#endif // DBMETEOPOINTS_H
