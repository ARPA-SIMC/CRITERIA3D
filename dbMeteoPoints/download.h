#ifndef DOWNLOAD_H
#define DOWNLOAD_H

    #ifndef DBARKIMET_H
        #include "dbArkimet.h"
    #endif

    class Download : public QObject
    {
        Q_OBJECT
        public:
            explicit Download(QString dbName, QObject* parent = nullptr): QObject(parent)
            { _dbMeteo = new DbArkimet(dbName); }

            ~Download() { delete _dbMeteo; }

            DbArkimet* getDbArkimet() { return _dbMeteo; }

            bool getPointProperties(const QList<QString> &datasetList, int utmZone, QString &errorString);
            bool getPointPropertiesFromId(const QString &id, int utmZone, Crit3DMeteoPoint &pointProp);

            QMap<QString,QString> getArmiketIdList(QList<QString> datasetList);
            void downloadMetadata(const QJsonObject &obj, int utmZone);

            bool downloadDailyData(const QDate &startDate, const QDate &endDate, const QString &dataset,
                                   QList<QString> &stations, QList<int> &variables, bool prec0024, QString &errorString);

            bool downloadHourlyData(const QDate &startDate, const QDate &endDate, const QString &dataset,
                                    const QList<QString> &stationList, const QList<int> &varList, QString &errorString);

        private:
            QList<QString> _datasetsList;
            DbArkimet* _dbMeteo;

            static const QByteArray _authorization;

    };

#endif // DOWNLOAD_H
