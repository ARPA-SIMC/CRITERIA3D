#ifndef DOWNLOAD_H
#define DOWNLOAD_H

    #ifndef DBARKIMET_H
        #include "dbArkimet.h"
    #endif

    class Download : public QObject
    {
        Q_OBJECT
        public:
            explicit Download(QString dbName, QObject* parent = nullptr);
            ~Download();

            bool getPointProperties(const QList<QString> &datasetList, QString &errorString);
            bool getPointPropertiesFromId(QString id, Crit3DMeteoPoint* pointProp);
            QMap<QString,QString> getArmiketIdList(QList<QString> datasetList);
            void downloadMetadata(QJsonObject obj);

            bool downloadDailyData(const QDate &startDate, const QDate &endDate, const QString &dataset,
                                   QList<QString> &stations, QList<int> &variables, bool prec0024, QString &errorString);

            bool downloadHourlyData(const QDate &startDate, const QDate &endDate, const QString &dataset,
                                    const QList<QString> &stationList, const QList<int> &varList, QString &errorString);

            DbArkimet* getDbArkimet();

        private:
            QList<QString> _datasetsList;
            DbArkimet* _dbMeteo;

            static const QByteArray _authorization;

    };

#endif // DOWNLOAD_H
