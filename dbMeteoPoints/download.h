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

            bool getPointProperties(QList<QString> datasetList);
            bool getPointPropertiesFromId(QString id, Crit3DMeteoPoint* pointProp);
            QMap<QString,QString> getArmiketIdList(QList<QString> datasetList);
            void downloadMetadata(QJsonObject obj);
            bool downloadDailyData(QDate startDate, QDate endDate, QString dataset, QList<QString> stations, QList<int> variables, bool prec0024);
            bool downloadHourlyData(QDate startDate, QDate endDate, QString dataset, QList<QString> stations, QList<int> variables);

            DbArkimet* getDbArkimet();

        private:
            QList<QString> _datasetsList;
            DbArkimet* _dbMeteo;

            static const QByteArray _authorization;

    };

#endif // DOWNLOAD_H
