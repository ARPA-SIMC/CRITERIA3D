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

            bool getPointProperties(QStringList datasetList);
            void downloadMetadata(QJsonObject obj);
            bool downloadDailyData(QDate startDate, QDate endDate, QString dataset, QStringList stations, QList<int> variables, bool prec0024);
            bool downloadHourlyData(QDate startDate, QDate endDate, QString dataset, QStringList stations, QList<int> variables);

            DbArkimet* getDbArkimet();

        private:
            QStringList _datasetsList;
            DbArkimet* _dbMeteo;

            static const QByteArray _authorization;

    };

#endif // DOWNLOAD_H
