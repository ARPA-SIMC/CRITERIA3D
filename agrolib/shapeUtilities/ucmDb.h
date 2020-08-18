#ifndef UCMDB_H
#define UCMDB_H

    #include <QObject>
    #include <QString>
    #include <QSqlDatabase>

    class UcmDb : public QObject
    {
        Q_OBJECT

    public:
        UcmDb(QString dbname);
        ~UcmDb();

        void createUnitsTable();

        bool writeListToUnitsTable(QStringList idCase, QStringList idCrop, QStringList idMeteo,
                                   QStringList idSoil, QList<double> ha);
        QString getError() const;

    private:
        QString error;
        QSqlDatabase db;
    };


#endif // UCMDB_H
