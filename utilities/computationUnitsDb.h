#ifndef COMPUTATIONUNITSDB_H
#define COMPUTATIONUNITSDB_H

    #include <QString>
    #include <QSqlDatabase>

    class ComputationUnitsDB
    {
    public:
        ComputationUnitsDB(QString dbname);

        void clear();

        void createUnitsTable();

        bool writeListToUnitsTable(QStringList idCase, QStringList idCrop, QStringList idMeteo,
                                   QStringList idSoil, QList<double> ha);
        QString getError() const;



    private:
        QString error;
        QSqlDatabase db;
    };


#endif // COMPUTATIONUNITSDB_H
