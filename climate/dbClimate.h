#ifndef DBCLIMATE_H
#define DBCLIMATE_H

    #ifndef QSQLDATABASE_H
        #include <QSqlDatabase>
    #endif

    #ifndef _VECTOR_
        #include <vector>
    #endif

    bool saveDailyElab(QSqlDatabase db, QString *myError, QString id, std::vector<float> allResults, QString elab);
    bool saveDecadalElab(QSqlDatabase db, QString *myError, QString id, std::vector<float> allResults, QString elab);
    bool saveMonthlyElab(QSqlDatabase db, QString *myError, QString id, std::vector<float> allResults, QString elab);
    bool saveSeasonalElab(QSqlDatabase db, QString *myError, QString id, std::vector<float> allResults, QString elab);
    bool saveAnnualElab(QSqlDatabase db, QString *myError, QString id, float result, QString elab);
    bool saveGenericElab(QSqlDatabase db, QString *myError, QString id, float result, QString elab);

    bool getClimateFieldsFromTable(QSqlDatabase db, QString *myError, QString climateTable, QList<QString>* fieldList);
    bool selectVarElab(QSqlDatabase db, QString *myError, QString table, QString variable, QList<QString>* listElab);
    bool getClimateTables(QSqlDatabase db, QString *myError, QList<QString>* climateTables);

    bool deleteElab(QSqlDatabase db, QString *myError, QString table, QString elab);

    float readClimateElab(const QSqlDatabase &db, const QString &table, const int &timeIndex, const QString &id, const QString &elab, QString *myError);
    QList<QString> getIdListFromElab(QSqlDatabase db, QString table, QString *myError, QString elab);
    QList<QString> getIdListFromElab(QSqlDatabase db, QString table, QString *myError, QString elab, int index);


#endif // DBCLIMATE_H
