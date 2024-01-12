#ifndef DBCLIMATE_H
#define DBCLIMATE_H

    #ifndef QSQLDATABASE_H
        #include <QSqlDatabase>
    #endif

    bool saveDailyElab(QSqlDatabase db, QString *myError, QString id, std::vector<float> allResults, QString elab);
    bool saveDecadalElab(QSqlDatabase db, QString *myError, QString id, std::vector<float> allResults, QString elab);
    bool saveMonthlyElab(QSqlDatabase db, QString *myError, QString id, std::vector<float> allResults, QString elab);
    bool saveSeasonalElab(QSqlDatabase db, QString *myError, QString id, std::vector<float> allResults, QString elab);
    bool saveAnnualElab(QSqlDatabase db, QString *myError, QString id, float result, QString elab);
    bool saveGenericElab(QSqlDatabase db, QString *myError, QString id, float result, QString elab);

    bool selectAllElab(QSqlDatabase db, QString *myError, QString table, QList<QString>* listElab);
    bool selectVarElab(QSqlDatabase db, QString *myError, QString table, QString variable, QList<QString>* listElab);
    bool showClimateTables(QSqlDatabase db, QString *myError, QList<QString>* climateTables);

    bool deleteElab(QSqlDatabase db, QString *myError, QString table, QString elab);

    QList<float> readElab(const QSqlDatabase &db, const QString &table, const QString &id, const QString &elab, QString *myError);
    QList<QString> getIdListFromElab(QSqlDatabase db, QString table, QString *myError, QString elab);
    QList<QString> getIdListFromElab(QSqlDatabase db, QString table, QString *myError, QString elab, int index);


#endif // DBCLIMATE_H
