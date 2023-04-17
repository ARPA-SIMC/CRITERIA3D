#ifndef CRITERIAOUTPUTELABORATION_H
#define CRITERIAOUTPUTELABORATION_H

    #include <QString>
    #include <QDate>
    #include <QSqlDatabase>
    #include <vector>

    #ifndef CRITERIAOUTPUTVARIABLE_H
        #include "criteriaOutputVariable.h"
    #endif
    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif

    int computeAllDtxUnit(QSqlDatabase db, QString idCase, QString &error);

    int computeAllDtxPeriod(QSqlDatabase db, QString idCase, unsigned int period,
                            std::vector<double> &dtx, QString &error);

    bool writeDtxToDB(QSqlDatabase db, QString idCase, std::vector<double> &dt30,
                      std::vector<double> &dt90, std::vector<double> &dt180, QString &error);

    int writeCsvOutputUnit(QString idCase, QString idCropClass, QSqlDatabase &dbData, QSqlDatabase &dbCrop,
                           QSqlDatabase &dbHistoricalData, QDate dateComputation,
                           CriteriaOutputVariable outputVariable, QString csvFileName, QString &error);

    int selectSimpleVar(QSqlDatabase &db, QString idCase, QString varName, QString computation, QDate firstDate,
                        QDate lastDate, float irriRatio, std::vector<float> &resVector, QString &error);

    int computeDTX(QSqlDatabase &db, QString idCase, int period, QString computation, QDate firstDate,
                   QDate lastDate, std::vector<float> &resVector, QString &error);

    int writeCsvAggrFromShape(Crit3DShapeHandler &refShapeFile, QString csvFileName, QDate dateComputation,
                              QList<QString> outputVarName, QString shapeField, QString &error);

    int orderCsvByField(QString csvFileName, QString field, QString &error);


#endif // CRITERIAOUTPUTELABORATION_H
