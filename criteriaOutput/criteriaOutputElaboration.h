#ifndef CRITERIAOUTPUTELABORATION_H
#define CRITERIAOUTPUTELABORATION_H

    #include <QString>
    #include <QDate>
    #include <QSqlDatabase>
    #include "vector"

    #ifndef CRITERIAOUTPUTVARIABLE_H
        #include "criteriaOutputVariable.h"
    #endif
    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif

    int computeAllDtxUnit(QSqlDatabase db, QString idCase, QString& projectError);

    int computeAllDtxPeriod(QSqlDatabase db, QString idCase, unsigned int period,
                            std::vector<double> &dtx, QString& projectError);

    bool writeDtxToDB(QSqlDatabase db, QString idCase, std::vector<double>& dt30,
                      std::vector<double>& dt90, std::vector<double>& dt180, QString& projectError);

    int writeCsvOutputUnit(QString idCase, QString idCropClass, QSqlDatabase dbData, QSqlDatabase dbCrop,
                           QSqlDatabase dbDataHistorical, QDate dateComputation,
                           CriteriaOutputVariable outputVariable, QString csvFileName, QString* projectError);

    int selectSimpleVar(QSqlDatabase db, QString idCase, QString varName, QString computation, QDate firstDate,
                        QDate lastDate, float irriRatio, QVector<float>* resVector, QString *projectError);

    int computeDTX(QSqlDatabase db, QString idCase, int period, QString computation, QDate firstDate,
                   QDate lastDate, QVector<float>* resVector, QString *projectError);

    int writeCsvAggrFromShape(Crit3DShapeHandler &refShapeFile, QString csvFileName, QDate dateComputation,
                              QStringList outputVarName, QString shapeField, QString &error);

    int orderCsvByField(QString csvFileName, QString field, QString &error);


#endif // CRITERIAOUTPUTELABORATION_H
