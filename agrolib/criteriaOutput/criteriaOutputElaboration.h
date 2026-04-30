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

    int computeAllDtxUnit(const QSqlDatabase &db, const QString &idCase, QString &errorStr);

    int computeAllDtxPeriod(const QSqlDatabase &db, const QString &idCase, unsigned int period,
                            std::vector<double> &dtx, QString &errorStr);

    bool writeDtxToDB(const QSqlDatabase &db, const QString &idCase, const std::vector<double> &dt30,
                      const std::vector<double> &dt90, const std::vector<double> &dt180, QString &errorStr);

    int writeCsvOutputUnit(const QString &idCase, const QString &idCropClass, const QList<QString> &dataTables,
                           const QSqlDatabase &dbData, const QSqlDatabase &dbCrop, const QSqlDatabase &dbClimateData,
                           const QDate &dateComputation, const CriteriaOutputVariable &outputVariable,
                           const QString &csvFileName, int &nrMissingData, QString &errorStr);

    int selectSimpleVar(const QSqlDatabase &db, const QString &idCase, const QString &varName, const QString &computation,
                        const QDate &firstDate, const QDate &lastDate, float irriRatio,
                        std::vector<float> &resultVector, QString &errorStr);

    int computeDTX(const QSqlDatabase &db, const QString &idCase, int period, const QString &computation,
                   const QDate &firstDate, const QDate &lastDate, std::vector<float> &resultVector, QString &errorStr);

    int writeCsvAggrFromShape(Crit3DShapeHandler &refShapeFile, const QString &csvFileName, const QDate &dateComputation,
                              const QList<QString> &outputVarName, const QString &shapeField, QString &errorStr);

    int orderCsvByField(const QString &csvFileName, const QString &field, QString &errorStr);


#endif // CRITERIAOUTPUTELABORATION_H
