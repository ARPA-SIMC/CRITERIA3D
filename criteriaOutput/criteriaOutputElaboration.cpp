#include "criteriaOutputElaboration.h"
#include "criteriaOutputProject.h"
#include "commonConstants.h"
#include "basicMath.h"
#include "utilities.h"
#include "cropDbQuery.h"
#include <QtSql>


int computeAllDtxUnit(QSqlDatabase db, QString idCase, QString &projectError)
{
    // check if table exist (skip otherwise)
    if (! db.tables().contains(idCase))
    {
        return CRIT3D_OK;
    }

    QSqlQuery qry(db);

    // check if DTX column should be added
    bool insertTD30Col = true;
    bool insertTD90Col = true;
    bool insertTD180Col = true;
    QString statement = QString("PRAGMA table_info(`%1`)").arg(idCase);
    QString name;
    if( !qry.exec(statement) )
    {
        projectError = qry.lastError().text();
        return ERROR_DBHISTORICAL;
    }
    qry.first();
    if (!qry.isValid())
    {
        projectError = qry.lastError().text();
        return ERROR_DBHISTORICAL ;
    }
    do
    {
        getValue(qry.value("name"), &name);
        if (name == "DT30")
        {
            insertTD30Col = false;
        }
        else if (name == "DT90")
        {
            insertTD90Col = false;
        }
        else if (name == "DT180")
        {
            insertTD180Col = false;
        }
    }
    while(qry.next());

    // add column DT30, DT90, DT180
    if (insertTD30Col)
    {
        statement = QString("ALTER TABLE `%1` ADD COLUMN DT30 REAL").arg(idCase);
        if( !qry.exec(statement) )
        {
            projectError = qry.lastError().text();
            return ERROR_DBHISTORICAL;
        }
    }
    if (insertTD90Col)
    {
        statement = QString("ALTER TABLE `%1` ADD COLUMN DT90 REAL").arg(idCase);
        if( !qry.exec(statement) )
        {
            projectError = qry.lastError().text();
            return ERROR_DBHISTORICAL;
        }
    }
    if (insertTD180Col)
    {
        statement = QString("ALTER TABLE `%1` ADD COLUMN DT180 REAL").arg(idCase);
        if( !qry.exec(statement) )
        {
            projectError = qry.lastError().text();
            return ERROR_DBHISTORICAL;
        }
    }

    // check if table is full (skip otherwise)
    qry.prepare("SELECT * FROM " + idCase);
    qry.exec();
    if (!qry.first())
    {
        return CRIT3D_OK;
    }
    qry.clear();

    // compute DTX30
    std::vector<double> dt30;
    int myResult = computeAllDtxPeriod(db, idCase, 30, dt30, projectError);
    if (myResult != CRIT3D_OK)
    {
        return myResult;
    }
    // compute DTX90
    std::vector<double> dt90;
    myResult = computeAllDtxPeriod(db, idCase, 90, dt90, projectError);
    if (myResult != CRIT3D_OK)
    {
        return myResult;
    }
    // compute DTX180
    std::vector<double> dt180;
    myResult = computeAllDtxPeriod(db, idCase, 180, dt180, projectError);
    if (myResult != CRIT3D_OK)
    {
        return myResult;
    }

    // write data
    if (dt30.size() > 0)
    {
        if (! writeDtxToDB(db, idCase, dt30, dt90, dt180, projectError))
        {
            return ERROR_TDXWRITE;
        }
    }

    return CRIT3D_OK;
}


int computeAllDtxPeriod(QSqlDatabase db, QString idCase, unsigned int period, std::vector<double>& dtx, QString& projectError)
{
    // read all data
    QSqlQuery qry(db);
    QString statement = QString("SELECT TRANSP_MAX, TRANSP FROM `%1`").arg(idCase);

    // error check
    if(!qry.exec(statement))
    {
        projectError = qry.lastError().text();
        return ERROR_OUTPUT_VARIABLES;
    }
    qry.first();
    if (!qry.isValid())
    {
        projectError = qry.lastError().text();
        return ERROR_OUTPUT_VARIABLES ;
    }

    // compute daily tranpiration deficit
    std::vector<double> dailyDt;
    double transpMax, transpReal;
    do
    {
        getValue(qry.value("TRANSP_MAX"), &transpMax);
        getValue(qry.value("TRANSP"), &transpReal);

        if (transpMax != NODATA && transpReal != NODATA)
        {
            dailyDt.push_back(transpMax - transpReal);
        }
        else
        {
            dailyDt.push_back(NODATA);
        }
    }
    while (qry.next());
    qry.clear();

    // compute DTX
    // it assumes that data are complete (no missing dates)
    dtx.resize(dailyDt.size());
    for (unsigned long i = 0; i < dtx.size(); i++)
    {
        if (i < period-1)
        {
            dtx[i] = NODATA;
        }
        else
        {
            dtx[i] = 0;
            unsigned j = 0;
            while (j < period && dailyDt[i-j] != NODATA)
            {
                dtx[i] += dailyDt[i-j];
                j++;
            }

            if (j < period && dailyDt[i-j] == NODATA)
            {
                dtx[i] = NODATA;
            }
        }
    }
    dailyDt.clear();

    return CRIT3D_OK;
}


QString getNumberStr(double value)
{
    if (value == NODATA)
    {
        return QString::number(NODATA);
    }
    else
    {
        return QString::number(value,'f',1);
    }
}


bool writeDtxToDB(QSqlDatabase db, QString idCase, std::vector<double>& dt30,
                  std::vector<double>& dt90, std::vector<double>& dt180, QString& projectError)
{
    QSqlQuery qry(db);
    qry.prepare("SELECT * FROM " + idCase);
    if( !qry.exec())
    {
        projectError = "DB error: " + qry.lastError().text();
        return false;
    }
    if (!qry.first())
    {
        // table void
        return true;
    }

    int nrColumns = qry.record().count();
    QString insertQuery = "INSERT INTO " + idCase + " VALUES ";

    unsigned int index = 0;
    do
    {
        insertQuery += "(";
        for (int i = 0; i < nrColumns; i++)
        {
            if (i < nrColumns-3)
            {
                insertQuery += "'" + qry.value(i).toString() + "'";
            }
            else if (i == nrColumns-3)
            {
                insertQuery += "'" + getNumberStr(dt30[index]) + "'";
            }
            else if (i == nrColumns-2)
            {
                insertQuery += "'" + getNumberStr(dt90[index]) + "'";
            }
            else if (i == nrColumns-1)
            {
                insertQuery += "'" + getNumberStr(dt180[index]) + "'";
            }
            if (i < nrColumns - 1)
            {
                insertQuery += ",";
            }
        }
        insertQuery += ")";

        if (index < dt30.size()-1)
        {
            insertQuery += ",";
        }
        index++;
    }
    while (qry.next());

    qry.clear();

    if( !qry.exec("DELETE FROM " + idCase))
    {
        projectError = "DELETE error: " + qry.lastError().text();
        return false;
    }

    if( !qry.exec(insertQuery))
    {
        projectError = "INSERT error: " + qry.lastError().text();
        return false;
    }

    qry.clear();
    insertQuery.clear();

    return true;
}


int writeCsvOutputUnit(QString idCase, QString idCropClass, QSqlDatabase dbData, QSqlDatabase dbCrop, QSqlDatabase dbDataHistorical,
                       QDate dateComputation, CriteriaOutputVariable outputVariable, QString csvFileName, QString* projectError)
{
    // IRRI RATIO (parameter for elaboration on IRRIGATION variable)
    float irriRatio = getIrriRatioFromClass(&(dbCrop), "crop_class", "id_class", idCropClass, projectError);

    QStringList results;
    QString statement;
    QDate firstDate, lastDate;
    QVector<float> resVector;
    float res = NODATA;
    int periodTDX = NODATA;
    QSqlQuery qry(dbData);

    // check if table exist (skip otherwise)
    if (! dbData.tables().contains(idCase))
    {
        return CRIT3D_OK;
    }

    for (int i = 0; i < outputVariable.varName.size(); i++)
    {
        resVector.clear();
        QString varName = outputVariable.varName[i];
        QString computation = outputVariable.computation[i];
        if (!computation.isEmpty())
        {
            // nrDays is required, because the computation should be done between values into interval referenceDate+-nrDays
            if (outputVariable.nrDays[i].isEmpty())
            {
                // write NODATA
                res = NODATA;
                results.append(QString::number(res));
                continue;
            }
            else
            {
                // nrDays can be a number to add or subtract to referenceDate, otherwise can be a starting date (es. format YYYY-01-01)
                // the interval goes from starting date to referenceDate (dateComputation +- referenceDay)
                if (outputVariable.nrDays[i].left(4) == "YYYY")
                {
                    // outputVariable.nrDays is a starting point
                    lastDate = dateComputation.addDays(outputVariable.referenceDay[i]);
                    QString tmp = outputVariable.nrDays[i];
                    tmp.replace("YYYY",QString::number(lastDate.year()));
                    firstDate = QDate::fromString(tmp, "yyyy-MM-dd");
                    if (lastDate<firstDate)
                    {
                        firstDate.setDate(firstDate.year()-1,firstDate.month(),firstDate.day());
                    }
                }
                else
                {
                    // outputVariable.nrDays should be added or subtracted to referenceDate
                    // (given by dateComputation +- referenceDay)
                    bool ok;
                    int nrDays = outputVariable.nrDays[i].toInt(&ok, 10);
                    if (!ok)
                    {
                        *projectError = "Parser CSV error";
                        return ERROR_PARSERCSV;
                    }
                    if (nrDays == 0)
                    {
                        firstDate = dateComputation.addDays(outputVariable.referenceDay[i]);
                        lastDate = firstDate;
                    }
                    else
                    {
                        if (nrDays < 0)
                        {
                            lastDate = dateComputation.addDays(outputVariable.referenceDay[i]);
                            firstDate = lastDate.addDays(nrDays+1);
                        }
                        else
                        {
                            firstDate = dateComputation.addDays(outputVariable.referenceDay[i]);
                            lastDate = firstDate.addDays(nrDays-1);
                        }
                    }
                }
            }
        }
        // computation is empty, there is no interval but a single date, firstDate = lastDate
        else
        {
            firstDate = dateComputation.addDays(outputVariable.referenceDay[i]);
            lastDate = firstDate;
        }

        // QUERY
        // All cases except DTX
        if (varName.left(2) != "DT")
        {
            int selectRes = selectSimpleVar(dbData, idCase, varName, computation, firstDate, lastDate, irriRatio, &resVector, projectError);
            if (selectRes == ERROR_INCOMPLETE_DATA)
            {
                res = NODATA;
            }
            else if(selectRes != CRIT3D_OK)
            {
                return selectRes;
            }
            else
            {
                res = resVector[0];
            }
        }
        else
        {
            // DTX case
            bool ok;
            periodTDX = varName.right(varName.size()-2).toInt(&ok, 10);
            if (!ok)
            {
                *projectError = "Parser CSV error";
                return ERROR_PARSERCSV;
            }
            int DTXRes = computeDTX(dbData, idCase, periodTDX, computation, firstDate, lastDate, &resVector, projectError);
            // check errors in computeDTX
            if (DTXRes == ERROR_INCOMPLETE_DATA)
            {
                res = NODATA;
            }
            else if (DTXRes != CRIT3D_OK)
            {
                return DTXRes;
            }
            else
            {
                res = resVector[0];
            }
        }

        if (res == NODATA)
        {
            results.append(QString::number(res));
        }
        else
        {
            // there is no climate computation
            if (outputVariable.climateComputation[i].isEmpty())
            {
                // fraction of available water [0-1] 3 decimal digits
                if (outputVariable.varName[i] == "FRACTION_AW")
                {
                    results.append(QString::number(res,'f',3));
                }
                else
                {
                    results.append(QString::number(res,'f',1));
                }
            }
            else
            {
                // first parameter for  historical analysis (threshold)
                if (outputVariable.param1[i] != NODATA && res < outputVariable.param1[i])
                {
                    // skip historical analysis
                    results.append(QString::number(NODATA));
                }
                else
                {
                    // find historical period available
                    QDate historicalFirstDate;
                    QDate historicalLastDate;
                    QSqlQuery qry(dbDataHistorical);
                    statement = QString("SELECT MIN(DATE),MAX(DATE) FROM `%1`").arg(idCase);
                    if( !qry.exec(statement) )
                    {
                        *projectError = qry.lastError().text();
                        return ERROR_DBHISTORICAL;
                    }
                    qry.first();
                    if (!qry.isValid())
                    {
                        *projectError = qry.lastError().text();
                        return ERROR_DBHISTORICAL ;
                    }
                    getValue(qry.value("MIN(DATE)"), &historicalFirstDate);
                    getValue(qry.value("MAX(DATE)"), &historicalLastDate);

                    if (!historicalFirstDate.isValid() || !historicalLastDate.isValid())
                    {
                        // incomplete data, there is not historical period to analyze
                        results.append(QString::number(NODATA));
                    }
                    else
                    {
                        QVector<float> resAllYearsVector;
                        // second parameter for  historical analysis (timewindow)
                        if (outputVariable.param2[i] != NODATA)
                        {
                            // historical period to compare, if outputVariable.param2[i] is empty, current value should be compare
                            // with previous value in the same day (firstDate = lastDate for all the year available into DB)
                            firstDate = firstDate.addDays(-outputVariable.param2[i]);
                            lastDate = lastDate.addDays(outputVariable.param2[i]);
                        }

                        int year = historicalFirstDate.year();
                        bool skip = false;
                        while(year <= historicalLastDate.year())
                        {
                            resVector.clear();
                            firstDate.setDate(year,firstDate.month(),firstDate.day());
                            lastDate.setDate(year,lastDate.month(),lastDate.day());
                            int selectRes;

                            selectRes = selectSimpleVar(dbDataHistorical, idCase, varName, computation, firstDate, lastDate, irriRatio, &resVector, projectError);
                            if (selectRes == ERROR_INCOMPLETE_DATA)
                            {
                                // only first year can be incomplete, otherwise the comparison is not valid and can be terminated
                                if (year != historicalFirstDate.year())
                                {
                                    res = NODATA;
                                    skip = true;
                                    break;
                                }
                            }

                            if (selectRes != CRIT3D_OK && selectRes != ERROR_INCOMPLETE_DATA)
                            {
                                // something wrong happened (if ERROR_INCOMPLETE_DATA res is NODATA)
                                return selectRes;
                            }
                            else
                            {
                                resAllYearsVector.append(resVector);
                            }
                            year = year+1;
                        }
                        resVector.clear();
                        if (skip)
                        {
                            // incomplete data
                            results.append(QString::number(NODATA));
                        }
                        else
                        {
                            // comparison between current value (res) and historical values during timewindow (resAllYearsVector)
                            if (outputVariable.climateComputation[i] == "PERCENTILE")
                            {
                                // compute percentile
                                bool sortValues = true;
                                std::vector<float> historicalVector = resAllYearsVector.toStdVector();
                                res = sorting::percentileRank(historicalVector, res, sortValues);
                                results.append(QString::number(res,'f',1));
                            }
                        }
                    }
                }
            }
        }
    }

    // write CSV
    QFile outputFile;
    outputFile.setFileName(csvFileName);
    if (!outputFile.open(QIODevice::ReadWrite | QIODevice::Append))
    {
        *projectError = "Open failure: " + csvFileName;
        return false;
    }
    QTextStream out(&outputFile);
    out << dateComputation.toString("yyyy-MM-dd");
    out << "," << idCase;
    out << "," << getCropFromClass(&(dbCrop), "crop_class", "id_class", idCropClass, projectError).toUpper();
    out << "," << results.join(",");
    out << "\n";

    outputFile.flush();

    return CRIT3D_OK;
}


int selectSimpleVar(QSqlDatabase db, QString idCase, QString varName, QString computation, QDate firstDate, QDate lastDate, float irriRatio, QVector<float>* resVector, QString* projectError)
{

    QSqlQuery qry(db);
    int count = 0;
    QString statement;
    float result = NODATA;

    if (computation != "")
    {
        statement = QString("SELECT COUNT(`%1`) FROM `%2` WHERE DATE >= '%3' AND DATE <= '%4'").arg(varName).arg(idCase).arg(firstDate.toString("yyyy-MM-dd")).arg(lastDate.toString("yyyy-MM-dd"));
        if( !qry.exec(statement) )
        {
            *projectError = "Wrong variable: " + varName + "\n" + qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES;
        }
        qry.first();
        if (!qry.isValid())
        {
            *projectError = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES ;
        }
        getValue(qry.value(0), &count);
        if (count < firstDate.daysTo(lastDate)+1)
        {
            return ERROR_INCOMPLETE_DATA;
        }
    }

    count = 0;
    statement = QString("SELECT %1(`%2`) FROM `%3` WHERE DATE >= '%4' AND DATE <= '%5'").arg(computation).arg(varName).arg(idCase).arg(firstDate.toString("yyyy-MM-dd")).arg(lastDate.toString("yyyy-MM-dd"));
    if( !qry.exec(statement) )
    {
        if (varName.left(2) == "DT")
        {
            if (qry.lastError().text().contains("no such column"))
            {
                *projectError = "Precompute DTX before: " + computation + "\n" + qry.lastError().text();
                return ERROR_MISSING_PRECOMPUTE_DTX ;
            }
        }
        *projectError = "Wrong computation: " + computation + "\n" + qry.lastError().text();
        return ERROR_OUTPUT_VARIABLES ;
    }
    qry.first();
    if (!qry.isValid())
    {
        *projectError = "Missing data: " + statement;
        return ERROR_MISSING_DATA ;
    }
    do
    {
        getValue(qry.value(0), &result);
        count = count+1;
        if (varName == "IRRIGATION")
        {
            result = result * irriRatio;
        }
        resVector->push_back(result);

    }
    while(qry.next());

    // check for simple queries
    if (computation == "")
    {
        if (count < firstDate.daysTo(lastDate)+1)
        {
            *projectError = "Incomplete data: " + statement;
            return ERROR_INCOMPLETE_DATA;
        }
    }

    return CRIT3D_OK;

}

int computeDTX(QSqlDatabase db, QString idCase, int period, QString computation, QDate firstDate, QDate lastDate, QVector<float>* resVector, QString *projectError)
{

    QSqlQuery qry(db);
    QString statement;
    float res = NODATA;
    QVector<float> dtx;
    int count = 0;
    int count2 = 0;
    float var1, var2;
    QDate end = firstDate;
    QDate start;
    while (end <= lastDate)
    {
        start = end.addDays(-period+1);
        statement = QString("SELECT COUNT(TRANSP_MAX),COUNT(TRANSP) FROM `%1` WHERE DATE >= '%2' AND DATE <= '%3'").arg(idCase).arg(start.toString("yyyy-MM-dd")).arg(end.toString("yyyy-MM-dd"));
        if( !qry.exec(statement) )
        {
            *projectError = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES;
        }
        qry.first();
        if (!qry.isValid())
        {
            *projectError = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES ;
        }
        getValue(qry.value(0), &count);
        getValue(qry.value(1), &count2);
        if (count+count2 < period*2)
        {
            dtx.push_back(NODATA);
            return ERROR_INCOMPLETE_DATA;
        }
        statement = QString("SELECT SUM(TRANSP_MAX),SUM(TRANSP) FROM `%1` WHERE DATE >= '%2' AND DATE <= '%3'").arg(idCase).arg(start.toString("yyyy-MM-dd")).arg(end.toString("yyyy-MM-dd"));
        if( !qry.exec(statement) )
        {
            *projectError = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES ;
        }
        qry.first();
        if (!qry.isValid())
        {
            *projectError = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES ;
        }
        getValue(qry.value("SUM(TRANSP_MAX)"), &var1);
        getValue(qry.value("SUM(TRANSP)"), &var2);
        dtx.push_back((var1 - var2));
        end = end.addDays(1);
    }
    if (computation.isEmpty())
    {
        resVector->append(dtx);
        return CRIT3D_OK;
    }
    else if (computation == "SUM")
    {
        res = 0;
        for(int i=0; i<dtx.size();i++)
        {
            res = res + dtx[i];
        }
    }
    else if (computation == "AVG")
    {
        res = 0;
        for(int i=0; i<dtx.size();i++)
        {
            res = res + dtx[i];
        }
        res = res/dtx.size();
    }
    else if (computation == "MAX")
    {
        res = *std::max_element(dtx.begin(), dtx.end());
    }
    else if (computation == "MIN")
    {
        res = *std::min_element(dtx.begin(), dtx.end());
    }

    resVector->push_back(res);
    return CRIT3D_OK;
}


int writeCsvAggrFromShape(Crit3DShapeHandler &refShapeFile, QString csvFileName, QDate dateComputation, QStringList outputVarName, QString shapeField, QString &error)
{
    QList<QStringList> valuesFromShape;
    // write CSV
    QFile outputFile;
    outputFile.setFileName(csvFileName);
    if (!outputFile.open(QIODevice::ReadWrite | QIODevice::Truncate))
    {
        error = "Open failure: " + csvFileName;
        return ERROR_WRITECSV;
    }

    int nrRefShapes = refShapeFile.getShapeCount();
    std::string shapeFieldStdString = shapeField.toStdString();

    QStringList values;
    QStringList shapeFieldList;
    int fieldIndex = -1;
    for (int row = 0; row < nrRefShapes; row++)
    {
        // read shapeField
        fieldIndex = refShapeFile.getDBFFieldIndex(shapeFieldStdString.c_str());
        if (fieldIndex == -1)
        {
            error = QString::fromStdString(refShapeFile.getFilepath()) + "has not field called " + shapeField;
            return ERROR_SHAPEFILE;
        }
        DBFFieldType fieldType = refShapeFile.getFieldType(fieldIndex);
        if (fieldType == FTInteger)
        {
            shapeFieldList.push_back(QString::number(refShapeFile.readIntAttribute(row,fieldIndex)));
        }
        else if (fieldType == FTDouble)
        {
            shapeFieldList.push_back(QString::number(refShapeFile.readDoubleAttribute(row,fieldIndex)));
        }
        else if (fieldType == FTString)
        {
            shapeFieldList.push_back(QString::fromStdString(refShapeFile.readStringAttribute(row,fieldIndex)));
        }
        // read outputVarName
        values.clear();
        for (int field = 0; field < outputVarName.size(); field++)
        {
            std::string valField = outputVarName[field].toStdString();
            fieldIndex = refShapeFile.getDBFFieldIndex(valField.c_str());
            if (fieldIndex == -1)
            {
                error = QString::fromStdString(refShapeFile.getFilepath()) + "has not field called " + outputVarName[field];
                return ERROR_SHAPEFILE;
            }
            DBFFieldType fieldType = refShapeFile.getFieldType(fieldIndex);
            if (fieldType == FTInteger)
            {
                values.push_back(QString::number(refShapeFile.readIntAttribute(row,fieldIndex)));
            }
            else if (fieldType == FTDouble)
            {
                values.push_back(QString::number(refShapeFile.readDoubleAttribute(row,fieldIndex)));
            }
            else if (fieldType == FTString)
            {
                values.push_back(QString::fromStdString(refShapeFile.readStringAttribute(row,fieldIndex)));
            }
        }
        valuesFromShape.push_back(values);
    }

    QString header = "DATE,ZONE ID," + outputVarName.join(",");
    QTextStream out(&outputFile);
    out << header << "\n";

    for (int row = 0; row < nrRefShapes; row++)
    {
        out << dateComputation.toString("yyyy-MM-dd");
        out << "," << shapeFieldList[row];
        out << "," << valuesFromShape[row].join(",");
        out << "\n";
    }

    outputFile.flush();

    return CRIT3D_OK;
}
