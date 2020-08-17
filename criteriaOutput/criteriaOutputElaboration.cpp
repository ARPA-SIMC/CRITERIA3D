#include "criteriaOutputElaboration.h"
#include "criteriaOutputProject.h"
#include "commonConstants.h"
#include "basicMath.h"
#include "utilities.h"
#include "cropDbQuery.h"
#include <QtSql>

int writeCsvOutputUnit(QString idCase, QString idCropClass, QSqlDatabase dbData, QSqlDatabase dbCrop, QSqlDatabase dbDataHistorical,
                       QDate dateComputation, CriteriaOutputVariable outputVariable, QString csvFileName, QString* projectError)
{
    // IRRI RATIO
    //float irriRatio = getIrriRatioFromClass(&(dbCrop), "crop_class", "id_class", unitList[unitIndex].idCropClass, &(projectError));
    float irriRatio = getIrriRatioFromClass(&(dbCrop), "crop_class", "id_class", idCropClass, projectError);

    //QString idCase = unitList[unitIndex].idCase;
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

    for (int i = 0; i<outputVariable.varName.size(); i++)
    {
        resVector.clear();
        QString varName = outputVariable.varName[i];
        QString computation = outputVariable.computation[i];
        if (!computation.isEmpty())
        {
            if (outputVariable.nrDays[i].isEmpty())
            {
                // write NODATA
                res = NODATA;
                results.append(QString::number(res));
                continue;
            }
            else
            {
                if (outputVariable.nrDays[i].left(4) == "YYYY")
                {
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
        // computation is empty
        else
        {
            firstDate = dateComputation.addDays(outputVariable.referenceDay[i]);
            lastDate = firstDate;
        }

        // QUERY
        // simple variable
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
            // DTX
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
            if (outputVariable.climateComputation[i].isEmpty())
            {
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
                // db_data_historical comparison
                if (outputVariable.param1[i] != NODATA && res < outputVariable.param1[i])
                {
                    // skip historical analysis
                    results.append(QString::number(NODATA));
                }
                else
                {

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
                        // incomplete data
                        results.append(QString::number(NODATA));
                    }
                    else
                    {
                        QVector<float> resAllYearsVector;
                        if (outputVariable.param2[i] != NODATA)
                        {
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

                            if (varName.left(2) != "DT")
                            {
                                // ALL CASES
                                selectRes = selectSimpleVar(dbDataHistorical, idCase, varName, computation, firstDate, lastDate, irriRatio, &resVector, projectError);
                                if (selectRes == ERROR_INCOMPLETE_DATA)
                                {
                                    if (year != historicalFirstDate.year())
                                    {
                                        res = NODATA;
                                        skip = true;
                                        break;
                                    }
                                }
                            }
                            else
                            {
                                // TDX
                                selectRes = computeDTX(dbDataHistorical, idCase, periodTDX , computation, firstDate, lastDate, &resVector, projectError);
                                if (selectRes == ERROR_INCOMPLETE_DATA)
                                {
                                    if (year != historicalFirstDate.year())
                                    {
                                        res = NODATA;
                                        skip = true;
                                        break;
                                    }
                                }
                            }
                            if (selectRes != CRIT3D_OK && selectRes != ERROR_INCOMPLETE_DATA)
                            {
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
                            if (outputVariable.climateComputation[i] == "PERCENTILE")
                            {
                                bool sortValues = true;
                                std::vector<float> historicalVector = resAllYearsVector.toStdVector();
                                res = sorting::percentileRank(historicalVector, res, sortValues);
                                if (outputVariable.varName[i] == "FRACTION_AW")
                                {
                                    results.append(QString::number(res,'f',3));
                                }
                                else
                                {
                                    results.append(QString::number(res,'f',1));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // write CSV

    // TO DO sistemare la scrittura del file in modo che vada in append a quanto già scritto
    QFile outputFile;
    outputFile.setFileName(csvFileName);
    if (!outputFile.open(QIODevice::ReadWrite | QIODevice::Truncate))
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
    statement = QString("SELECT %1(`%2`) FROM `%3` WHERE DATE >= '%4' AND DATE <= '%5'").arg(computation).arg(varName).arg(idCase).arg(firstDate.toString("yyyy-MM-dd")).arg(lastDate.toString("yyyy-MM-dd"));
    if( !qry.exec(statement) )
    {
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


    if (count < firstDate.daysTo(lastDate)+1)
    {
        *projectError = "Incomplete data: " + statement;
        return ERROR_INCOMPLETE_DATA;
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
