#include "criteriaOutputElaboration.h"
#include "criteriaOutputProject.h"
#include "commonConstants.h"
#include "basicMath.h"
#include "utilities.h"
#include "../crop/cropDbQuery.h"

#include <QSqlQuery>
#include <QSqlError>
#include <QSqlRecord>
#include <QTextStream>

int computeAllDtxUnit(QSqlDatabase db, QString idCase, QString &errorStr)
{
    // check if table exist (skip otherwise)
    if (! db.tables().contains(idCase))
    {
        return CRIT1D_OK;
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
        errorStr = qry.lastError().text();
        return ERROR_DBCLIMATE;
    }
    qry.first();
    if (!qry.isValid())
    {
        errorStr = qry.lastError().text();
        return ERROR_DBCLIMATE ;
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
            errorStr = qry.lastError().text();
            return ERROR_DBCLIMATE;
        }
    }
    if (insertTD90Col)
    {
        statement = QString("ALTER TABLE `%1` ADD COLUMN DT90 REAL").arg(idCase);
        if( !qry.exec(statement) )
        {
            errorStr = qry.lastError().text();
            return ERROR_DBCLIMATE;
        }
    }
    if (insertTD180Col)
    {
        statement = QString("ALTER TABLE `%1` ADD COLUMN DT180 REAL").arg(idCase);
        if( !qry.exec(statement) )
        {
            errorStr = qry.lastError().text();
            return ERROR_DBCLIMATE;
        }
    }

    // check if table is full (skip otherwise)
    qry.prepare("SELECT * FROM " + idCase);
    qry.exec();
    if (!qry.first())
    {
        return CRIT1D_OK;
    }
    qry.clear();

    // compute DTX30
    std::vector<double> dt30;
    int myResult = computeAllDtxPeriod(db, idCase, 30, dt30, errorStr);
    if (myResult != CRIT1D_OK)
    {
        return myResult;
    }
    // compute DTX90
    std::vector<double> dt90;
    myResult = computeAllDtxPeriod(db, idCase, 90, dt90, errorStr);
    if (myResult != CRIT1D_OK)
    {
        return myResult;
    }
    // compute DTX180
    std::vector<double> dt180;
    myResult = computeAllDtxPeriod(db, idCase, 180, dt180, errorStr);
    if (myResult != CRIT1D_OK)
    {
        return myResult;
    }

    // write data
    if (dt30.size() > 0)
    {
        if (! writeDtxToDB(db, idCase, dt30, dt90, dt180, errorStr))
        {
            return ERROR_TDXWRITE;
        }
    }

    return CRIT1D_OK;
}


int computeAllDtxPeriod(QSqlDatabase db, QString idCase, unsigned int period, std::vector<double>& dtx, QString& errorStr)
{
    // read all data
    QSqlQuery qry(db);
    QString statement = QString("SELECT TRANSP_MAX, TRANSP FROM `%1`").arg(idCase);

    // errorStr check
    if(!qry.exec(statement))
    {
        errorStr = qry.lastError().text();
        return ERROR_OUTPUT_VARIABLES;
    }
    qry.first();
    if (!qry.isValid())
    {
        errorStr = qry.lastError().text();
        return ERROR_OUTPUT_VARIABLES ;
    }

    // compute daily tranpiration deficit
    std::vector<double> dailyDt;
    double transpMax, transpReal;
    do
    {
        getValue(qry.value("TRANSP_MAX"), &transpMax);
        getValue(qry.value("TRANSP"), &transpReal);

        if ((int(transpMax) != int(NODATA)) && (int(transpReal) != int(NODATA)))
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
            while (j < period && int(dailyDt[i-j]) != int(NODATA))
            {
                dtx[i] += dailyDt[i-j];
                j++;
            }

            if (j < period && int(dailyDt[i-j]) == int(NODATA))
            {
                dtx[i] = NODATA;
            }
        }
    }
    dailyDt.clear();

    return CRIT1D_OK;
}


QString getNumberStr(double value)
{
    if (int(value) == int(NODATA))
    {
        return QString::number(NODATA);
    }
    else
    {
        return QString::number(value,'f',1);
    }
}


bool writeDtxToDB(QSqlDatabase db, QString idCase, std::vector<double>& dt30,
                  std::vector<double>& dt90, std::vector<double>& dt180, QString& errorStr)
{
    QSqlQuery qry(db);
    qry.prepare("SELECT * FROM " + idCase);
    if( !qry.exec())
    {
        errorStr = "DB errorStr: " + qry.lastError().text();
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
        errorStr = "DELETE errorStr: " + qry.lastError().text();
        return false;
    }

    if( !qry.exec(insertQuery))
    {
        errorStr = "INSERT errorStr: " + qry.lastError().text();
        return false;
    }

    qry.clear();
    insertQuery.clear();

    return true;
}


int writeCsvOutputUnit(const QString &idCase, const QString &idCropClass, const QList<QString> &dataTables,
                       QSqlDatabase &dbData, QSqlDatabase &dbCrop, QSqlDatabase &dbClimateData,
                       const QDate &dateComputation, const CriteriaOutputVariable &outputVariable,
                       const QString &csvFileName, QString &errorStr)
{
    // IRRI RATIO (parameter for elaboration on IRRIGATION variable)
    float irriRatio = NODATA;
    for (int i = 0; i < outputVariable.varNameList.size(); i++)
    {
        if (outputVariable.varNameList[i].toUpper() == "IRRIGATION")
        {
            irriRatio = getIrriRatioFromCropClass(dbCrop, "crop_class", "id_class", idCropClass, errorStr);
            break;
        }
    }

    QList<QString> resultList;
    QString statement;
    QDate firstDate, lastDate;
    std::vector<float> resultVector;

    double result = NODATA;
    int periodTDX = NODATA;

    // check if table for idCase exist (skip otherwise)
    if (! dataTables.contains(idCase))
    {
        return CRIT1D_OK;
    }

    for (int i = 0; i < outputVariable.varNameList.size(); i++)
    {
        resultVector.clear();
        QString varName = outputVariable.varNameList[i];
        QString computation = outputVariable.computationList[i];
        if (! computation.isEmpty())
        {
            // nrDays is required, because the computation should be done between values into interval referenceDate+-nrDays
            if (outputVariable.nrDays[i].isEmpty())
            {
                // if nrDays is missing write NODATA
                result = NODATA;
                resultList.append(QString::number(result));
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
                    if (lastDate < firstDate)
                    {
                        firstDate.setDate(firstDate.year()-1, firstDate.month(), firstDate.day());
                    }
                }
                else
                {
                    // outputVariable.nrDays should be added or subtracted to referenceDate,
                    // given by dateComputation +- referenceDay
                    bool ok;
                    int nrDays = outputVariable.nrDays[i].toInt(&ok, 10);
                    if (!ok)
                    {
                        errorStr = "Parser CSV errorStr";
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
            int selectRes = selectSimpleVar(dbData, idCase, varName, computation, firstDate, lastDate, irriRatio, resultVector, errorStr);
            if (selectRes == ERROR_DB_INCOMPLETE_DATA)
            {
                result = NODATA;
            }
            else if(selectRes != CRIT1D_OK)
            {
                return selectRes;
            }
            else
            {
                result = double(resultVector[0]);
            }
        }
        else
        {
            // DTX case
            bool ok;
            periodTDX = varName.right(varName.size()-2).toInt(&ok, 10);
            if (! ok)
            {
                errorStr = "Parser CSV errorStr";
                return ERROR_PARSERCSV;
            }
            int DTXRes = computeDTX(dbData, idCase, periodTDX, computation, firstDate, lastDate, resultVector, errorStr);
            // check errors in computeDTX
            if (DTXRes == ERROR_DB_INCOMPLETE_DATA)
            {
                result = NODATA;
            }
            else if (DTXRes != CRIT1D_OK)
            {
                return DTXRes;
            }
            else
            {
                result = double(resultVector[0]);
            }
        }

        if (isEqual(result, NODATA))
        {
            resultList.append(QString::number(result));
        }
        else
        {
            // there is no climate computation
            if (outputVariable.climateComputation[i].isEmpty())
            {
                // fraction and index [0-1] requires 3 decimal digits
                QString varName = outputVariable.varNameList[i];
                if (varName == "FRACTION_AW" || varName.left(3) == "FAW" || varName.left(3) == "SWI")
                {
                    resultList.append(QString::number(result,'f', 3));
                }
                else
                {
                    resultList.append(QString::number(result,'f', 1));
                }
            }
            else
            {
                // first parameter for  climate analysis (threshold)
                if (outputVariable.param1[i] != NODATA && result < outputVariable.param1[i])
                {
                    // skip climate analysis
                    resultList.append(QString::number(NODATA));
                }
                else
                {
                    // find climate period available
                    QDate climateFirstDate;
                    QDate climateLastDate;
                    QSqlQuery qry(dbClimateData);
                    statement = QString("SELECT MIN(DATE),MAX(DATE) FROM `%1`").arg(idCase);
                    if( !qry.exec(statement) )
                    {
                        errorStr = "Error in query climate data";
                        return ERROR_DBCLIMATE;
                    }

                    qry.first();
                    if (!qry.isValid())
                    {
                        errorStr = "climate data: " + qry.lastError().text();
                        return ERROR_DBCLIMATE ;
                    }

                    getValue(qry.value("MIN(DATE)"), &climateFirstDate);
                    getValue(qry.value("MAX(DATE)"), &climateLastDate);

                    if (!climateFirstDate.isValid() || !climateLastDate.isValid())
                    {
                        // incomplete data, there is not climate period to analyze
                        resultList.append(QString::number(NODATA));
                    }
                    else
                    {
                        std::vector<float> allYearsVector;

                        int year = climateFirstDate.year();
                        bool skip = false;
                        while(year <= climateLastDate.year())
                        {
                            // set date
                            QDate previousFirstDate, previousLastDate;
                            previousFirstDate.setDate(year, firstDate.month(), firstDate.day());
                            previousLastDate.setDate(year, lastDate.month(), lastDate.day());
                            if (lastDate.year() == (firstDate.year()+1))
                                previousLastDate.setDate(year+1, lastDate.month(), lastDate.day());

                            // second parameter for climate analysis (timewindow)
                            // if outputVariable.param2 is empty, current value should be compare with previous value in the same day
                            if (outputVariable.param2[i] != NODATA)
                            {
                                previousFirstDate = previousFirstDate.addDays(-outputVariable.param2[i]);
                                previousLastDate = previousLastDate.addDays(outputVariable.param2[i]);
                            }

                            resultVector.clear();
                            int queryResult = selectSimpleVar(dbClimateData, idCase, varName, computation,
                                                        previousFirstDate, previousLastDate, irriRatio, resultVector, errorStr);
                            if (queryResult == ERROR_DB_INCOMPLETE_DATA)
                            {
                                // only first and last years can be incomplete, otherwise the comparison is not valid and can be terminated
                                if (year != climateFirstDate.year() && year != climateLastDate.year())
                                {
                                    skip = true;
                                    break;
                                }
                            }

                            if (queryResult != CRIT1D_OK && queryResult != ERROR_DB_INCOMPLETE_DATA)
                            {
                                // something wrong happened (if ERROR_DB_INCOMPLETE_DATA result is NODATA)
                                return queryResult;
                            }
                            else
                            {
                                allYearsVector.insert(std::end(allYearsVector), std::begin(resultVector), std::end(resultVector));
                            }
                            year = year+1;
                        }

                        resultVector.clear();
                        if (skip)
                        {
                            // incomplete data
                            resultList.append(QString::number(NODATA));
                        }
                        else
                        {
                            // comparison between current value (result) and climate values during timewindow (allYearsVector)
                            if (outputVariable.climateComputation[i] == "PERCENTILE")
                            {
                                // compute percentile
                                bool sortValues = true;
                                result = double(sorting::percentileRank(allYearsVector, float(result), sortValues));
                                resultList.append(QString::number(result,'f',1));
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
    if (! outputFile.open(QIODevice::ReadWrite | QIODevice::Append))
    {
        errorStr = "Open failure: " + csvFileName;
        return false;
    }

    QTextStream out(&outputFile);
    out << dateComputation.toString("yyyy-MM-dd");
    out << "," << idCase;
    out << "," << getIdCropFromClass(dbCrop, "crop_class", "id_class", idCropClass, errorStr).toUpper();
    out << "," << resultList.join(",");
    out << "\n";

    outputFile.flush();

    return CRIT1D_OK;
}


// TODO: possibile problema con computation != "" e valori pari a -9999
int selectSimpleVar(QSqlDatabase& db, QString idCase, QString varName, QString computation,
                    QDate firstDate, QDate lastDate, float irriRatio, std::vector<float>& resultVector, QString& errorStr)
{

    QSqlQuery qry(db);
    int count = 0;
    QString statement;
    float result = NODATA;

    // check nr of values
    if (computation != "")
    {
        statement = QString("SELECT COUNT(`%1`) FROM `%2` WHERE DATE >= '%3' AND DATE <= '%4'")
                        .arg(varName, idCase, firstDate.toString("yyyy-MM-dd"), lastDate.toString("yyyy-MM-dd"));
        if( !qry.exec(statement) )
        {
            errorStr = "Wrong variable: " + varName + "\n" + qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES;
        }

        qry.first();
        if (!qry.isValid())
        {
            errorStr = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES ;
        }

        getValue(qry.value(0), &count);
        if (count < firstDate.daysTo(lastDate)+1)
        {
            return ERROR_DB_INCOMPLETE_DATA;
        }
    }

    count = 0;
    statement = QString("SELECT %1(`%2`) FROM `%3` WHERE DATE >= '%4' AND DATE <= '%5'")
                    .arg(computation, varName, idCase, firstDate.toString("yyyy-MM-dd"), lastDate.toString("yyyy-MM-dd"));
    if( !qry.exec(statement) )
    {
        if (varName.left(2) == "DT")
        {
            if (qry.lastError().text().contains("no such column"))
            {
                errorStr = "Precompute DTX before: " + computation + "\n" + qry.lastError().text();
                return ERROR_DB_MISSING_PRECOMPUTED_DTX ;
            }
        }
        errorStr = "Wrong computation: " + computation + "\n" + qry.lastError().text();
        return ERROR_OUTPUT_VARIABLES ;
    }

    qry.first();
    if (!qry.isValid())
    {
        errorStr = "Missing data: " + statement;
        return ERROR_DB_MISSING_DATA ;
    }

    do
    {
        getValue(qry.value(0), &result);
        count = count+1;
        if (varName == "IRRIGATION")
        {
            result = result * irriRatio;
        }
        resultVector.push_back(result);
    }
    while(qry.next());

    // check for simple queries
    if (computation == "")
    {
        if (count < firstDate.daysTo(lastDate)+1)
        {
            errorStr = "Incomplete data: " + statement;
            return ERROR_DB_INCOMPLETE_DATA;
        }
    }

    return CRIT1D_OK;
}


int computeDTX(QSqlDatabase &db, QString idCase, int period, QString computation,
               QDate firstDate, QDate lastDate, std::vector<float>& resultVector, QString &errorStr)
{
    QSqlQuery qry(db);
    QString statement;
    double result = NODATA;
    std::vector<float> dtx;
    int count = 0;
    int count2 = 0;
    float var1, var2;
    QDate end = firstDate;
    QDate start;
    while (end <= lastDate)
    {
        start = end.addDays(-period+1);
        statement = QString("SELECT COUNT(TRANSP_MAX),COUNT(TRANSP) FROM `%1` "
                            "WHERE DATE >= '%2' AND DATE <= '%3'").arg(idCase, start.toString("yyyy-MM-dd"), end.toString("yyyy-MM-dd"));
        if( !qry.exec(statement) )
        {
            errorStr = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES;
        }
        qry.first();
        if (!qry.isValid())
        {
            errorStr = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES ;
        }
        getValue(qry.value(0), &count);
        getValue(qry.value(1), &count2);
        if (count+count2 < period*2)
        {
            dtx.push_back(NODATA);
            return ERROR_DB_INCOMPLETE_DATA;
        }
        statement = QString("SELECT SUM(TRANSP_MAX),SUM(TRANSP) FROM `%1` WHERE DATE >= '%2' AND DATE <= '%3'")
                        .arg(idCase, start.toString("yyyy-MM-dd"), end.toString("yyyy-MM-dd"));
        if( !qry.exec(statement) )
        {
            errorStr = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES ;
        }
        qry.first();
        if (!qry.isValid())
        {
            errorStr = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES ;
        }
        getValue(qry.value("SUM(TRANSP_MAX)"), &var1);
        getValue(qry.value("SUM(TRANSP)"), &var2);
        dtx.push_back((var1 - var2));
        end = end.addDays(1);
    }
    if (computation.isEmpty())
    {
        resultVector.insert(std::end(resultVector), std::begin(dtx), std::end(dtx));
        return CRIT1D_OK;
    }
    else if (computation == "SUM")
    {
        result = 0;
        for(unsigned int i=0; i < dtx.size(); i++)
        {
            result = result + double(dtx[i]);
        }
    }
    else if (computation == "AVG")
    {
        result = 0;
        for(unsigned int i=0; i < dtx.size(); i++)
        {
            result = result + double(dtx[i]);
        }
        result = result/double(dtx.size());
    }
    else if (computation == "MAX")
    {
        result = double(*std::max_element(dtx.begin(), dtx.end()));
    }
    else if (computation == "MIN")
    {
        result = double(*std::min_element(dtx.begin(), dtx.end()));
    }

    resultVector.push_back(float(result));
    return CRIT1D_OK;
}


int writeCsvAggrFromShape(Crit3DShapeHandler &refShapeFile, QString csvFileName,
                          QDate dateComputation, QList<QString> outputVarName, QString shapeField, QString &errorStr)
{
    QList<QList<QString>> valuesFromShape;
    // write CSV
    QFile outputFile;
    outputFile.setFileName(csvFileName);
    if (!outputFile.open(QIODevice::ReadWrite | QIODevice::Truncate))
    {
        errorStr = "Open failure: " + csvFileName;
        return ERROR_WRITECSV;
    }

    int nrRefShapes = refShapeFile.getShapeCount();
    std::string shapeFieldStdString = shapeField.toStdString();

    QList<QString> values;
    QList<QString> shapeFieldList;
    int fieldIndex = -1;
    for (int row = 0; row < nrRefShapes; row++)
    {
        // read shapeField
        fieldIndex = refShapeFile.getDBFFieldIndex(shapeFieldStdString.c_str());
        if (fieldIndex == -1)
        {
            errorStr = QString::fromStdString(refShapeFile.getFilepath()) + "has not field called " + shapeField;
            return ERROR_SHAPEFILE;
        }
        DBFFieldType fieldType = refShapeFile.getFieldType(fieldIndex);
        if (fieldType == FTInteger)
        {
            shapeFieldList.push_back(QString::number(refShapeFile.readIntAttribute(row, fieldIndex)));
        }
        else if (fieldType == FTDouble)
        {
            shapeFieldList.push_back(QString::number(refShapeFile.readDoubleAttribute(row, fieldIndex), 'f', 1));
        }
        else if (fieldType == FTString)
        {
            shapeFieldList.push_back(QString::fromStdString(refShapeFile.readStringAttribute(row, fieldIndex)));
        }

        // read outputVarName
        values.clear();
        for (int field = 0; field < outputVarName.size(); field++)
        {
            std::string valField = outputVarName[field].toStdString();
            fieldIndex = refShapeFile.getDBFFieldIndex(valField.c_str());
            if (fieldIndex == -1)
            {
                errorStr = QString::fromStdString(refShapeFile.getFilepath()) + "has not field called " + outputVarName[field];
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

    QString dateStr = dateComputation.toString("yyyy-MM-dd");
    QString header = "DATE,ZONE ID," + outputVarName.join(",");
    QTextStream out(&outputFile);
    out << header << "\n";

    for (int row = 0; row < nrRefShapes; row++)
    {
        out << dateStr;
        out << "," << shapeFieldList[row];
        out << "," << valuesFromShape[row].join(",");
        out << "\n";
    }

    outputFile.flush();
    outputFile.close();

    return CRIT1D_OK;
}


int orderCsvByField(QString csvFileName, QString field, QString &errorStr)
{
    QFile fileCsv;
    fileCsv.setFileName(csvFileName);
    if (!fileCsv.open(QIODevice::ReadWrite))
    {
        errorStr = "Open failure: " + csvFileName;
        return ERROR_WRITECSV;
    }

    QTextStream in(&fileCsv);
    //skip header
    QString line = in.readLine();
    QList<QString> header = line.split(",");  // save header
    int pos = int(header.indexOf(field));   // save field to order position
    if (pos == -1)
    {
        errorStr = "missing field";
        return false;
    }

    bool isNumeric = false;
    int countNumericKey = 0;
    QList<QString> keyList;
    QList<QList<QString>> itemsList;

    while (!in.atEnd())
    {
        line = in.readLine();
        QList<QString> items = line.split(",");
        keyList << items[pos].toUpper();
        items.removeAt(pos);
        itemsList << items;
    }

    // check if field values are all numbers
    for (int i = 0; i<keyList.size(); i++)
    {
        keyList[i].toInt(&isNumeric, 10);
        if (isNumeric)
        {
            countNumericKey = countNumericKey + 1;
        }
    }
    if (countNumericKey == keyList.size())
    {
        // field is a number
        QMap<int, QList<QString>> mapCsv;
        int key;
        for (int i = 0; i<keyList.size(); i++)
        {
            key = keyList[i].toInt();
            mapCsv[key] = itemsList[i];
        }
        // reorder csv file
        in.seek(0); //start file from the beginning
        in << header.join(",") << "\n";

        QMapIterator<int, QList<QString>> i(mapCsv);
        QList<QString> line;
        while (i.hasNext()) {
            i.next();
            line = i.value();
            line.insert(pos, QString::number(i.key()));
            in << line.join(",");
            in << "\n";
        }
    }
    else
    {
        // field is not a number
        QMap<QString, QList<QString>> mapCsv;
        for (int i = 0; i<keyList.size(); i++)
        {
            mapCsv[keyList[i]] = itemsList[i];
        }
        // reorder csv file
        in.seek(0); //start file from the beginning
        in << header.join(",") << "\n";

        QMapIterator<QString, QList<QString>> i(mapCsv);
        QList<QString> line;
        while (i.hasNext()) {
            i.next();
            line = i.value();
            line.insert(pos, i.key());
            in << line.join(",");
            in << "\n";
        }
    }

    fileCsv.flush();
    return CRIT1D_OK;
}
