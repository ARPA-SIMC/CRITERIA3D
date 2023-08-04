#include "criteriaOutputElaboration.h"
#include "criteriaOutputProject.h"
#include "commonConstants.h"
#include "basicMath.h"
#include "utilities.h"
#include "cropDbQuery.h"

#include <QSqlQuery>
#include <QSqlError>
#include <QSqlRecord>
#include <QTextStream>

int computeAllDtxUnit(QSqlDatabase db, QString idCase, QString &error)
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
        error = qry.lastError().text();
        return ERROR_DBHISTORICAL;
    }
    qry.first();
    if (!qry.isValid())
    {
        error = qry.lastError().text();
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
            error = qry.lastError().text();
            return ERROR_DBHISTORICAL;
        }
    }
    if (insertTD90Col)
    {
        statement = QString("ALTER TABLE `%1` ADD COLUMN DT90 REAL").arg(idCase);
        if( !qry.exec(statement) )
        {
            error = qry.lastError().text();
            return ERROR_DBHISTORICAL;
        }
    }
    if (insertTD180Col)
    {
        statement = QString("ALTER TABLE `%1` ADD COLUMN DT180 REAL").arg(idCase);
        if( !qry.exec(statement) )
        {
            error = qry.lastError().text();
            return ERROR_DBHISTORICAL;
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
    int myResult = computeAllDtxPeriod(db, idCase, 30, dt30, error);
    if (myResult != CRIT1D_OK)
    {
        return myResult;
    }
    // compute DTX90
    std::vector<double> dt90;
    myResult = computeAllDtxPeriod(db, idCase, 90, dt90, error);
    if (myResult != CRIT1D_OK)
    {
        return myResult;
    }
    // compute DTX180
    std::vector<double> dt180;
    myResult = computeAllDtxPeriod(db, idCase, 180, dt180, error);
    if (myResult != CRIT1D_OK)
    {
        return myResult;
    }

    // write data
    if (dt30.size() > 0)
    {
        if (! writeDtxToDB(db, idCase, dt30, dt90, dt180, error))
        {
            return ERROR_TDXWRITE;
        }
    }

    return CRIT1D_OK;
}


int computeAllDtxPeriod(QSqlDatabase db, QString idCase, unsigned int period, std::vector<double>& dtx, QString& error)
{
    // read all data
    QSqlQuery qry(db);
    QString statement = QString("SELECT TRANSP_MAX, TRANSP FROM `%1`").arg(idCase);

    // error check
    if(!qry.exec(statement))
    {
        error = qry.lastError().text();
        return ERROR_OUTPUT_VARIABLES;
    }
    qry.first();
    if (!qry.isValid())
    {
        error = qry.lastError().text();
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
                  std::vector<double>& dt90, std::vector<double>& dt180, QString& error)
{
    QSqlQuery qry(db);
    qry.prepare("SELECT * FROM " + idCase);
    if( !qry.exec())
    {
        error = "DB error: " + qry.lastError().text();
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
        error = "DELETE error: " + qry.lastError().text();
        return false;
    }

    if( !qry.exec(insertQuery))
    {
        error = "INSERT error: " + qry.lastError().text();
        return false;
    }

    qry.clear();
    insertQuery.clear();

    return true;
}


int writeCsvOutputUnit(QString idCase, QString idCropClass, QSqlDatabase& dbData, QSqlDatabase& dbCrop, QSqlDatabase& dbHistoricalData,
                       QDate dateComputation, CriteriaOutputVariable outputVariable, QString csvFileName, QString &error)
{
    // IRRI RATIO (parameter for elaboration on IRRIGATION variable)
    float irriRatio = getIrriRatioFromCropClass(dbCrop, "crop_class", "id_class", idCropClass, error);

    QList<QString> results;
    QString statement;
    QDate firstDate, lastDate;
    std::vector<float> resVector;
    double res = NODATA;
    int periodTDX = NODATA;
    QSqlQuery qry(dbData);

    // check if table exist (skip otherwise)
    if (! dbData.tables().contains(idCase))
    {
        return CRIT1D_OK;
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
                        error = "Parser CSV error";
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
            int selectRes = selectSimpleVar(dbData, idCase, varName, computation, firstDate, lastDate, irriRatio, resVector, error);
            if (selectRes == ERROR_DB_INCOMPLETE_DATA)
            {
                res = NODATA;
            }
            else if(selectRes != CRIT1D_OK)
            {
                return selectRes;
            }
            else
            {
                res = double(resVector[0]);
            }
        }
        else
        {
            // DTX case
            bool ok;
            periodTDX = varName.right(varName.size()-2).toInt(&ok, 10);
            if (!ok)
            {
                error = "Parser CSV error";
                return ERROR_PARSERCSV;
            }
            int DTXRes = computeDTX(dbData, idCase, periodTDX, computation, firstDate, lastDate, resVector, error);
            // check errors in computeDTX
            if (DTXRes == ERROR_DB_INCOMPLETE_DATA)
            {
                res = NODATA;
            }
            else if (DTXRes != CRIT1D_OK)
            {
                return DTXRes;
            }
            else
            {
                res = double(resVector[0]);
            }
        }

        if (int(res) == int(NODATA))
        {
            results.append(QString::number(res));
        }
        else
        {
            // there is no climate computation
            if (outputVariable.climateComputation[i].isEmpty())
            {
                // fraction of available water [0-1] requires 3 decimal digits
                QString varName = outputVariable.varName[i];
                if (varName == "FRACTION_AW" || varName.left(3) == "FAW")
                {
                    results.append(QString::number(res,'f', 3));
                }
                else
                {
                    results.append(QString::number(res,'f', 1));
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
                    QSqlQuery qry(dbHistoricalData);
                    statement = QString("SELECT MIN(DATE),MAX(DATE) FROM `%1`").arg(idCase);
                    if( !qry.exec(statement) )
                    {
                        error = qry.lastError().text();
                        return ERROR_DBHISTORICAL;
                    }
                    qry.first();
                    if (!qry.isValid())
                    {
                        error = qry.lastError().text();
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
                        std::vector<float> allYearsVector;
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

                            selectRes = selectSimpleVar(dbHistoricalData, idCase, varName, computation, firstDate, lastDate, irriRatio, resVector, error);
                            if (selectRes == ERROR_DB_INCOMPLETE_DATA)
                            {
                                // only first year can be incomplete, otherwise the comparison is not valid and can be terminated
                                if (year != historicalFirstDate.year())
                                {
                                    res = NODATA;
                                    skip = true;
                                    break;
                                }
                            }

                            if (selectRes != CRIT1D_OK && selectRes != ERROR_DB_INCOMPLETE_DATA)
                            {
                                // something wrong happened (if ERROR_DB_INCOMPLETE_DATA res is NODATA)
                                return selectRes;
                            }
                            else
                            {
                                allYearsVector.insert(std::end(allYearsVector), std::begin(resVector), std::end(resVector));
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
                                res = double(sorting::percentileRank(allYearsVector, float(res), sortValues));
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
        error = "Open failure: " + csvFileName;
        return false;
    }
    QTextStream out(&outputFile);
    out << dateComputation.toString("yyyy-MM-dd");
    out << "," << idCase;
    out << "," << getIdCropFromClass(dbCrop, "crop_class", "id_class", idCropClass, error).toUpper();
    out << "," << results.join(",");
    out << "\n";

    outputFile.flush();

    return CRIT1D_OK;
}


int selectSimpleVar(QSqlDatabase& db, QString idCase, QString varName, QString computation, QDate firstDate, QDate lastDate, float irriRatio, std::vector<float>& resVector, QString& error)
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
            error = "Wrong variable: " + varName + "\n" + qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES;
        }
        qry.first();
        if (!qry.isValid())
        {
            error = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES ;
        }
        getValue(qry.value(0), &count);
        if (count < firstDate.daysTo(lastDate)+1)
        {
            return ERROR_DB_INCOMPLETE_DATA;
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
                error = "Precompute DTX before: " + computation + "\n" + qry.lastError().text();
                return ERROR_DB_MISSING_PRECOMPUTED_DTX ;
            }
        }
        error = "Wrong computation: " + computation + "\n" + qry.lastError().text();
        return ERROR_OUTPUT_VARIABLES ;
    }
    qry.first();
    if (!qry.isValid())
    {
        error = "Missing data: " + statement;
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
        resVector.push_back(result);

    }
    while(qry.next());

    // check for simple queries
    if (computation == "")
    {
        if (count < firstDate.daysTo(lastDate)+1)
        {
            error = "Incomplete data: " + statement;
            return ERROR_DB_INCOMPLETE_DATA;
        }
    }

    return CRIT1D_OK;

}

int computeDTX(QSqlDatabase &db, QString idCase, int period, QString computation, QDate firstDate, QDate lastDate, std::vector<float>& resVector, QString &Error)
{
    QSqlQuery qry(db);
    QString statement;
    double res = NODATA;
    std::vector<float> dtx;
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
            Error = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES;
        }
        qry.first();
        if (!qry.isValid())
        {
            Error = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES ;
        }
        getValue(qry.value(0), &count);
        getValue(qry.value(1), &count2);
        if (count+count2 < period*2)
        {
            dtx.push_back(NODATA);
            return ERROR_DB_INCOMPLETE_DATA;
        }
        statement = QString("SELECT SUM(TRANSP_MAX),SUM(TRANSP) FROM `%1` WHERE DATE >= '%2' AND DATE <= '%3'").arg(idCase).arg(start.toString("yyyy-MM-dd")).arg(end.toString("yyyy-MM-dd"));
        if( !qry.exec(statement) )
        {
            Error = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES ;
        }
        qry.first();
        if (!qry.isValid())
        {
            Error = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES ;
        }
        getValue(qry.value("SUM(TRANSP_MAX)"), &var1);
        getValue(qry.value("SUM(TRANSP)"), &var2);
        dtx.push_back((var1 - var2));
        end = end.addDays(1);
    }
    if (computation.isEmpty())
    {
        resVector.insert(std::end(resVector), std::begin(dtx), std::end(dtx));
        return CRIT1D_OK;
    }
    else if (computation == "SUM")
    {
        res = 0;
        for(unsigned int i=0; i < dtx.size(); i++)
        {
            res = res + double(dtx[i]);
        }
    }
    else if (computation == "AVG")
    {
        res = 0;
        for(unsigned int i=0; i < dtx.size(); i++)
        {
            res = res + double(dtx[i]);
        }
        res = res/double(dtx.size());
    }
    else if (computation == "MAX")
    {
        res = double(*std::max_element(dtx.begin(), dtx.end()));
    }
    else if (computation == "MIN")
    {
        res = double(*std::min_element(dtx.begin(), dtx.end()));
    }

    resVector.push_back(float(res));
    return CRIT1D_OK;
}


int writeCsvAggrFromShape(Crit3DShapeHandler &refShapeFile, QString csvFileName, QDate dateComputation, QList<QString> outputVarName, QString shapeField, QString &error)
{
    QList<QList<QString>> valuesFromShape;
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

    QList<QString> values;
    QList<QString> shapeFieldList;
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
            shapeFieldList.push_back(QString::number(refShapeFile.readDoubleAttribute(row,fieldIndex),'f',1));
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

    return CRIT1D_OK;
}


int orderCsvByField(QString csvFileName, QString field, QString &error)
{
    QFile fileCsv;
    fileCsv.setFileName(csvFileName);
    if (!fileCsv.open(QIODevice::ReadWrite))
    {
        error = "Open failure: " + csvFileName;
        return ERROR_WRITECSV;
    }

    QTextStream in(&fileCsv);
    //skip header
    QString line = in.readLine();
    QList<QString> header = line.split(",");  // save header
    int pos = int(header.indexOf(field));   // save field to order position
    if (pos == -1)
    {
        error = "missing field";
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
