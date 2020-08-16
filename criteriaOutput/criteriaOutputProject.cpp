#include "commonConstants.h"
#include "basicMath.h"
#include "criteriaOutputProject.h"
#include "logger.h"
#include "utilities.h"
#include "cropDbQuery.h"
#include "shapeHandler.h"
#include "ucmUtilities.h"

#include <QtSql>
#include <iostream>


CriteriaOutputProject::CriteriaOutputProject()
{
    initialize();
}

void CriteriaOutputProject::initialize()
{
    isProjectLoaded = false;

    projectName = "";
    dbUnitsName = "";
    dbDataName = "";
    dbDataHistoricalName = "";
    dbCropName = "";

    variableListFileName = "";
    csvFileName = "";

    ucmFileName = "";
    shapeFileName = "";
    shapeFilePath = "";
    fieldListFileName = "";

    aggregationShape = "";
    shapeFieldName = "";

    projectError = "";
    nrUnits = 0;
}

void CriteriaOutputProject::closeProject()
{
    if (isProjectLoaded)
    {
        logger.writeInfo("Close Project...");

        initialize();

        unitList.clear();
        outputFile.close();
        logFile.close();
        dbData.close();
        dbCrop.close();
        dbDataHistorical.close();

        isProjectLoaded = false;
    }
}


int CriteriaOutputProject::initializeProjectCsv()
{
    // check DB Crop
    logger.writeInfo("DB Crop: " + dbCropName);
    if (!QFile(dbCropName).exists())
    {
        projectError = "DB Crop file doesn't exist";
        return ERROR_DBPARAMETERS;
    }
    // open DB Crop
    dbCrop = QSqlDatabase::addDatabase("QSQLITE", "crop");
    dbCrop.setDatabaseName(dbCropName);
    if (! dbCrop.open())
    {
        projectError = "Open Crop DB failed: " + dbCrop.lastError().text();
        return ERROR_DBPARAMETERS;
    }

    // check DB data
    logger.writeInfo("DB Data: " + dbDataName);
    if (!QFile(dbDataName).exists())
    {
        projectError = "DB data file doesn't exist";
        return ERROR_DBPARAMETERS;
    }
    // open DB Data
    dbData = QSqlDatabase::addDatabase("QSQLITE", "data");
    dbData.setDatabaseName(dbDataName);
    if (! dbData.open())
    {
        projectError = "Open DB data failed: " + dbData.lastError().text();
        return ERROR_DBPARAMETERS;
    }

    // open DB Data Historical
    if(!dbDataHistoricalName.isEmpty())
    {
        logger.writeInfo("DB data historical: " + dbDataHistoricalName);
        if (!QFile(dbDataHistoricalName).exists())
        {
            projectError = "DB data historical doesn't exist";
            return ERROR_DBPARAMETERS;
        }

        dbDataHistorical = QSqlDatabase::addDatabase("QSQLITE", "dataHistorical");
        dbDataHistorical.setDatabaseName(dbDataHistoricalName);
        if (! dbDataHistorical.open())
        {
            projectError = "Open DB data historical failed: " + dbDataHistorical.lastError().text();
            return ERROR_DBPARAMETERS;
        }
    }

    return CRIT3D_OK;
}


int CriteriaOutputProject::initializeProject(QString settingsFileName, QDate dateComputation)
{
    closeProject();
    initialize();
    this->dateComputation = dateComputation;

    if (settingsFileName == "")
    {
        projectError = "Missing settings File.";
        return ERROR_SETTINGS_MISSING;
    }

    // Configuration file
    QFile myFile(settingsFileName);
    if (myFile.exists())
    {
        configFileName = QDir(myFile.fileName()).canonicalPath();
        configFileName = QDir().cleanPath(configFileName);

        QFileInfo fileInfo(configFileName);
        path = fileInfo.path() + "/";
    }
    else
    {
        projectError = "Cannot find settings file: " + settingsFileName;
        return ERROR_SETTINGS_WRONGFILENAME;
    }

    if (!readSettings())
    {
        projectError = "Read settings error";
        return ERROR_SETTINGS_MISSINGDATA;
    }

    logger.setLog(path,projectName);

    isProjectLoaded = true;
    return CRIT3D_OK;
}


bool CriteriaOutputProject::readSettings()
{
    QSettings* projectSettings;
    projectSettings = new QSettings(configFileName, QSettings::IniFormat);
    projectSettings->beginGroup("project");

    QString dateStr = dateComputation.toString("yyyy-MM-dd");

    projectName = projectSettings->value("name","").toString();

    // unit list
    dbUnitsName = projectSettings->value("db_units","").toString();
    if (dbUnitsName.left(1) == ".")
    {
        dbUnitsName = path + QDir::cleanPath(dbUnitsName);
    }
    if (dbUnitsName == "")
    {
        projectError = "Missing information on units";
        return false;
    }

    dbDataName = projectSettings->value("db_data","").toString();
    if (dbDataName.left(1) == ".")
    {
        dbDataName = path + QDir::cleanPath(dbDataName);
    }

    dbCropName = projectSettings->value("db_crop","").toString();
    if (dbCropName.left(1) == ".")
    {
        dbCropName = path + QDir::cleanPath(dbCropName);
    }

    dbDataHistoricalName = projectSettings->value("db_data_historical","").toString();
    if (dbDataHistoricalName.left(1) == ".")
    {
        dbDataHistoricalName = path + QDir::cleanPath(dbDataHistoricalName);
    }
    projectSettings->endGroup();

    projectSettings->beginGroup("csv");
    // variables list
    variableListFileName = projectSettings->value("variable_list","").toString();
    if (variableListFileName.left(1) == ".")
    {
        variableListFileName = path + QDir::cleanPath(variableListFileName);
    }

    bool addDate = projectSettings->value("add_date_to_filename","").toBool();

    // csv output
    csvFileName = projectSettings->value("csv_output","").toString();
    if (csvFileName.right(4) == ".csv")
    {
        csvFileName = csvFileName.left(csvFileName.length()-4);
    }
    if (addDate) csvFileName += "_" + dateStr;
    csvFileName += ".csv";

    if (csvFileName.left(1) == ".")
    {
        csvFileName = path + QDir::cleanPath(csvFileName);
    }
    projectSettings->endGroup();

    projectSettings->beginGroup("shapefile");
    // UCM
    ucmFileName = projectSettings->value("UCM","").toString();
    if (ucmFileName.left(1) == ".")
    {
        ucmFileName = path + QDir::cleanPath(ucmFileName);
    }

    // Field list
    fieldListFileName = projectSettings->value("field_list", "").toString();
    if (fieldListFileName.left(1) == ".")
    {
        fieldListFileName = path + QDir::cleanPath(fieldListFileName);
    }

    // Shapefile
    shapeFilePath = getFilePath(csvFileName) + dateStr;
    shapeFileName = shapeFilePath + "/" + getFileName(csvFileName) + ".shp";

    projectSettings->endGroup();

    projectSettings->beginGroup("aggregation");
    // Aggregation Shape
    aggregationShape = projectSettings->value("aggregation_shape","").toString();
    if (aggregationShape.left(1) == ".")
    {
        aggregationShape = path + QDir::cleanPath(aggregationShape);
    }

    // Shape Field
    shapeFieldName = projectSettings->value("shape_field", "").toString();
    if (shapeFieldName.isNull() || shapeFieldName.isEmpty())
    {
        projectError = "Missing shape field name";
        return false;
    }

    // Aggregation List
    aggregationListFileName = projectSettings->value("aggregation_list","").toString();
    if (aggregationListFileName.left(1) == ".")
    {
        aggregationListFileName = path + QDir::cleanPath(aggregationListFileName);
    }

    // Aggregation cell size
    bool ok;
    aggregationCellSize = projectSettings->value("aggregation_cellsize","").toString().toInt(&ok, 10);
    if (!ok)
    {
        projectError = "Invalid aggregation cell size";
        return false;
    }

    addDate = projectSettings->value("add_date_to_filename","").toBool();

    // aggregation output
    csvAggregationOutputFileName = projectSettings->value("aggregation_output","").toString();
    if (csvAggregationOutputFileName.right(4) == ".csv")
    {
        csvAggregationOutputFileName = csvAggregationOutputFileName.left(csvAggregationOutputFileName.length()-4);
    }
    if (addDate) csvAggregationOutputFileName += "_" + dateStr;
    csvAggregationOutputFileName += ".csv";

    if (csvAggregationOutputFileName.left(1) == ".")
    {
        csvAggregationOutputFileName = path + QDir::cleanPath(csvAggregationOutputFileName);
    }

    projectSettings->endGroup();

    return true;
}


int CriteriaOutputProject::createCsvFile()
{
    logger.writeInfo("Create CSV...");

    initializeProjectCsv();

    // load computation unit list
    if (! loadUnitList(dbUnitsName, unitList, projectError))
    {
        return ERROR_READ_UNITS;
    }
    logger.writeInfo("Query result: " + QString::number(unitList.size()) + " distinct computation units.");

    if (!initializeCsvOutputFile())
    {
        return ERROR_PARSERCSV;
    }

    // write output
    for (unsigned int i=0; i < unitList.size(); i++)
    {
        int myResult = writeCsvOutputUnit(i);
        if (myResult != CRIT3D_OK)
        {
            return myResult;
        }
    }

    return CRIT3D_OK;
}


int CriteriaOutputProject::createShapeFile()
{
    if (! QDir(csvFileName).exists())
    {
        // create CSV
        int myResult = createCsvFile();
        if (myResult != CRIT3D_OK)
        {
            return myResult;
        }
    }

    logger.writeInfo("Create SHAPEFILE...");

    Crit3DShapeHandler inputShape, outputShape;

    if (!inputShape.open(ucmFileName.toStdString()))
    {
        projectError = "Wrong shapefile: " + ucmFileName;
        return ERROR_SHAPEFILE;
    }

    logger.writeInfo("UCM shapefile: " + ucmFileName);
    logger.writeInfo("CSV data: " + csvFileName);
    logger.writeInfo("Shape field list: " + fieldListFileName);
    logger.writeInfo("Write shapefile...");

    if (! QDir(shapeFilePath).exists())
    {
        QDir().mkdir(shapeFilePath);
    }
    if (! shapeFromCsv(&inputShape, &outputShape, csvFileName, fieldListFileName, shapeFileName, projectError))
    {
        return ERROR_SHAPEFILE;
    }

    logger.writeInfo("Output shapefile: " + shapeFileName);
    return CRIT3D_OK;
}


int CriteriaOutputProject::createAggregationFile()
{
    if (! QDir(shapeFileName).exists())
    {
        // create shapefile
        int myResult = createShapeFile();
        if (myResult != CRIT3D_OK)
        {
            return myResult;
        }
    }

    logger.writeInfo("Create AGGREGATION...");


    // TODO


    return CRIT3D_OK;
}


bool CriteriaOutputProject::initializeCsvOutputFile()
{
    outputFile.setFileName(csvFileName);
    if (!outputFile.open(QIODevice::ReadWrite | QIODevice::Truncate))
    {
        projectError = "Open failure: " + csvFileName;
        return false;
    }
    else
    {
        logger.writeInfo("Output file: " + csvFileName + "\n");
    }

    if (!outputVariable.parserOutputVariable(variableListFileName, projectError))
    {
        projectError = "Open failure: " + variableListFileName + "\n" + projectError;
        return false;
    }
    QString header = "date,ID_CASE,CROP," + outputVariable.outputVarName.join(",");
    QTextStream out(&outputFile);
    out << header << "\n";

    return true;
}


int CriteriaOutputProject::writeCsvOutputUnit(unsigned int unitIndex)
{
    // IRRI RATIO
    float irriRatio = getIrriRatioFromClass(&(dbCrop), "crop_class", "id_class", unitList[unitIndex].idCropClass, &(projectError));

    QString idCase = unitList[unitIndex].idCase;
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
                        projectError = "Parser CSV error";
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

            int selectRes = selectSimpleVar(dbData, idCase, varName, computation, firstDate, lastDate, irriRatio, &resVector);
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
                projectError = "Parser CSV error";
                return ERROR_PARSERCSV;
            }
            int DTXRes = computeDTX(dbData, idCase, periodTDX, computation, firstDate, lastDate, &resVector);
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
                        projectError = qry.lastError().text();
                        return ERROR_DBHISTORICAL;
                    }
                    qry.first();
                    if (!qry.isValid())
                    {
                        projectError = qry.lastError().text();
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
                                selectRes = selectSimpleVar(dbDataHistorical, idCase, varName, computation, firstDate, lastDate, irriRatio, &resVector);
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
                                selectRes = computeDTX(dbDataHistorical, idCase, periodTDX , computation, firstDate, lastDate, &resVector);
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
    QTextStream out(&outputFile);
    out << dateComputation.toString("yyyy-MM-dd");
    out << "," << unitList[unitIndex].idCase;
    out << "," << getCropFromClass(&(dbCrop), "crop_class", "id_class", unitList[unitIndex].idCropClass, &(projectError)).toUpper();
    out << "," << results.join(",");
    out << "\n";

    outputFile.flush();

    return CRIT3D_OK;
}

int CriteriaOutputProject::selectSimpleVar(QSqlDatabase db, QString idCase, QString varName, QString computation, QDate firstDate, QDate lastDate, float irriRatio, QVector<float>* resVector)
{

    QSqlQuery qry(db);
    int count = 0;
    QString statement;
    float result = NODATA;
    statement = QString("SELECT %1(`%2`) FROM `%3` WHERE DATE >= '%4' AND DATE <= '%5'").arg(computation).arg(varName).arg(idCase).arg(firstDate.toString("yyyy-MM-dd")).arg(lastDate.toString("yyyy-MM-dd"));
    if( !qry.exec(statement) )
    {
        projectError = "Wrong computation: " + computation + "\n" + qry.lastError().text();
        return ERROR_OUTPUT_VARIABLES ;
    }
    qry.first();
    if (!qry.isValid())
    {
        projectError = "Missing data: " + statement;
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
        projectError = "Incomplete data: " + statement;
        return ERROR_INCOMPLETE_DATA;
    }

    return CRIT3D_OK;

}

int CriteriaOutputProject::computeDTX(QSqlDatabase db, QString idCase, int period, QString computation, QDate firstDate, QDate lastDate, QVector<float>* resVector)
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
            projectError = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES;
        }
        qry.first();
        if (!qry.isValid())
        {
            projectError = qry.lastError().text();
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
            projectError = qry.lastError().text();
            return ERROR_OUTPUT_VARIABLES ;
        }
        qry.first();
        if (!qry.isValid())
        {
            projectError = qry.lastError().text();
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
