#include "commonConstants.h"
#include "gis.h"
#include "criteriaOutputProject.h"
#include "criteriaOutputElaboration.h"
#include "logger.h"
#include "utilities.h"
#include "shapeHandler.h"
#include "shapeFromCsv.h"
#include "shapeUtilities.h"
#include "shapeToRaster.h"
#include "zonalStatistic.h"

#include <QtSql>
#include <iostream>


CriteriaOutputProject::CriteriaOutputProject()
{
    initialize();
}

void CriteriaOutputProject::initialize()
{
    isProjectLoaded = false;

    path = "";
    projectName = "";
    dbUnitsName = "";
    dbDataName = "";
    dbDataHistoricalName = "";
    dbCropName = "";
    variableListFileName = "";
    ucmFileName = "";
    aggregationShapeFileName = "";
    shapeFieldName = "";
    fieldListFileName = "";

    outputCsvFileName = "";
    outputShapeFileName = "";
    outputShapeFilePath = "";
    outputAggrCsvFileName = "";

    dbUnitsName = "";
    dbDataName = "";
    dbCropName = "";
    dbDataHistoricalName = "";

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


int CriteriaOutputProject::initializeProjectDtx()
{
    // open DB Data Historical
    if(!dbDataHistoricalName.isEmpty())
    {
        logger.writeInfo("DB data historical: " + dbDataHistoricalName);
        if (! QFile(dbDataHistoricalName).exists())
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
        projectError = "Read settings: " + projectError;
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
    outputCsvFileName = projectSettings->value("csv_output","").toString();
    if (outputCsvFileName.right(4) == ".csv")
    {
        outputCsvFileName = outputCsvFileName.left(outputCsvFileName.length()-4);
    }
    if (addDate) outputCsvFileName += "_" + dateStr;
    outputCsvFileName += ".csv";

    if (outputCsvFileName.left(1) == ".")
    {
        outputCsvFileName = path + QDir::cleanPath(outputCsvFileName);
    }
    projectSettings->endGroup();

    projectSettings->beginGroup("shapefile");
    // UCM
    ucmFileName = projectSettings->value("UCM","").toString();
    if (ucmFileName.left(1) == ".")
    {
        ucmFileName = path + QDir::cleanPath(ucmFileName);
    }

    // Field listgetFileNamegetFileName
    fieldListFileName = projectSettings->value("field_list", "").toString();
    if (fieldListFileName.left(1) == ".")
    {
        fieldListFileName = path + QDir::cleanPath(fieldListFileName);
    }

    // Shapefile
    outputShapeFilePath = getFilePath(outputCsvFileName) + dateStr;
    QFileInfo csvFileInfo(outputCsvFileName);
    outputShapeFileName = outputShapeFilePath + "/" + csvFileInfo.baseName() + ".shp";

    projectSettings->endGroup();

    projectSettings->beginGroup("aggregation");
    // Aggregation Shape
    aggregationShapeFileName = projectSettings->value("aggregation_shape","").toString();
    if (aggregationShapeFileName.left(1) == ".")
    {
        aggregationShapeFileName = path + QDir::cleanPath(aggregationShapeFileName);
    }

    // Shape Field
    shapeFieldName = projectSettings->value("shape_field", "").toString();

    // Aggregation List
    aggregationListFileName = projectSettings->value("aggregation_list","").toString();
    if (aggregationListFileName.left(1) == ".")
    {
        aggregationListFileName = path + QDir::cleanPath(aggregationListFileName);
    }

    // Aggregation cell size
    aggregationCellSize = projectSettings->value("aggregation_cellsize","").toString();

    addDate = projectSettings->value("add_date_to_filename","").toBool();
    // aggregation output
    outputAggrCsvFileName = projectSettings->value("aggregation_output","").toString();
    if (outputAggrCsvFileName.right(4) == ".csv")
    {
        outputAggrCsvFileName = outputAggrCsvFileName.left(outputAggrCsvFileName.length()-4);
    }

    if (addDate) outputAggrCsvFileName += "_" + dateStr;
    outputAggrCsvFileName += ".csv";

    if (outputAggrCsvFileName.left(1) == ".")
    {
        outputAggrCsvFileName = path + QDir::cleanPath(outputAggrCsvFileName);
    }

    projectSettings->endGroup();

    return true;
}


int CriteriaOutputProject::precomputeDtx()
{
    logger.writeInfo("PRECOMPUTE DTX");

    int myResult = initializeProjectDtx();
    if (myResult != CRIT3D_OK)
    {
        return myResult;
    }

    // load computation unit list
    logger.writeInfo("DB computation units: " + dbUnitsName);
    if (! loadUnitList(dbUnitsName, unitList, projectError))
    {
        return ERROR_READ_UNITS;
    }

    logger.writeInfo("Query result: " + QString::number(unitList.size()) + " distinct computation units.");
    logger.writeInfo("Compute dtx...");

    QString idCase;
    for (unsigned int i=0; i < unitList.size(); i++)
    {
        idCase = unitList[i].idCase;
        logger.writeInfo(QString::number(i) + " ID CASE: " + idCase);

        int myResult = computeAllDtxUnit(dbDataHistorical, idCase, projectError);
        if (myResult != CRIT3D_OK)
        {
            projectError = "ID CASE: " + idCase + "\n" + projectError;
            return myResult;
        }
    }

    return CRIT3D_OK;
}


int CriteriaOutputProject::createCsvFile()
{
    logger.writeInfo("Create CSV");

    int myResult = initializeProjectCsv();
    if (myResult != CRIT3D_OK)
    {
        return myResult;
    }

    // load computation unit list
    logger.writeInfo("DB computation units: " + dbUnitsName);
    if (! loadUnitList(dbUnitsName, unitList, projectError))
    {
        return ERROR_READ_UNITS;
    }
    logger.writeInfo("Query result: " + QString::number(unitList.size()) + " distinct computation units.");

    if (!initializeCsvOutputFile())
    {
        return ERROR_PARSERCSV;
    }

    logger.writeInfo("Write csv...");

    // write output
    QString idCase;
    QString idCropClass;
    for (unsigned int i=0; i < unitList.size(); i++)
    {
        idCase = unitList[i].idCase;
        idCropClass = unitList[i].idCropClass;

        myResult = writeCsvOutputUnit(idCase, idCropClass, dbData, dbCrop, dbDataHistorical, dateComputation, outputVariable, outputCsvFileName, &projectError);
        if (myResult != CRIT3D_OK)
        {
            QDir().remove(outputCsvFileName);
            return myResult;
        }
    }

    return CRIT3D_OK;
}


int CriteriaOutputProject::createShapeFile()
{
    if (! QFile(outputCsvFileName).exists())
    {
        // create CSV
        int myResult = createCsvFile();
        if (myResult != CRIT3D_OK)
        {
            return myResult;
        }
    }

    logger.writeInfo("Create SHAPEFILE");

    Crit3DShapeHandler inputShape;

    if (!inputShape.open(ucmFileName.toStdString()))
    {
        projectError = "Wrong shapefile: " + ucmFileName;
        return ERROR_SHAPEFILE;
    }

    logger.writeInfo("UCM shapefile: " + ucmFileName);
    logger.writeInfo("CSV data: " + outputCsvFileName);
    logger.writeInfo("Shape field list: " + fieldListFileName);
    logger.writeInfo("Output shapefile: " + outputShapeFileName);
    logger.writeInfo("Write shapefile...");

    if (! QDir(outputShapeFilePath).exists())
    {
        QDir().mkdir(outputShapeFilePath);
    }
    if (! shapeFromCsv(inputShape, outputCsvFileName, fieldListFileName, outputShapeFileName, projectError))
    {
        return ERROR_SHAPEFILE;
    }

    return CRIT3D_OK;
}


int CriteriaOutputProject::createAggregationFile()
{
    if (shapeFieldName.isNull() || shapeFieldName.isEmpty())
    {
        projectError = "Missing shape field name.";
        return ERROR_SETTINGS_MISSINGDATA;
    }

    // Aggregation cell size
    bool ok;
    int cellSize = aggregationCellSize.toInt(&ok, 10);
    if (!ok)
    {
        projectError = "Invalid aggregation cellsize: " + aggregationCellSize;
        return ERROR_SETTINGS_WRONGFILENAME;
    }

    if (outputAggrCsvFileName.right(4) != ".csv")
    {
        projectError = "aggregation output is not a csv file.";
        return ERROR_SETTINGS_WRONGFILENAME;
    }

    if (! QFile(outputShapeFileName).exists())
    {
        // create shapefile
        int myResult = createShapeFile();
        if (myResult != CRIT3D_OK)
        {
            return myResult;
        }
    }

    logger.writeInfo("AGGREGATION");
    Crit3DShapeHandler shapeVal, shapeRef;

    if (!shapeVal.open(outputShapeFileName.toStdString()))
    {
        projectError = "Load shapefile failed: " + outputShapeFileName;
        return ERROR_SHAPEFILE;
    }

    QFileInfo aggrFileInfo(outputAggrCsvFileName);
    QString outputAggrShapePath = outputShapeFilePath + "/" + aggrFileInfo.baseName();

    logger.writeInfo("Aggregation shapefile: " + aggregationShapeFileName);

    if (! QFile(aggregationShapeFileName).exists())
    {
        projectError = aggregationShapeFileName + " not exists";
        return ERROR_SHAPEFILE;
    }

    QString outputAggrShapeFileName = cloneShapeFile(aggregationShapeFileName, outputAggrShapePath);

    if (!shapeRef.open(outputAggrShapeFileName.toStdString()))
    {
        projectError = "Load shapefile failed: " + outputAggrShapeFileName;
        return ERROR_SHAPEFILE;
    }

    // check shape type
    if ( shapeRef.getTypeString() != shapeVal.getTypeString() || shapeRef.getTypeString() != "2D Polygon" )
    {
        projectError = "shape type error: not 2D Polygon type" ;
        return false;
    }

    // check proj
    if (shapeRef.getIsWGS84() == false || shapeVal.getIsWGS84() == false)
    {
        projectError = "projection error: not WGS84" ;
        return false;
    }

    // check utm zone
    if (shapeRef.getUtmZone() != shapeVal.getUtmZone())
    {
        projectError = "utm zone: different utm zones" ;
        return false;
    }

    // parser aggregation list
    if (!aggregationVariable.parserAggregationVariable(aggregationListFileName, projectError))
    {
        projectError = "Open failure: " + aggregationListFileName + "\n" + projectError;
        return false;
    }

    logger.writeInfo("output shapefile: " + outputAggrShapeFileName);
    logger.writeInfo("output csv file: " + outputAggrCsvFileName);
    logger.writeInfo("Compute aggregation...");

    //shape to raster
    gis::Crit3DRasterGrid rasterRef;
    gis::Crit3DRasterGrid rasterVal;
    initializeRasterFromShape(shapeRef, rasterRef, cellSize);
    initializeRasterFromShape(shapeVal, rasterVal, cellSize);

    fillRasterWithShapeNumber(rasterRef, shapeRef);
    fillRasterWithShapeNumber(rasterVal, shapeVal);

    std::vector <int> vectorNull;
    std::vector <std::vector<int> > matrix = computeMatrixAnalysis(shapeRef, shapeVal, rasterRef, rasterVal, vectorNull);
    bool isOk = false;
    for(int i=0; i < aggregationVariable.outputVarName.size(); i++)
    {
        std::string error;
        if (aggregationVariable.aggregationType[i] == "MAJORITY")
        {
            isOk = zonalStatisticsShapeMajority(shapeRef, shapeVal, matrix, vectorNull,
                                                aggregationVariable.inputField[i].toStdString(),
                                                aggregationVariable.outputVarName[i].toStdString(), error);
        }
        else
        {
            isOk = zonalStatisticsShape(shapeRef, shapeVal, matrix, vectorNull, aggregationVariable.inputField[i].toStdString(),
                                        aggregationVariable.outputVarName[i].toStdString(),
                                        aggregationVariable.aggregationType[i].toStdString(), error);
        }

        if (!isOk) break;
    }

    rasterRef.clear();
    rasterVal.clear();
    vectorNull.clear();
    matrix.clear();
    shapeVal.close();

    if (!isOk)
    {
        shapeRef.close();
        return ERROR_ZONAL_STATISTICS_SHAPE;
    }

    // write csv aggragation data
    int myResult = writeCsvAggrFromShape(shapeRef, outputAggrCsvFileName, dateComputation,
                                 aggregationVariable.outputVarName, shapeFieldName, projectError);

    shapeRef.close();
    return myResult;
}


bool CriteriaOutputProject::initializeCsvOutputFile()
{
    // parse output variables
    if (!outputVariable.parserOutputVariable(variableListFileName, projectError))
    {
        projectError = "Open failure: " + variableListFileName + "\n" + projectError;
        return false;
    }

    // check output csv directory
    QString csvFilePath = getFilePath(outputCsvFileName);
    if (! QDir(csvFilePath).exists())
    {
        QDir().mkdir(csvFilePath);
    }

    // open outputCsvFileName
    outputFile.setFileName(outputCsvFileName);
    if (!outputFile.open(QIODevice::ReadWrite | QIODevice::Truncate))
    {
        projectError = "Open failure: " + outputCsvFileName;
        return false;
    }
    else
    {
        logger.writeInfo("Output file: " + outputCsvFileName);
    }

    QString header = "date,ID_CASE,CROP," + outputVariable.outputVarName.join(",");
    QTextStream out(&outputFile);
    out << header << "\n";
    outputFile.close();

    return true;
}
