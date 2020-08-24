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

    // Field listgetFileNamegetFileName
    fieldListFileName = projectSettings->value("field_list", "").toString();
    if (fieldListFileName.left(1) == ".")
    {
        fieldListFileName = path + QDir::cleanPath(fieldListFileName);
    }

    // Shapefile
    shapeFilePath = getFilePath(csvFileName) + dateStr;
    QFileInfo csvFileInfo(csvFileName);
    shapeFileName = shapeFilePath + "/" + csvFileInfo.baseName() + ".shp";

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
    logger.writeInfo("Write csv...");

    if (!initializeCsvOutputFile())
    {
        return ERROR_PARSERCSV;
    }

    // write output
    QString idCase;
    QString idCropClass;
    for (unsigned int i=0; i < unitList.size(); i++)
    {
        idCase = unitList[i].idCase;
        idCropClass = unitList[i].idCropClass;

        myResult = writeCsvOutputUnit(idCase, idCropClass, dbData, dbCrop, dbDataHistorical, dateComputation, outputVariable, csvFileName, &projectError);
        if (myResult != CRIT3D_OK)
        {
            return myResult;
        }
    }

    return CRIT3D_OK;
}


int CriteriaOutputProject::createShapeFile()
{
    if (! QFile(csvFileName).exists())
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
    logger.writeInfo("CSV data: " + csvFileName);
    logger.writeInfo("Shape field list: " + fieldListFileName);
    logger.writeInfo("Write shapefile...");

    if (! QDir(shapeFilePath).exists())
    {
        QDir().mkdir(shapeFilePath);
    }
    if (! shapeFromCsv(inputShape, csvFileName, fieldListFileName, shapeFileName, projectError))
    {
        return ERROR_SHAPEFILE;
    }

    logger.writeInfo("Output shapefile: " + shapeFileName);
    return CRIT3D_OK;
}


int CriteriaOutputProject::createAggregationFile()
{
    if (shapeFieldName.isNull() || shapeFieldName.isEmpty())
    {
        projectError = "Missing shape field name";
        return ERROR_SETTINGS_MISSINGDATA;
    }

    // Aggregation cell size
    bool ok;
    int cellSize = aggregationCellSize.toInt(&ok, 10);
    if (!ok)
    {
        projectError = "Invalid aggregation cell size";
        return ERROR_SETTINGS_WRONGFILENAME;
    }

    if (csvAggregationOutputFileName.right(4) != ".csv")
    {
        projectError = "aggregation output is not a csv file";
        return ERROR_SETTINGS_WRONGFILENAME;
    }

    if (! QFile(shapeFileName).exists())
    {
        // create shapefile
        int myResult = createShapeFile();
        if (myResult != CRIT3D_OK)
        {
            return myResult;
        }
    }

    logger.writeInfo("Create AGGREGATION...");

    if (!shapeVal.open(shapeFileName.toStdString()))
    {
        projectError = "Load shapefile failed: " + shapeFileName;
        return ERROR_SHAPEFILE;
    }

    QFileInfo aggFileInfo(csvAggregationOutputFileName);
    QString shapeRefPath = shapeFilePath + "/" + aggFileInfo.baseName();

    if (! QFile(aggregationShape).exists())
    {
        projectError = aggregationShape + " not exists";
        return ERROR_SHAPEFILE;
    }
    QString shapeRefFileName = cloneShapeFile(aggregationShape, shapeRefPath);
    if (!shapeRef.open(shapeRefFileName.toStdString()))
    {
        projectError = "Load shapefile failed: " + shapeRefFileName;
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

    if (!isOk)
    {
        return ERROR_ZONAL_STATISTICS_SHAPE;
    }
    else
    {
        return CRIT3D_OK;
    }
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
    QString csvFilePath = getFilePath(csvFileName);
    if (! QDir(csvFilePath).exists())
    {
        QDir().mkdir(csvFilePath);
    }

    // open csvFileName
    outputFile.setFileName(csvFileName);
    if (!outputFile.open(QIODevice::ReadWrite | QIODevice::Truncate))
    {
        projectError = "Open failure: " + csvFileName;
        return false;
    }
    else
    {
        logger.writeInfo("Output file: " + csvFileName);
    }

    QString header = "date,ID_CASE,CROP," + outputVariable.outputVarName.join(",");
    QTextStream out(&outputFile);
    out << header << "\n";
    outputFile.close();

    return true;
}
