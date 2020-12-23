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
#include "computationUnitsDb.h"

#ifdef GDAL
    #include "gdalShapeFunctions.h"
#endif

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
    aggregationListFileName = "";
    aggregationCellSize = "";

    mapListFileName = "";
    mapCellSize = "";
    mapFormat = "";
    mapProjection = "";

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


int CriteriaOutputProject::initializeProject(QString settingsFileName, QDate dateComputation, bool isLog)
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

    if (isLog)
    {
        logger.setLog(path,projectName);
    }

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
    if (dbDataName.isEmpty())
    {
        dbDataName = projectSettings->value("db_output","").toString();
    }
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

    projectSettings->beginGroup("maps");
    // MAPS
    mapListFileName = projectSettings->value("map_list","").toString();
    if (mapListFileName.left(1) == ".")
    {
        mapListFileName = path + QDir::cleanPath(mapListFileName);
    }

    // format
    mapFormat = projectSettings->value("format", "").toString();
    // projection
    mapProjection = projectSettings->value("projection", "").toString();
    // map cell size
    mapCellSize = projectSettings->value("cellsize","").toString();

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

    // read unit list
    logger.writeInfo("DB computation units: " + dbUnitsName);
    if (! readUnitList(dbUnitsName, unitList, projectError))
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

    // read unit list
    logger.writeInfo("DB computation units: " + dbUnitsName);
    if (! readUnitList(dbUnitsName, unitList, projectError))
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
            if (QFile(outputCsvFileName).exists())
            {
                QDir().remove(outputCsvFileName);
            }
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


#ifdef GDAL
int CriteriaOutputProject::createMaps()
{
    // check map list
    if (! QFile(mapListFileName).exists())
    {
        projectError = "Missing map list: " + mapListFileName;
        return ERROR_SETTINGS_MISSINGDATA;
    }

    // check cellsize
    bool ok;
    mapCellSize.toInt(&ok, 10);
    if (!ok)
    {
        projectError = "Invalid map cellsize: " + mapCellSize;
        return ERROR_SETTINGS_MISSINGDATA;
    }

    // check format and projection
    if (mapProjection.isEmpty())
    {
        projectError = "Missing projection ";
        return ERROR_SETTINGS_MISSINGDATA;
    }

    if (!mapExtensionShortName.contains(mapFormat))
    {
        projectError = "Unknown output format ";
        return ERROR_SETTINGS_MISSINGDATA;
    }

    // check shapefile
    if (! QFile(outputShapeFileName).exists())
    {
        int myResult = createShapeFile();
        if (myResult != CRIT3D_OK)
        {
            return myResult;
        }
    }

    logger.writeInfo("MAPS");

    // parser csv file mapListFileName
    QStringList inputField;
    QStringList outputName;
    QFile mapList(mapListFileName);
    if ( !mapList.open(QFile::ReadOnly | QFile::Text) )
    {
        projectError = "Map List csv file not exists: " + mapListFileName;
        return ERROR_SETTINGS_MISSINGDATA;
    }
    else
    {
        QTextStream in(&mapList);
        //skip header
        QString line = in.readLine();
        QStringList header = line.split(",");
        // whitespace removed from the start and the end.
        QMutableListIterator<QString> it(header);
        while (it.hasNext()) {
            it.next();
            it.value() = it.value().trimmed();
        }
        while (!in.atEnd())
        {
            line = in.readLine();
            QStringList items = line.split(",");
            if (items.size() < REQUIREDMAPLISTCSVINFO)
            {
                projectError = "invalid map list format CSV, input field and output file name required";
                return ERROR_SETTINGS_MISSINGDATA;
            }
            int pos = header.indexOf("input field (shapefile)");
            if (pos == -1)
            {
                projectError = "missing input field";
                return ERROR_SETTINGS_MISSINGDATA;
            }
            // remove whitespace
            inputField.push_back(items[pos].toUpper().trimmed());
            if (inputField.isEmpty())
            {
                projectError = "missing input field";
                return ERROR_SETTINGS_MISSINGDATA;
            }

            pos = header.indexOf("output map name");
            if (pos == -1)
            {
                projectError = "missing output map name";
                return ERROR_SETTINGS_MISSINGDATA;
            }
            // remove whitespace
            outputName.push_back(items[pos].toUpper().trimmed());
            if (outputName.isEmpty())
            {
                projectError = "missing output map name";
                return ERROR_SETTINGS_MISSINGDATA;
            }
        }

    }

    int rasterOK = 0;

    for (int i=0; i < inputField.size(); i++)
    {
        QString mapName = outputShapeFilePath + "/" + outputName[i]+ "." + mapFormat;
        std::string inputFieldStd = inputField[i].toStdString();
        if (shapeToRaster(outputShapeFileName, inputFieldStd, mapCellSize, mapProjection, mapName, projectError))
        {
            rasterOK = rasterOK + 1;
        }
    }

    if (rasterOK == inputField.size())
    {
        return CRIT3D_OK;
    }
    else
    {
        int nRasterError = inputField.size() - rasterOK;
        projectError = QString::number(nRasterError) + " invalid raster - " + projectError;
        return false;
    }
}
#endif


int CriteriaOutputProject::createAggregationFile()
{
    logger.writeInfo("AGGREGATION");

    // check aggregation file
    QString aggregationPath = getFilePath(outputAggrCsvFileName);
    if (! QDir(aggregationPath).exists())
    {
        QDir().mkdir(aggregationPath);
    }
    if (QFile(outputAggrCsvFileName).exists())
    {
        logger.writeInfo("Remove old aggregation: " + outputAggrCsvFileName);
        QFile().remove(outputAggrCsvFileName);
    }

    if (shapeFieldName.isNull() || shapeFieldName.isEmpty())
    {
        projectError = "Missing shape field name.";
        return ERROR_SETTINGS_MISSINGDATA;
    }

    // check aggregation cell size
    bool ok;
    int cellSize = aggregationCellSize.toInt(&ok, 10);
    if (!ok)
    {
        projectError = "Invalid aggregation cellsize: " + aggregationCellSize;
        return ERROR_SETTINGS_MISSINGDATA;
    }

    // check shapefile
    if (! QFile(outputShapeFileName).exists())
    {
        // create shapefile
        int myResult = createShapeFile();
        if (myResult != CRIT3D_OK)
        {
            return myResult;
        }
    }

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

bool CriteriaOutputProject::getAllDbVariable(QString &projectError)
{
    // check DB
    if (!QFile(dbDataName).exists())
    {
        projectError = "missing file: " + dbDataName;
        return false;
    }
    // open DB
    dbData = QSqlDatabase::addDatabase("QSQLITE", "data");
    dbData.setDatabaseName(dbDataName);
    if (! dbData.open())
    {
        projectError = "open DB data failed: " + dbData.lastError().text();
        return false;
    }

    QSqlQuery qry(dbData);
    QString statement = QString("SELECT name FROM sqlite_master WHERE type ='table' AND name NOT LIKE 'sqlite_%' ESCAPE '^'");
    QString tableName;
    QStringList varList;
    if( !qry.exec(statement) )
    {
        projectError = qry.lastError().text();
        return false;
    }
    qry.first();
    if (!qry.isValid())
    {
        projectError = qry.lastError().text();
        return false ;
    }
    getValue(qry.value("name"), &tableName);
    statement = QString("PRAGMA table_info(`%1`)").arg(tableName);
    QString name;
    if( !qry.exec(statement) )
    {
        projectError = qry.lastError().text();
        return false;
    }
    qry.first();
    if (!qry.isValid())
    {
        projectError = qry.lastError().text();
        return false;
    }
    do
    {
        getValue(qry.value("name"), &name);
        if (name != "DATE")
        {
            varList<<name;
        }
    }
    while(qry.next());

    if (varList.isEmpty())
    {
        return false;
    }
    else
    {
        outputVariable.varName = varList;
        return true;
    }
}

bool CriteriaOutputProject::getDbDataDates(QDate* firstDate, QDate* lastDate, QString &projectError)
{
    QStringList tablesList = dbData.tables();
    if (tablesList.isEmpty())
    {
        projectError = "Db is empty";
        return false;
    }

    QSqlQuery qry(dbData);
    QString idCase;
    QString statement;
    QDate firstTmp;
    QDate lastTmp;

    *firstDate = QDate::currentDate();
    *lastDate = QDate(1800,1,1);

    for (int i = 0; i < tablesList.size(); i++)
    {
        idCase = tablesList[i];
        statement = QString("SELECT MIN(DATE),MAX(DATE) FROM `%1`").arg(idCase);
        if( !qry.exec(statement) )
        {
            projectError = qry.lastError().text();
            return false;
        }
        qry.first();
        if (!qry.isValid())
        {
            projectError = qry.lastError().text();
            return false ;
        }
        getValue(qry.value("MIN(DATE)"), &firstTmp);
        getValue(qry.value("MAX(DATE)"), &lastTmp);

        if (firstTmp < *firstDate)
        {
            *firstDate = firstTmp;
        }
        if (lastTmp > *lastDate)
        {
            *lastDate = lastTmp;
        }
    }

    if (!firstDate->isValid() || !lastDate->isValid())
    {
        projectError = "Invalid date";
        return false;
    }

    return true;
}

int CriteriaOutputProject::createCsvFileFromGUI(QDate dateComputation, QString csvFileName)
{

    int myResult = initializeProjectCsv();
    if (myResult != CRIT3D_OK)
    {
        return myResult;
    }

    outputCsvFileName = csvFileName;
    // open outputCsvFileName and write header
    outputFile.setFileName(outputCsvFileName);
    if (!outputFile.open(QIODevice::ReadWrite | QIODevice::Truncate))
    {
        projectError = "Open failure: " + outputCsvFileName;
        return ERROR_CSVFILE;
    }

    QString header = "date,ID_CASE,CROP," + outputVariable.outputVarName[0];
    QTextStream out(&outputFile);
    out << header << "\n";
    outputFile.close();

    // read unit list
    if (! readUnitList(dbUnitsName, unitList, projectError))
    {
        return ERROR_READ_UNITS;
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
            if (QFile(csvFileName).exists())
            {
                QDir().remove(csvFileName);
            }
            return myResult;
        }
    }
    return CRIT3D_OK;
}

int CriteriaOutputProject::createShapeFileFromGUI()
{
    Crit3DShapeHandler inputShape;

    if (!inputShape.open(ucmFileName.toStdString()))
    {
        projectError = "Wrong shapefile: " + ucmFileName;
        return ERROR_SHAPEFILE;
    }

    fieldListFileName = "";
    outputShapeFilePath = getFilePath(outputCsvFileName);
    QFileInfo csvFileInfo(outputCsvFileName);
    outputShapeFileName = outputShapeFilePath + "/" + csvFileInfo.baseName() + ".shp";

    if (! shapeFromCsv(inputShape, outputCsvFileName, fieldListFileName, outputShapeFileName, projectError))
    {
        return ERROR_SHAPEFILE;
    }

    return CRIT3D_OK;
}
