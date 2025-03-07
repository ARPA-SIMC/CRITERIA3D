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
#include "netcdfHandler.h"
#include "utilities.h"

#ifdef GDAL
    #include "gdalShapeFunctions.h"
#endif

#include <QtSql>
#include <iostream>
#include <math.h>


CriteriaOutputProject::CriteriaOutputProject()
{
    initialize();
}

void CriteriaOutputProject::initialize()
{
    isProjectLoaded = false;

    path = "";
    projectName = "";
    operation = "";
    dbComputationUnitsName = "";
    dbDataName = "";
    dbClimateDataName = "";
    dbCropName = "";
    variableListFileName = "";
    ucmFileName = "";
    aggregationShapeFileName = "";
    shapeFieldName = "";
    fieldListFileName = "";
    aggregationListFileName = "";
    aggregationCellSize = "";
    aggregationThreshold = "";

    mapListFileName = "";
    mapPalettePath = "";
    mapCellSize = "";
    mapFormat = "";
    mapAreaName = "";
    mapProjection = "";
    isPngCopy = false;
    pngProjection = "";

    outputCsvFileName = "";
    outputShapeFileName = "";
    outputShapeFilePath = "";
    outputAggrCsvFileName = "";

    dbComputationUnitsName = "";
    dbDataName = "";
    dbCropName = "";
    dbClimateDataName = "";

    projectError = "";
    nrUnits = 0;

    logFileName = "";
    addDateTimeLogFile = false;
}


void CriteriaOutputProject::closeProject()
{
    if (isProjectLoaded)
    {
        logger.writeInfo("Close Project...");

        initialize();

        compUnitList.clear();
        outputFile.close();
        logFile.close();
        dbData.close();
        dbCrop.close();
        dbClimateData.close();

        isProjectLoaded = false;
    }
}


int CriteriaOutputProject::initializeProjectDtx()
{
    // open DB climate
    if(dbClimateDataName.isEmpty())
    {
        projectError = "Missing climate data filename in the ini file ('db_data_climate')";
        return ERROR_DBPARAMETERS;
    }

    logger.writeInfo("DB climate data: " + dbClimateDataName);

    if (! QFile(dbClimateDataName).exists())
    {
        projectError = "DB climate data doesn't exist";
        return ERROR_DBPARAMETERS;
    }

    dbClimateData = QSqlDatabase::addDatabase("QSQLITE", "climateData");
    dbClimateData.setDatabaseName(dbClimateDataName);
    if (! dbClimateData.open())
    {
        projectError = "Open DB climate data failed: " + dbClimateData.lastError().text();
        return ERROR_DBPARAMETERS;
    }

    return CRIT1D_OK;
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
    if (! dbData.open() || ! dbData.lastError().text().isEmpty())
    {
        projectError = "Open DB data failed: " + dbData.lastError().text();
        return ERROR_DBPARAMETERS;
    }

    // open DB Data climate
    if(!dbClimateDataName.isEmpty())
    {
        logger.writeInfo("DB data climate: " + dbClimateDataName);
        if (!QFile(dbClimateDataName).exists())
        {
            projectError = "DB data climate doesn't exist";
            return ERROR_DBPARAMETERS;
        }

        dbClimateData = QSqlDatabase::addDatabase("QSQLITE", "climateData");
        dbClimateData.setDatabaseName(dbClimateDataName);
        if (! dbClimateData.open())
        {
            projectError = "Open DB data climate failed: " + dbClimateData.lastError().text();
            return ERROR_DBPARAMETERS;
        }
    }

    return CRIT1D_OK;
}


int CriteriaOutputProject::initializeProject(const QString &settingsFileName, const QString &operationStr,
                                             const QDate &_dateComputation, bool isLog)
{
    closeProject();
    initialize();
    this->dateComputation = _dateComputation;
    this->operation = operationStr;

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

    if (! readSettings())
    {
        projectError = "Read settings: " + projectError;
        return ERROR_SETTINGS_MISSINGDATA;
    }

    if (isLog)
    {
        QString fileName = projectName + "_" + operation;
        logger.setLog(path, fileName, addDateTimeLogFile);
    }

    isProjectLoaded = true;
    return CRIT1D_OK;
}


bool CriteriaOutputProject::readSettings()
{
    QSettings* projectSettings;
    projectSettings = new QSettings(configFileName, QSettings::IniFormat);

    // PROJECT
    projectSettings->beginGroup("project");

    QString dateStr = dateComputation.toString("yyyy-MM-dd");

    projectName = projectSettings->value("name", "").toString();

    // computational units
    dbComputationUnitsName = projectSettings->value("db_comp_units", "").toString();
    if (dbComputationUnitsName.isEmpty())
    {
        // check old name
        dbComputationUnitsName = projectSettings->value("db_units", "").toString();
    }
    if (dbComputationUnitsName.isEmpty())
    {
        projectError = "Missing db_comp_units";
        return false;
    }
    else if (dbComputationUnitsName.at(0) == '.')
    {
        dbComputationUnitsName = QDir().cleanPath(path + dbComputationUnitsName);
    }

    dbDataName = projectSettings->value("db_data","").toString();
    if (dbDataName.isEmpty())
    {
        dbDataName = projectSettings->value("db_output","").toString();
    }
    if (! dbDataName.isEmpty())
    {
        if (dbDataName.at(0) == '.')
        {
            dbDataName = QDir::cleanPath(path + dbDataName);
        }
    }

    dbCropName = projectSettings->value("db_crop", "").toString();
    if (! dbCropName.isEmpty())
    {
        if (dbCropName.at(0) == '.')
        {
            dbCropName = QDir::cleanPath(path + dbCropName);
        }
    }

    dbClimateDataName = projectSettings->value("db_data_climate", "").toString();
    if (dbClimateDataName.isEmpty())
    {
        // check old name
        dbClimateDataName = projectSettings->value("db_data_historical","").toString();
    }

    if (! dbClimateDataName.isEmpty())
    {
        if (dbClimateDataName.at(0) == '.')
        {
            dbClimateDataName = QDir::cleanPath(path + dbClimateDataName);
        }
    }

    addDateTimeLogFile = projectSettings->value("add_date_to_log","").toBool();
    projectSettings->endGroup();

    // CSV
    projectSettings->beginGroup("csv");

    variableListFileName = projectSettings->value("variable_list","").toString();
    if (! variableListFileName.isEmpty() && variableListFileName.at(0) == '.')
    {
        variableListFileName = QDir::cleanPath(path + variableListFileName);
    }

    bool addDate = projectSettings->value("add_date_to_filename","").toBool();

    outputCsvFileName = projectSettings->value("csv_output","").toString();
    if (! outputCsvFileName.isEmpty())
    {
        if (outputCsvFileName.at(0) == '.')
        {
            outputCsvFileName = QDir::cleanPath(path + outputCsvFileName);
        }
        if (outputCsvFileName.right(4) == ".csv")
        {
            outputCsvFileName = outputCsvFileName.left(outputCsvFileName.length()-4);
        }

        if (addDate) outputCsvFileName += "_" + dateStr;
        outputCsvFileName += ".csv";
    }

    projectSettings->endGroup();

    // SHAPEFILE
    projectSettings->beginGroup("shapefile");

    ucmFileName = projectSettings->value("UCM","").toString();
    if (! ucmFileName.isEmpty() && ucmFileName.at(0) == '.')
    {
        ucmFileName = QDir::cleanPath(path + ucmFileName);
    }

    fieldListFileName = projectSettings->value("field_list", "").toString();
    if (! fieldListFileName.isEmpty() && fieldListFileName.at(0) == '.')
    {
        fieldListFileName = QDir::cleanPath(path + fieldListFileName);
    }

    // output shapefile
    outputShapeFilePath = getFilePath(outputCsvFileName) + dateStr;
    QFileInfo csvFileInfo(outputCsvFileName);
    outputShapeFileName = outputShapeFilePath + "/" + csvFileInfo.baseName() + ".shp";

    projectSettings->endGroup();

    // AGGREGATION
    projectSettings->beginGroup("aggregation");

    aggregationShapeFileName = projectSettings->value("aggregation_shape","").toString();
    if (aggregationShapeFileName.left(1) == ".")
    {
        aggregationShapeFileName = QDir::cleanPath(path + aggregationShapeFileName);
    }

    shapeFieldName = projectSettings->value("shape_field", "").toString();
    if (shapeFieldName.left(1) == ".")
    {
        shapeFieldName = QDir::cleanPath(path + shapeFieldName);
    }

    aggregationListFileName = projectSettings->value("aggregation_list","").toString();
    if (aggregationListFileName.left(1) == ".")
    {
        aggregationListFileName = QDir::cleanPath(path + aggregationListFileName);
    }

    aggregationCellSize = projectSettings->value("aggregation_cellsize","").toString();

    aggregationThreshold = projectSettings->value("aggregation_threshold","").toString();
    // default threshold
    if (aggregationThreshold == "") aggregationThreshold = "0.5";

    addDate = projectSettings->value("add_date_to_filename","").toBool();

    // aggregation output file name
    outputAggrCsvFileName = projectSettings->value("aggregation_output","").toString();
    if (! outputAggrCsvFileName.isEmpty())
    {
        if (outputAggrCsvFileName.right(4) == ".csv")
            outputAggrCsvFileName = outputAggrCsvFileName.left(outputAggrCsvFileName.length()-4);

        if (addDate)
            outputAggrCsvFileName += "_" + dateStr;

        outputAggrCsvFileName += ".csv";

        if (outputAggrCsvFileName.at(0) == '.')
            outputAggrCsvFileName = QDir::cleanPath(path + outputAggrCsvFileName);
    }
    projectSettings->endGroup();

    // MAPS
    projectSettings->beginGroup("maps");

    mapListFileName = projectSettings->value("map_list","").toString();
    if (! mapListFileName.isEmpty() && mapListFileName.at(0) == '.')
    {
        mapListFileName = QDir::cleanPath(path + mapListFileName);
    }

    // palette
    mapPalettePath = projectSettings->value("palette","").toString();
    if (mapPalettePath.isEmpty())
    {
        mapPalettePath = projectSettings->value("palette_path","").toString();
    }

    if (! mapPalettePath.isEmpty() && mapPalettePath.at(0) == '.')
    {
        mapPalettePath = QDir::cleanPath(path + mapPalettePath);
    }

    // file format (default = tif)
    mapFormat = projectSettings->value("format", "tif").toString();
    // map projection
    mapProjection = projectSettings->value("projection", "").toString();
    // map cell size
    mapCellSize = projectSettings->value("cellsize", "").toString();
    // map area name
    mapAreaName = projectSettings->value("area_name", "").toString();

    // make a copy in png format (default = false)
    isPngCopy = projectSettings->value("png_copy", false).toBool();
    // png projection
    pngProjection = projectSettings->value("png_projection", "").toString();

    projectSettings->endGroup();

    return true;
}


int CriteriaOutputProject::precomputeDtx()
{
    logger.writeInfo("PRECOMPUTE DTX");

    int myResult = initializeProjectDtx();
    if (myResult != CRIT1D_OK)
    {
        return myResult;
    }

    // read unit list
    logger.writeInfo("DB computational units: " + dbComputationUnitsName);
    if (! readComputationUnitList(dbComputationUnitsName, compUnitList, projectError))
    {
        return ERROR_READ_UNITS;
    }
    logger.writeInfo("Query result: " + QString::number(compUnitList.size()) + " distinct computational units.");
    logger.writeInfo("Compute dtx...");

    QString idCase;
    int step = MAXVALUE(compUnitList.size() * 0.01, 1);

    for (unsigned int i=0; i < compUnitList.size(); i++)
    {
        idCase = compUnitList[i].idCase;

        int myResult = computeAllDtxUnit(dbClimateData, idCase, projectError);
        if (myResult != CRIT1D_OK)
        {
            projectError = "ID CASE: " + idCase + "\n" + projectError;
            return myResult;
        }

        // counter
        if (i % step == 0)
        {
            int percentage = round(i * 100.0 / compUnitList.size());
            std::cout << percentage << "..";
        }
        if (i == compUnitList.size()-1)
        {
            std::cout << "100\n";
        }
    }

    return CRIT1D_OK;
}


int CriteriaOutputProject::createCsvFile()
{
    logger.writeInfo("Create CSV");

    int myResult = initializeProjectCsv();
    if (myResult != CRIT1D_OK)
    {
        return myResult;
    }

    // read unit list
    logger.writeInfo("DB computational units: " + dbComputationUnitsName);
    if (! readComputationUnitList(dbComputationUnitsName, compUnitList, projectError))
    {
        return ERROR_READ_UNITS;
    }
    logger.writeInfo("Query result: " + QString::number(compUnitList.size()) + " distinct computational units.");

    if (! initializeCsvOutputFile())
    {
        return ERROR_PARSERCSV;
    }

    logger.writeInfo("Write csv...");

    // write output
    QString idCase;
    QString idCropClass;
    int step = compUnitList.size() * 0.01;

    for (unsigned int i=0; i < compUnitList.size(); i++)
    {
        idCase = compUnitList[i].idCase;
        idCropClass = compUnitList[i].idCropClass;

        myResult = writeCsvOutputUnit(idCase, idCropClass, dbData, dbCrop, dbClimateData,
                                      dateComputation, outputVariable, outputCsvFileName, projectError);
        if (myResult != CRIT1D_OK)
        {
            if (QFile(outputCsvFileName).exists() && i == 0)
            {
                // delete empty file
                QDir().remove(outputCsvFileName);
            }
            return myResult;
        }

        // counter
        if (i % step == 0)
        {
            int percentage = round(i * 100.0 / compUnitList.size());
            std::cout << percentage << "..";
        }
        if (i == compUnitList.size()-1)
        {
            std::cout << "100\n";
        }
    }

    return CRIT1D_OK;
}


int CriteriaOutputProject::createShapeFile()
{
    if (! QFile(outputCsvFileName).exists())
    {
        // create CSV
        logger.writeInfo("Missing CSV -> createCsvFile");
        int myResult = createCsvFile();
        if (myResult != CRIT1D_OK)
        {
            return myResult;
        }
    }

    logger.writeInfo("Create SHAPEFILE");

    Crit3DShapeHandler inputShape;

    if (! inputShape.open(ucmFileName.toStdString()))
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

    return CRIT1D_OK;
}


#ifdef GDAL

int CriteriaOutputProject::createMaps()
{
    // check map list
    if (! QFile(mapListFileName).exists())
    {
        projectError = "Wrong map list: " + mapListFileName;
        return ERROR_SETTINGS_MISSINGDATA;
    }

    // check cellsize
    bool isOk;
    mapCellSize.toInt(&isOk, 10);
    if (! isOk)
    {
        projectError = "Invalid map cellsize: " + mapCellSize;
        return ERROR_SETTINGS_MISSINGDATA;
    }

    // check map format
    if (! mapExtensionShortName.contains(mapFormat))
    {
        projectError = "Unknown output format ";
        return ERROR_SETTINGS_MISSINGDATA;
    }

    // check shapefile
    if (! QFile(outputShapeFileName).exists())
    {
        int myResult = createShapeFile();
        if (myResult != CRIT1D_OK)
        {
            return myResult;
        }
    }

    logger.writeInfo("MAPS");

    QList<QString> inputFieldName;
    QList<QString> outputFileName;
    QList<QString> paletteFileName;

    // parse csv file mapListFileName
    QFile mapList(mapListFileName);
    if (! mapList.open(QFile::ReadOnly | QFile::Text) )
    {
        projectError = "Wrong map list file: " + mapListFileName;
        return ERROR_SETTINGS_MISSINGDATA;
    }

    QTextStream in(&mapList);
    // skip header
    QString line = in.readLine();

    while (! in.atEnd())
    {
        line = in.readLine();
        QList<QString> items = line.split(",");

        if (! mapPalettePath.isEmpty())
        {
            if (items.size() < 3)
            {
                projectError = "invalid line in map list:\n" + line + "\n"
                               + "Required: input field name, output file name, palette file name.";
                return ERROR_SETTINGS_MISSINGDATA;
            }
        }
        else
        {
            if (items.size() < 2)
            {
                projectError = "invalid line in map list:\n" + line + "\n"
                               + "Required: input field name, output file name.";
                return ERROR_SETTINGS_MISSINGDATA;
            }
        }

        // input field (remove whitespace)
        inputFieldName.push_back(items[0].trimmed());
        if ( inputFieldName.last().isEmpty() )
        {
            projectError = "missing shape input field in line:\n" + line;
            return ERROR_SETTINGS_MISSINGDATA;
        }

        // output file (remove whitespace)
        outputFileName.push_back(items[1].trimmed());
        if ( outputFileName.last().isEmpty() )
        {
            projectError = "missing output map name in line:\n" + line;
            return ERROR_SETTINGS_MISSINGDATA;
        }

        if (! mapPalettePath.isEmpty())
        {
            // palette file name (remove whitespace)
            paletteFileName.push_back(items[2].trimmed());
            if ( paletteFileName.last().isEmpty() )
            {
                projectError = "missing palette file name in line:\n" + line;
                return ERROR_SETTINGS_MISSINGDATA;
            }
        }
    }

    if (inputFieldName.isEmpty())
    {
        projectError = "Map list is void: " + mapListFileName;
        return ERROR_SETTINGS_MISSINGDATA;
    }

    // create png directory
    if (isPngCopy)
    {
        QString pngPath = outputShapeFilePath + "/png";
        if (! QDir(pngPath).exists())
        {
            QDir().mkdir(pngPath);
        }
    }

    // main cycle
    int nrRasterOK= 0;
    for (int i=0; i < inputFieldName.size(); i++)
    {
        QString mapFileName = outputShapeFilePath + "/" + outputFileName[i]+ "." + mapFormat;
        QString pngFileName = outputShapeFilePath + "/png/" + outputFileName[i]+ ".png";

        QString paletteCompleteFileName = "";
        if (! mapPalettePath.isEmpty())
        {
            paletteCompleteFileName = mapPalettePath + "/" + paletteFileName[i];
            if (! QFile(paletteCompleteFileName).exists())
            {
                logger.writeError("Missing palette file: " + paletteCompleteFileName);
                // skip this map
                continue;
            }
        }

        logger.writeInfo("Write map: " + mapFileName);
        if (gdalShapeToRaster(outputShapeFileName, inputFieldName[i], mapCellSize,
                              mapProjection, mapFileName, paletteCompleteFileName,
                              isPngCopy, pngFileName, pngProjection, projectError))
        {
            nrRasterOK++;
        }
        else
        {
            logger.writeError(projectError);
        }
    }

    if (nrRasterOK == inputFieldName.size())
    {
        return CRIT1D_OK;
    }
    else
    {
        int nRasterError = inputFieldName.size() - nrRasterOK;
        projectError = QString::number(nRasterError) + " invalid raster.";
        return ERROR_MAPS;
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
        projectError = "Invalid aggregation_cellsize: " + aggregationCellSize;
        return ERROR_WRONGPARAMETER;
    }

    // check aggregation threshold
    double threshold = aggregationThreshold.toDouble(&ok);
    if (!ok)
    {
        projectError = "Invalid aggregation_threshold: " + aggregationThreshold;
        return ERROR_WRONGPARAMETER;
    }
    if ((threshold < 0) || (threshold > 1))
    {
        projectError = "Invalid aggregation_threshold (must be between 0 and 1): " + aggregationThreshold;
        return ERROR_WRONGPARAMETER;
    }

    // check shapefile
    if (! QFile(outputShapeFileName).exists())
    {
        // create shapefile
        int myResult = createShapeFile();
        if (myResult != CRIT1D_OK)
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
        return ERROR_SHAPEFILE;
    }

    // check proj
    if (shapeRef.getIsWGS84() == false)
    {
        projectError = QString::fromStdString(shapeRef.getFilepath()) +  " projection error: not WGS84" ;
        return ERROR_SHAPEFILE;
    }
    if (shapeVal.getIsWGS84() == false)
    {
        projectError = QString::fromStdString(shapeVal.getFilepath()) + " projection error: not WGS84" ;
        return ERROR_SHAPEFILE;
    }

    // check utm zone
    if (shapeRef.getUtmZone() != shapeVal.getUtmZone())
    {
        projectError = "utm zone: different utm zones" ;
        return ERROR_SHAPEFILE;
    }

    // parser aggregation list
    if (!aggregationVariable.parserAggregationVariable(aggregationListFileName, projectError))
    {
        projectError = "Open failure: " + aggregationListFileName + "\n" + projectError;
        return ERROR_ZONAL_STATISTICS_SHAPE;
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
                                                aggregationVariable.inputFieldName[i].toStdString(),
                                                aggregationVariable.outputVarName[i].toStdString(),
                                                threshold, error);
        }
        else
        {
            isOk = zonalStatisticsShape(shapeRef, shapeVal, matrix, vectorNull, aggregationVariable.inputFieldName[i].toStdString(),
                                        aggregationVariable.outputVarName[i].toStdString(),
                                        aggregationVariable.aggregationType[i].toStdString(),
                                        threshold, error);
        }

        if (!isOk)
        {
            projectError = QString::fromStdString(error);
            break;
        }
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

    // write csv aggregation data
    int myResult = writeCsvAggrFromShape(shapeRef, outputAggrCsvFileName, dateComputation,
                                 aggregationVariable.outputVarName, shapeFieldName, projectError);

    shapeRef.close();

    bool reorder = true;  // enable/disable csv reorder
    if (reorder)
    {
        return orderCsvByField(outputAggrCsvFileName,"ZONE ID",projectError);
    }

    return myResult;
}


int CriteriaOutputProject::createNetcdf()
{
    // check field list
    if (fieldListFileName.isNull() || fieldListFileName.isEmpty())
    {
        projectError = "Missing 'field_list' in group [shapefile]";
        return ERROR_SETTINGS_MISSINGDATA;
    }

    // check aggregation cell size
    if (mapCellSize.isNull() || mapCellSize.isEmpty())
    {
        projectError = "Missing 'cellsize' in group [maps]";
        return ERROR_SETTINGS_MISSINGDATA;
    }
    bool isNumberOk;
    int cellSize = mapCellSize.toInt(&isNumberOk, 10);
    if (!isNumberOk)
    {
        projectError = "Invalid cellsize (it must be an integer): " + mapCellSize;
        return ERROR_SETTINGS_MISSINGDATA;
    }

    // check shapefile
    if (! QFile(outputShapeFileName).exists())
    {
        // create shapefile
        logger.writeInfo("Missing shapefile -> createShapeFile");
        int myResult = createShapeFile();
        if (myResult != CRIT1D_OK)
        {
            return myResult;
        }
    }

    logger.writeInfo("EXPORT TO NETCDF");

    Crit3DShapeHandler shapeHandler;
    if (!shapeHandler.open(outputShapeFileName.toStdString()))
    {
        projectError = "Load shapefile failed: " + outputShapeFileName;
        return ERROR_SHAPEFILE;
    }

    // read field list
    QMap<QString, QList<QString>> fieldList;
    if (! getFieldList(fieldListFileName, fieldList, projectError))
    {
        return ERROR_NETCDF;
    }

    // cycle on field list
    foreach (QList<QString> valuesList, fieldList)
    {
        QString field = valuesList[0];
        QString fileName = outputShapeFilePath + "/" + mapAreaName + "_" + field + ".nc";
        std::string variableName = field.left(4).toStdString();         // TODO inserire var name nel file
        std::string variableUnit = "mm";                                // TODO inserire var unit nel file
        Crit3DDate computationDate = getCrit3DDate(dateComputation);
        int nrDays = 28;                                                // TODO inserire var nr days nel file

        logger.writeInfo("Export file: " + fileName);
        if (! convertShapeToNetcdf(shapeHandler, fileName.toStdString(), field.toStdString(), variableName,
                                   variableUnit, cellSize, computationDate, nrDays))
        {
            projectError = "Error in export to NetCDF: " + projectError;
            return ERROR_NETCDF;
        }
    }

    return CRIT1D_OK;
}


bool CriteriaOutputProject::convertShapeToNetcdf(Crit3DShapeHandler &shapeHandler, std::string outputFileName,
                                                 std::string field, std::string variableName, std::string variableUnit, double cellSize,
                                                 const Crit3DDate &computationDate, int nrDays)
{
    if (! shapeHandler.getIsWGS84())
    {
        projectError = "Shapefile is not WGS84.";
        return false;
    }

    // rasterize shape
    gis::Crit3DRasterGrid tmpRaster;
    if (! rasterizeShape(shapeHandler, tmpRaster, field, cellSize))
    {
        projectError = "Error in rasterize shape.";
        return false;
    }

    // set UTM zone and emisphere
    gis::Crit3DGisSettings gisSettings;
    gisSettings.utmZone = shapeHandler.getUtmZone();
    double sign = 1;
    if (! shapeHandler.getIsNorth()) sign = -1;
    gisSettings.startLocation.latitude = sign * abs(gisSettings.startLocation.latitude);

    // convert to lat lon raster
    gis::Crit3DLatLonHeader latLonHeader;
    gis::getGeoExtentsFromUTMHeader(gisSettings, tmpRaster.header, &latLonHeader);

    // initialize data raster (only for values)
    gis::Crit3DRasterGrid dataRaster;
    dataRaster.header->nrRows = latLonHeader.nrRows;
    dataRaster.header->nrCols = latLonHeader.nrCols;
    dataRaster.header->flag = latLonHeader.flag;
    dataRaster.header->llCorner.y = latLonHeader.llCorner.latitude;
    dataRaster.header->llCorner.x = latLonHeader.llCorner.longitude;
    dataRaster.header->cellSize = (latLonHeader.dx + latLonHeader.dy) * 0.5;
    dataRaster.initializeGrid(latLonHeader.flag);

    // assign lat lon values
    double lat, lon, x, y;
    int utmRow, utmCol;
    for (int row = 0; row < latLonHeader.nrRows; row++)
    {
        for (int col = 0; col < latLonHeader.nrCols; col++)
        {
            gis::getLatLonFromRowCol(latLonHeader, row, col, &lat, &lon);
            gis::latLonToUtmForceZone(gisSettings.utmZone, lat, lon, &x, &y);
            if (! gis::isOutOfGridXY(x, y, tmpRaster.header))
            {
                gis::getRowColFromXY(*(tmpRaster.header), x, y, &utmRow, &utmCol);
                float value = tmpRaster.getValueFromRowCol(utmRow, utmCol);
                if (int(value) != int(tmpRaster.header->flag))
                {
                    dataRaster.value[row][col] = value;
                }
            }
        }
    }

    // create netcdf
    NetCDFHandler myNetCDF;
    myNetCDF.createNewFile(outputFileName);

    std::string title = projectName.toStdString();

    if (! myNetCDF.writeMetadata(latLonHeader, title, variableName, variableUnit,
                                computationDate, nrDays, NODATA, NODATA))
    {
        projectError = "Error in write metadata to netcdf.";
        myNetCDF.close();
        return false;
    }

    if (! myNetCDF.writeData_NoTime(dataRaster))
    {
        projectError = "Error in write data to netcdf.";
        myNetCDF.close();
        return false;
    }

    myNetCDF.close();

    return true;
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

bool CriteriaOutputProject::getAllDbVariable()
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

    // read tables
    QList<QString> tablesList = dbData.tables();
    if (tablesList.isEmpty())
    {
        projectError = "Db is empty";
        return false;
    }

    // read table_info
    QString tableName = tablesList[0];
    QString statement = QString("PRAGMA table_info(`%1`)").arg(tableName);
    QSqlQuery qry(dbData);
    if( !qry.exec(statement) )
    {
        projectError = qry.lastError().text();
        return false;
    }

    // read fields
    QString fieldName;
    QList<QString> varList;
    qry.first();
    if (!qry.isValid())
    {
        projectError = qry.lastError().text();
        return false;
    }
    do
    {
        getValue(qry.value("name"), &fieldName);
        if (fieldName != "DATE")
        {
            varList<<fieldName;
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

bool CriteriaOutputProject::getDbDataDates(QDate &firstDate, QDate &lastDate)
{
    QList<QString> tablesList = dbData.tables();
    if (tablesList.isEmpty())
    {
        projectError = "Db is empty";
        return false;
    }

    QDate firstTmp;
    QDate lastTmp;

    firstDate = QDate::currentDate();
    lastDate = QDate(1800,1,1);

    QString idCase = tablesList[0];
    QString statement = QString("SELECT MIN(DATE),MAX(DATE) FROM `%1`").arg(idCase);
    QSqlQuery qry(dbData);
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

    if (firstTmp < firstDate)
    {
        firstDate = firstTmp;
    }
    if (lastTmp > lastDate)
    {
        lastDate = lastTmp;
    }

    if (!firstDate.isValid() || !lastDate.isValid())
    {
        projectError = "Invalid date";
        return false;
    }

    return true;
}


int CriteriaOutputProject::createCsvFileFromGUI(const QDate &dateComputation, const QString &csvFileName)
{
    int myResult = initializeProjectCsv();
    if (myResult != CRIT1D_OK)
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
    if (! readComputationUnitList(dbComputationUnitsName, compUnitList, projectError))
    {
        return ERROR_READ_UNITS;
    }

    // write output
    QString idCase;
    QString idCropClass;
    for (unsigned int i=0; i < compUnitList.size(); i++)
    {
        idCase = compUnitList[i].idCase;
        idCropClass = compUnitList[i].idCropClass;

        myResult = writeCsvOutputUnit(idCase, idCropClass, dbData, dbCrop, dbClimateData, dateComputation, outputVariable, csvFileName, projectError);
        if (myResult != CRIT1D_OK)
        {
            if (QFile(csvFileName).exists() && i == 0)
            {
                // delete empty file
                QDir().remove(csvFileName);
            }
            return myResult;
        }
    }
    return CRIT1D_OK;
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

    return CRIT1D_OK;
}
