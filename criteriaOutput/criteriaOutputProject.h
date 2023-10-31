#ifndef CRITERIAOUTPUTPROJECT_H
#define CRITERIAOUTPUTPROJECT_H

#include <QString>
#include <QDate>
#include <fstream>
#include <QSqlDatabase>

#include "logger.h"
#include "criteriaOutputVariable.h"
#include "criteriaAggregationVariable.h"
#include "computationUnitsDb.h"
#include "shapeHandler.h"
#include "crit3dDate.h"

#define REQUIREDMAPLISTCSVINFO 3

#define ERROR_MISSINGPARAMETERS -900
#define ERROR_WRONGPARAMETER -901

#define ERROR_SETTINGS_MISSING -1
#define ERROR_SETTINGS_WRONGFILENAME -2
#define ERROR_SETTINGS_MISSINGDATA -3
#define ERROR_DBPARAMETERS -5
#define ERROR_DBCLIMATE -6
#define ERROR_TDXWRITE -7
#define ERROR_DBOUTPUT -10
#define ERROR_WRONGDATE -11
#define ERROR_PARSERCSV -12
#define ERROR_READ_UNITS -15

#define ERROR_DB_MISSING_DATA -20
#define ERROR_DB_INCOMPLETE_DATA -21
#define ERROR_DB_MISSING_PRECOMPUTED_DTX -22

#define ERROR_WRITECSV -50
#define ERROR_OUTPUT_VARIABLES -60
#define ERROR_CSVFILE -65
#define ERROR_SHAPEFILE -70
#define ERROR_NETCDF -75
#define ERROR_ZONAL_STATISTICS_SHAPE -80
#define ERROR_MAPS -85
#define ERROR_MISSING_GDAL -100


class CriteriaOutputProject
{
public:
    bool isProjectLoaded;

    QString path;
    QString projectName;
    QString operation;
    QString configFileName;
    QString projectError;
    QString ucmFileName;
    QString fieldListFileName;
    QString variableListFileName;
    QString aggregationShapeFileName;
    QString shapeFieldName;
    QString aggregationListFileName;
    QString aggregationCellSize;
    QString aggregationThreshold;

    QString mapListFileName;
    QString mapPalettePath;
    QString mapCellSize;
    QString mapFormat;
    QString mapProjection;
    QString mapAreaName;

    QString outputCsvFileName;
    QString outputShapeFileName;
    QString outputShapeFilePath;
    QString outputAggrCsvFileName;

    QDate dateComputation;

    QString dbComputationUnitsName;
    QString dbDataName;
    QString dbCropName;
    QString dbClimateDataName;

    QSqlDatabase dbCrop;
    QSqlDatabase dbData;
    QSqlDatabase dbClimateData;

    int nrUnits;
    std::vector<Crit1DCompUnit> compUnitList;
    CriteriaOutputVariable outputVariable;
    CriteriaAggregationVariable aggregationVariable;

    QFile outputFile;
    QString logFileName;
    std::ofstream logFile;
    Logger logger;
    bool addDateTimeLogFile;

    CriteriaOutputProject();

    void initialize();
    void closeProject();

    int initializeProject(QString settingsFileName, QString operation, QDate dateComputation, bool isLog);
    int initializeProjectDtx();
    int initializeProjectCsv();

    bool readSettings();

    int precomputeDtx();
    int createCsvFile();
    int createShapeFile();
    int createAggregationFile();
    int createNetcdf();
    int createMaps();

    bool initializeCsvOutputFile();
    bool getAllDbVariable();
    bool getDbDataDates(QDate &firstDate, QDate &lastDate);
    int createCsvFileFromGUI(QDate dateComputation, QString csvFileName);
    int createShapeFileFromGUI();
    bool convertShapeToNetcdf(Crit3DShapeHandler &shape, std::string outputFileName,
                              std::string field, std::string variableName, std::string variableUnit, double cellSize,
                              Crit3DDate computationDate, int nrDays);

};


#endif // CRITERIAOUTPUTPROJECT_H
