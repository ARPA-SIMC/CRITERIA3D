#ifndef CSVTOSHAPEPROJECT_H
#define CSVTOSHAPEPROJECT_H

#include <QString>
#include <fstream>
#include <QSqlDatabase>
#include "logger.h"
#include "criteria1DUnit.h"
#include "criteriaOutputVariable.h"

#define ERROR_SETTINGS_MISSING -1
#define ERROR_SETTINGS_WRONGFILENAME -2
#define ERROR_SETTINGS_MISSINGDATA -3
#define ERROR_DBPARAMETERS -5
#define ERROR_DBHISTORICAL -6
#define ERROR_DBOUTPUT -10
#define ERROR_WRONGDATE -11
#define ERROR_PARSERCSV -12
#define ERROR_READ_UNITS -15

#define ERROR_WRITECSV -50
#define ERROR_OUTPUT_VARIABLES -60
#define ERROR_MISSING_DATA -70
#define ERROR_INCOMPLETE_DATA -80
#define ERROR_SHAPEFILE -100


class CriteriaOutputProject
{
public:
    bool isProjectLoaded;
    bool isCsv;

    QString path;
    QString dataPath;
    QString projectName;
    QString configFileName;
    QString projectError;
    QString ucmFileName;
    QString shapeFileName;
    QString fieldListFileName;
    QString variableListFileName;
    QString csvFileName;

    QDate dateComputation;

    QString dbUnitsName;
    QString dbDataName;
    QString dbCropName;
    QString dbDataHistoricalName;

    QSqlDatabase dbCrop;
    QSqlDatabase dbData;
    QSqlDatabase dbDataHistorical;

    int nrUnits;
    std::vector<Crit1DUnit> unitList;
    CriteriaOutputVariable outputVariable;

    QFile outputFile;
    QString logFileName;
    std::ofstream logFile;
    Logger logger;

    CriteriaOutputProject();

    void initialize();
    void closeProject();
    int initializeProject(QString settingsFileName, QDate dateComputation, bool isCsv);
    int initializeProjectCsv();
    bool readSettings();

    int createCsvFile();
    int createShapeFile();
    int createAggregationFile();

    bool initializeCsvOutputFile();
    int writeCsvOutputUnit(unsigned int unitIndex);

    int selectSimpleVar(QSqlDatabase db, QString idCase, QString varName, QString computation, QDate firstDate, QDate lastDate, float irriRatio, QVector<float> *resVector);
    int computeDTX(QSqlDatabase db, QString idCase, int period, QString computation, QDate firstDate, QDate lastDate, QVector<float>* resVector);

};


#endif // CSVTOSHAPEPROJECT_H
