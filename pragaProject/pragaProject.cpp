#include "commonConstants.h"
#include "basicMath.h"
#include "climate.h"
#include "crit3dElabList.h"
#include "crit3dDroughtList.h"
#include "dbClimate.h"
#include "download.h"
#include "dbAggregationsHandler.h"
#include "utilities.h"
#include "project.h"
#include "aggregation.h"
#include "interpolationCmd.h"
#include "interpolation.h"
#include "pragaProject.h"
#include <qdebug.h>
#include <QFile>
#include <QDir>
#include <QtSql>

PragaProject::PragaProject()
{
    initializePragaProject();
}

bool PragaProject::getIsElabMeteoPointsValue() const
{
    return isElabMeteoPointsValue;
}

void PragaProject::setIsElabMeteoPointsValue(bool value)
{
    isElabMeteoPointsValue = value;
}

void PragaProject::initializePragaProject()
{
    clima = new Crit3DClimate();
    climaFromDb = nullptr;
    referenceClima = nullptr;
    synchronicityWidget = nullptr;
    synchReferencePoint = "";
    pragaDefaultSettings = nullptr;
    pragaDailyMaps = nullptr;
    users.clear();
    lastElabTargetisGrid = false;

    outputMeteoPointsDbHandler = nullptr;
    outputMeteoPointsLoaded = false;
}

void PragaProject::clearPragaProject()
{
    if (isProjectLoaded) clearProject();

    closeOutputMeteoPointsDB();

    users.clear();

    dataRaster.clear();

    if (clima != nullptr)
    {
        delete clima;
        clima = nullptr;
    }

    if (pragaDailyMaps != nullptr)
    {
        delete pragaDailyMaps;
        pragaDailyMaps = nullptr;
    }
}

void PragaProject::createPragaProject(QString path_, QString name_, QString description_)
{
    createProject(path_, name_, description_);
    savePragaParameters();
}

void PragaProject::savePragaProject()
{
    saveProject();
    savePragaParameters();
}

bool PragaProject::loadPragaProject(QString myFileName)
{
    clearPragaProject();
    initializeProject();
    initializePragaProject();

    if (myFileName == "") return(false);

    projectPragaFolder = QFileInfo(myFileName).absolutePath();

    if (! loadProjectSettings(myFileName))
        return false;

    if (! loadProject())
        return false;

    if (! loadPragaSettings())
        return false;

    if (DEM.isLoaded)
    {
        pragaDailyMaps = new Crit3DDailyMeteoMaps(DEM);
        pragaHourlyMaps = new PragaHourlyMeteoMaps(DEM);
    }

    isProjectLoaded = true;

    if (projectName != "")
    {
        logInfo("Project " + projectName + " loaded");
    }
    return true;
}


bool PragaProject::loadPragaSettings()
{
    pragaDefaultSettings = new QSettings(getDefaultPath() + PATH_SETTINGS + "pragaDefault.ini", QSettings::IniFormat);

    Q_FOREACH (QString group, parametersSettings->childGroups())
    {
        if (group == "elaboration")
        {
            parametersSettings->beginGroup(group);
            Crit3DElaborationSettings* elabSettings = clima->getElabSettings();

            if (parametersSettings->contains("anomaly_pts_max_distance") && !parametersSettings->value("anomaly_pts_max_distance").toString().isEmpty())
            {
                elabSettings->setAnomalyPtsMaxDistance(parametersSettings->value("anomaly_pts_max_distance").toFloat());
            }
            if (parametersSettings->contains("anomaly_pts_max_delta_z") && !parametersSettings->value("anomaly_pts_max_delta_z").toString().isEmpty())
            {
                elabSettings->setAnomalyPtsMaxDeltaZ(parametersSettings->value("anomaly_pts_max_delta_z").toFloat());
            }
            if (parametersSettings->contains("grid_min_coverage") && !parametersSettings->value("grid_min_coverage").toString().isEmpty())
            {
                elabSettings->setGridMinCoverage(parametersSettings->value("grid_min_coverage").toFloat());
            }
            if (parametersSettings->contains("merge_joint_stations") && !parametersSettings->value("merge_joint_stations").toString().isEmpty())
            {
                elabSettings->setMergeJointStations(parametersSettings->value("merge_joint_stations").toBool());
            }

            parametersSettings->endGroup();

        }
        if (group == "quality")
        {
            parametersSettings->beginGroup(group);

            if (parametersSettings->contains("users"))
            {
                users = parametersSettings->value("users").toStringList();
            }

            parametersSettings->endGroup();

        }
        else if (group == "id_arkimet")
        {
            parametersSettings->beginGroup(group);
            QList<QString> myList;
            QList<int> intList;
            if ( parametersSettings->contains(QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyAirTemperatureAvg))) )
            {
                intList.clear();
                QString dailyTavg = QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyAirTemperatureAvg));
                myList = parametersSettings->value(dailyTavg).toStringList();
                for (int i = 0; i < myList.size(); i++)
                {
                    if (myList[i].toInt() > 0 && !intList.contains(myList[i].toInt()))
                    {
                        intList << myList[i].toInt();
                    }
                }
                idArkimetDailyMap[dailyTavg] = intList;
            }
            if ( parametersSettings->contains(QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyAirTemperatureMax))) )
            {
                intList.clear();
                QString dailyTmax = QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyAirTemperatureMax));
                myList = parametersSettings->value(dailyTmax).toStringList();
                for (int i = 0; i < myList.size(); i++)
                {
                    if (myList[i].toInt() > 0 && !intList.contains(myList[i].toInt()))
                    {
                        intList << myList[i].toInt();
                    }
                }
                idArkimetDailyMap[dailyTmax] = intList;
            }
            if ( parametersSettings->contains(QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyAirTemperatureMin))) )
            {
                intList.clear();
                QString dailyTmin = QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyAirTemperatureMin));
                myList = parametersSettings->value(dailyTmin).toStringList();
                for (int i = 0; i < myList.size(); i++)
                {
                    if (myList[i].toInt() > 0 && !intList.contains(myList[i].toInt()))
                    {
                        intList << myList[i].toInt();
                    }
                }
                idArkimetDailyMap[dailyTmin] = intList;
            }
            if ( parametersSettings->contains(QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyPrecipitation))) )
            {
                intList.clear();
                QString dailyPrec = QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyPrecipitation));
                myList = parametersSettings->value(dailyPrec).toStringList();
                for (int i = 0; i < myList.size(); i++)
                {
                    if (myList[i].toInt() > 0 && !intList.contains(myList[i].toInt()))
                    {
                        intList << myList[i].toInt();
                    }
                }
                idArkimetDailyMap[dailyPrec] = intList;
            }
            if ( parametersSettings->contains(QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyAirRelHumidityAvg))) )
            {
                intList.clear();
                QString dailRHavg = QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyAirRelHumidityAvg));
                myList = parametersSettings->value(dailRHavg).toStringList();
                for (int i = 0; i < myList.size(); i++)
                {
                    if (myList[i].toInt() > 0 && !intList.contains(myList[i].toInt()))
                    {
                        intList << myList[i].toInt();
                    }
                }
                idArkimetDailyMap[dailRHavg] = intList;
            }
            if ( parametersSettings->contains(QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyAirRelHumidityMax))) )
            {
                intList.clear();
                QString dailRHmax = QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyAirRelHumidityMax));
                myList = parametersSettings->value(dailRHmax).toStringList();
                for (int i = 0; i < myList.size(); i++)
                {
                    if (myList[i].toInt() > 0 && !intList.contains(myList[i].toInt()))
                    {
                        intList << myList[i].toInt();
                    }
                }
                idArkimetDailyMap[dailRHmax] = intList;
            }
            if ( parametersSettings->contains(QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyAirRelHumidityMin))) )
            {
                intList.clear();
                QString dailRHmin = QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyAirRelHumidityMin));
                myList = parametersSettings->value(dailRHmin).toStringList();
                for (int i = 0; i < myList.size(); i++)
                {
                    if (myList[i].toInt() > 0 && !intList.contains(myList[i].toInt()))
                    {
                        intList << myList[i].toInt();
                    }
                }
                idArkimetDailyMap[dailRHmin] = intList;
            }
            if (parametersSettings->contains(QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyGlobalRadiation))))
            {
                intList.clear();
                QString dailyRad = QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyGlobalRadiation));
                myList = parametersSettings->value(dailyRad).toStringList();
                for (int i = 0; i < myList.size(); i++)
                {
                    if (myList[i].toInt() > 0 && !intList.contains(myList[i].toInt()))
                    {
                        intList << myList[i].toInt();
                    }
                }
                idArkimetDailyMap[dailyRad] = intList;
            }
            if ( parametersSettings->contains(QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyWindScalarIntensityAvg))) )
            {
                intList.clear();
                QString dailyWindScalIntAvg = QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyWindScalarIntensityAvg));
                myList = parametersSettings->value(dailyWindScalIntAvg).toStringList();
                for (int i = 0; i < myList.size(); i++)
                {
                    if (myList[i].toInt() > 0 && !intList.contains(myList[i].toInt()))
                    {
                        intList << myList[i].toInt();
                    }
                }
                idArkimetDailyMap[dailyWindScalIntAvg] = intList;
            }
            if ( parametersSettings->contains(QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyWindScalarIntensityMax))) )
            {
                intList.clear();
                QString dailyWindScalIntMax = QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyWindScalarIntensityMax));
                myList = parametersSettings->value(dailyWindScalIntMax).toStringList();
                for (int i = 0; i < myList.size(); i++)
                {
                    if (myList[i].toInt() > 0 && !intList.contains(myList[i].toInt()))
                    {
                        intList << myList[i].toInt();
                    }
                }
                idArkimetDailyMap[dailyWindScalIntMax] = intList;
            }
            if ( parametersSettings->contains(QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyWindVectorDirectionPrevailing))) )
            {
                intList.clear();
                QString dailyWindVecDirPrev = QString::fromStdString(getKeyStringMeteoMap(MapDailyMeteoVar, dailyWindVectorDirectionPrevailing));
                myList = parametersSettings->value(dailyWindVecDirPrev).toStringList();
                for (int i = 0; i < myList.size(); i++)
                {
                    if (myList[i].toInt() > 0 && !intList.contains(myList[i].toInt()))
                    {
                        intList << myList[i].toInt();
                    }
                }
                idArkimetDailyMap[dailyWindVecDirPrev] = intList;
            }
            if (parametersSettings->contains(QString::fromStdString(getKeyStringMeteoMap(MapHourlyMeteoVar, airTemperature))))
            {
                intList.clear();
                QString airTemp = QString::fromStdString(getKeyStringMeteoMap(MapHourlyMeteoVar, airTemperature));
                myList = parametersSettings->value(airTemp).toStringList();
                for (int i = 0; i < myList.size(); i++)
                {
                    if (myList[i].toInt() > 0 && !intList.contains(myList[i].toInt()))
                    {
                        intList << myList[i].toInt();
                    }
                }
                idArkimetHourlyMap[airTemp] = intList;
            }
            if (parametersSettings->contains(QString::fromStdString(getKeyStringMeteoMap(MapHourlyMeteoVar, precipitation))))
            {
                intList.clear();
                QString prec = QString::fromStdString(getKeyStringMeteoMap(MapHourlyMeteoVar, precipitation));
                myList = parametersSettings->value(prec).toStringList();
                for (int i = 0; i < myList.size(); i++)
                {
                    if (myList[i].toInt() > 0 && !intList.contains(myList[i].toInt()))
                    {
                        intList << myList[i].toInt();
                    }
                }
                idArkimetHourlyMap[prec] = intList;
            }
            if (parametersSettings->contains(QString::fromStdString(getKeyStringMeteoMap(MapHourlyMeteoVar, airRelHumidity))))
            {
                intList.clear();
                QString airRelH = QString::fromStdString(getKeyStringMeteoMap(MapHourlyMeteoVar, airRelHumidity));
                myList = parametersSettings->value(airRelH).toStringList();
                for (int i = 0; i < myList.size(); i++)
                {
                    if (myList[i].toInt() > 0 && !intList.contains(myList[i].toInt()))
                    {
                        intList << myList[i].toInt();
                    }
                }
                idArkimetHourlyMap[airRelH] = intList;
            }
            if (parametersSettings->contains(QString::fromStdString(getKeyStringMeteoMap(MapHourlyMeteoVar, globalIrradiance))))
            {
                intList.clear();
                QString globalIrr = QString::fromStdString(getKeyStringMeteoMap(MapHourlyMeteoVar, globalIrradiance));
                myList = parametersSettings->value(globalIrr).toStringList();
                for (int i = 0; i < myList.size(); i++)
                {
                    if (myList[i].toInt() > 0 && !intList.contains(myList[i].toInt()))
                    {
                        intList << myList[i].toInt();
                    }
                }
                idArkimetHourlyMap[globalIrr] = intList;
            }
            if (parametersSettings->contains(QString::fromStdString(getKeyStringMeteoMap(MapHourlyMeteoVar, windScalarIntensity))))
            {
                intList.clear();
                QString windScaInt = QString::fromStdString(getKeyStringMeteoMap(MapHourlyMeteoVar, windScalarIntensity));
                myList = parametersSettings->value(windScaInt).toStringList();
                for (int i = 0; i < myList.size(); i++)
                {
                    if (myList[i].toInt() > 0 && !intList.contains(myList[i].toInt()))
                    {
                        intList << myList[i].toInt();
                    }
                }
                idArkimetHourlyMap[windScaInt] = intList;
            }
            if (parametersSettings->contains(QString::fromStdString(getKeyStringMeteoMap(MapHourlyMeteoVar, windVectorDirection))))
            {
                intList.clear();
                QString windVecDir = QString::fromStdString(getKeyStringMeteoMap(MapHourlyMeteoVar, windVectorDirection));
                myList = parametersSettings->value(windVecDir).toStringList();
                for (int i = 0; i < myList.size(); i++)
                {
                    if (myList[i].toInt() > 0 && !intList.contains(myList[i].toInt()))
                    {
                        intList << myList[i].toInt();
                    }
                }
                idArkimetHourlyMap[windVecDir] = intList;
            }

/*
            for(std::map<QString, QList<int> >::const_iterator it = idArkimetDailyMap.begin();
                it != idArkimetDailyMap.end(); ++it)
            {
                qDebug() << "idArkimetDailyMap " << it->first << ":" << it->second << "\n";
            }
            for(std::map<QString, QList<int> >::const_iterator it = idArkimetHourlyMap.begin();
                it != idArkimetHourlyMap.end(); ++it)
            {
                qDebug() << "idArkimetHourlyMap " << it->first << ":" << it->second << "\n";
            }

*/
            parametersSettings->endGroup();

        }

    }

    return true;
}


bool PragaProject::saveGrid(meteoVariable myVar, frequencyType myFrequency, const Crit3DTime& myTime, bool showInfo)
{
    std::string id;
    int infoStep = 1;

    if (myFrequency == daily)
    {
        if (showInfo)
        {
            QString infoStr = "Save meteo grid daily data";
            infoStep = setProgressBar(infoStr, this->meteoGridDbHandler->gridStructure().header().nrRows);
        }

        for (int row = 0; row < this->meteoGridDbHandler->gridStructure().header().nrRows; row++)
        {
            if (showInfo)
            {
                if ((row % infoStep) == 0)
                    updateProgressBar(row);
            }
            for (int col = 0; col < this->meteoGridDbHandler->gridStructure().header().nrCols; col++)
            {
                if (this->meteoGridDbHandler->meteoGrid()->getMeteoPointActiveId(row, col, &id))
                {
                    if (!this->meteoGridDbHandler->gridStructure().isFixedFields())
                    {
                        this->meteoGridDbHandler->saveCellCurrentGridDaily(&errorString, QString::fromStdString(id), QDate(myTime.date.year, myTime.date.month, myTime.date.day), this->meteoGridDbHandler->getDailyVarCode(myVar), this->meteoGridDbHandler->meteoGrid()->meteoPoint(row,col).currentValue);
                    }
                    else
                    {
                        this->meteoGridDbHandler->saveCellCurrentGridDailyFF(errorString, QString::fromStdString(id), QDate(myTime.date.year, myTime.date.month, myTime.date.day), QString::fromStdString(this->meteoGridDbHandler->getDailyPragaName(myVar)), this->meteoGridDbHandler->meteoGrid()->meteoPoint(row,col).currentValue);
                    }
                }
            }
        }
    }
    else if (myFrequency == hourly)
    {
        if (showInfo)
        {
            QString infoStr = "Save meteo grid hourly data";
            infoStep = setProgressBar(infoStr, this->meteoGridDbHandler->gridStructure().header().nrRows);
        }

        for (int row = 0; row < this->meteoGridDbHandler->gridStructure().header().nrRows; row++)
        {
            if (showInfo && (row % infoStep) == 0)
                updateProgressBar(row);
            for (int col = 0; col < this->meteoGridDbHandler->gridStructure().header().nrCols; col++)
            {
                if (this->meteoGridDbHandler->meteoGrid()->getMeteoPointActiveId(row, col, &id))
                {
                    if (!this->meteoGridDbHandler->gridStructure().isFixedFields())
                    {
                        this->meteoGridDbHandler->saveCellCurrentGridHourly(errorString, QString::fromStdString(id), QDateTime(QDate(myTime.date.year, myTime.date.month, myTime.date.day), QTime(myTime.getHour(), myTime.getMinutes(), myTime.getSeconds()), Qt::UTC), this->meteoGridDbHandler->getHourlyVarCode(myVar), this->meteoGridDbHandler->meteoGrid()->meteoPoint(row,col).currentValue);
                    }
                    else
                    {
                        this->meteoGridDbHandler->saveCellCurrentGridHourlyFF(errorString, QString::fromStdString(id), QDateTime(QDate(myTime.date.year, myTime.date.month, myTime.date.day), QTime(myTime.getHour(), myTime.getMinutes(), myTime.getSeconds()), Qt::UTC), QString::fromStdString(this->meteoGridDbHandler->getHourlyPragaName(myVar)), this->meteoGridDbHandler->meteoGrid()->meteoPoint(row,col).currentValue);
                    }
                }
            }
        }
    }

    if (showInfo) closeProgressBar();

    return true;
}

bool PragaProject::elaborationCheck(bool isMeteoGrid, bool isAnomaly)
{

    if (isMeteoGrid)
    {
        if (this->meteoGridDbHandler == nullptr)
        {
            errorString = "Load grid";
            return false;
        }
        else
        {
            if (this->clima == nullptr)
            {
                this->clima = new Crit3DClimate();
            }
            clima->setDb(this->meteoGridDbHandler->db());
        }
    }
    else
    {
        if (this->meteoPointsDbHandler == nullptr)
        {
            errorString = "Load meteo Points";
            return false;
        }
        else
        {
            if (this->clima == nullptr)
            {
                this->clima = new Crit3DClimate();
            }
            clima->setDb(this->meteoPointsDbHandler->getDb());
        }
    }
    if (isAnomaly)
    {
        if (this->referenceClima == nullptr)
        {
            this->referenceClima = new Crit3DClimate();
        }
        if (isMeteoGrid)
        {
            referenceClima->setDb(this->meteoGridDbHandler->db());
        }
        else
        {
            referenceClima->setDb(this->meteoPointsDbHandler->getDb());
        }
    }

    return true;
}

bool PragaProject::showClimateFields(bool isMeteoGrid, QList<QString>* climateDbElab, QList<QString>* climateDbVarList)
{
    QSqlDatabase db;
    if (isMeteoGrid)
    {
        if (this->meteoGridDbHandler == nullptr)
        {
            errorString = "Load grid";
            return false;
        }
        db = this->meteoGridDbHandler->db();
    }
    else
    {
        if (this->meteoPointsDbHandler == nullptr)
        {
            errorString = "Load meteo Points";
            return false;
        }
        db = this->meteoPointsDbHandler->getDb();
    }
    QList<QString> climateTables;

    if ( !getClimateTables(db, &errorString, &climateTables) )
    {
        errorString = "No climate tables";
        return false;
    }
    else
    {
        for (int i=0; i < climateTables.size(); i++)
        {
            getClimateFieldsFromTable(db, &errorString, climateTables[i], climateDbElab);
        }
        if (climateDbElab->isEmpty())
        {
            errorString = "Empty climate tables";
            return false;
        }
    }
    for (int i=0; i < climateDbElab->size(); i++)
    {
        QString elab = climateDbElab->at(i);
        QList<QString> words = elab.split('_');
        QString var = words[1];
        if (!climateDbVarList->contains(var))
        {
            climateDbVarList->append(var);
        }
    }
    return true;

}

void PragaProject::readClimate(bool isMeteoGrid, QString climateSelected, int climateIndex, bool showInfo)
{

    int infoStep = 0;
    QString infoStr;

    QSqlDatabase db;
    float result;

    Crit3DClimateList climateList;
    QList<QString> climate;
    climate.push_back(climateSelected);

    climateList.setListClimateElab(climate);

    errorString = "";
    climateList.parserElaboration(errorString);
    if (! errorString.isEmpty())
    {
        logError();
        errorString = "";
    }

    // copy elaboration to clima
    clima->setYearStart(climateList.listYearStart().at(0));
    clima->setYearEnd(climateList.listYearEnd().at(0));
    clima->setPeriodType(climateList.listPeriodType().at(0));
    clima->setPeriodStr(climateList.listPeriodStr().at(0));
    clima->setGenericPeriodDateStart(climateList.listGenericPeriodDateStart().at(0));
    clima->setGenericPeriodDateEnd(climateList.listGenericPeriodDateEnd().at(0));
    clima->setNYears(climateList.listNYears().at(0));
    clima->setVariable(climateList.listVariable().at(0));
    clima->setElab1(climateList.listElab1().at(0));
    clima->setElab2(climateList.listElab2().at(0));
    clima->setParam1(climateList.listParam1().at(0));
    clima->setParam2(climateList.listParam2().at(0));
    clima->setParam1IsClimate(climateList.listParam1IsClimate().at(0));
    clima->setParam1ClimateField(climateList.listParam1ClimateField().at(0));

    QString table;
    if (clima->periodType() == genericPeriod)
        table = "climate_generic";
    else
        table = "climate_" + climateList.listPeriodStr().at(0);

    if (isMeteoGrid)
    {
        if (showInfo)
        {
            infoStr = "Read Climate - Meteo Grid";
            infoStep = setProgressBar(infoStr, this->meteoGridDbHandler->gridStructure().header().nrRows);
        }
        std::string id;
        db = this->meteoGridDbHandler->db();
        for (int row = 0; row < meteoGridDbHandler->gridStructure().header().nrRows; row++)
        {
            if (showInfo && (row % infoStep) == 0 )
            {
                 updateProgressBar(row);
            }
            for (int col = 0; col < meteoGridDbHandler->gridStructure().header().nrCols; col++)
            {
                if (meteoGridDbHandler->meteoGrid()->getMeteoPointActiveId(row, col, &id))
                {
                    Crit3DMeteoPoint* meteoPoint = meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col);
                    if (climateIndex != NODATA)
                    {
                        result = readClimateElab(db, table.toLower(), climateIndex, QString::fromStdString(meteoPoint->id), climateSelected, &errorString);
                        meteoPoint->climate = result;
                    }
                    else
                    {
                        meteoPoint->climate = NODATA;
                    }
                }
             }
        }
        meteoGridDbHandler->meteoGrid()->setIsElabValue(true);
    }
    else
    {
        if (showInfo)
        {
            infoStr = "Read Climate - Meteo Points";
            infoStep = setProgressBar(infoStr, nrMeteoPoints);
        }
        db = this->meteoPointsDbHandler->getDb();
        for (int i = 0; i < nrMeteoPoints; i++)
        {
            if (meteoPoints[i].active)
            {
                if (showInfo && (i % infoStep) == 0)
                {
                    updateProgressBar(i);
                }
                QString id = QString::fromStdString(meteoPoints[i].id);
                if (climateIndex == NODATA)
                {
                    errorString = "climate index error";
                    meteoPoints[i].climate = NODATA;
                }
                else
                {
                    result = readClimateElab(db, table.toLower(), climateIndex, id, climateSelected, &errorString);
                    meteoPoints[i].climate = result;
                }
            }
        }
        setIsElabMeteoPointsValue(true);

    }
    if (showInfo) closeProgressBar();
}

bool PragaProject::deleteClimate(bool isMeteoGrid, QString climaSelected)
{
    QSqlDatabase db;

    QList<QString> words = climaSelected.split('_');
    QString period = words[2];
    QString table = "climate_" + period;

    if (isMeteoGrid)
    {
        db = this->meteoGridDbHandler->db();
    }
    else
    {
        db = this->meteoPointsDbHandler->getDb();
    }

    return deleteElab(db, &errorString, table.toLower(), climaSelected);
}


bool PragaProject::elaboration(bool isMeteoGrid, bool isAnomaly, bool saveClima)
{
    if (isMeteoGrid)
    {
        if (saveClima)
        {
            if (!climatePointsCycleGrid(true))
            {
                return false;
            }
            else
            {
                return true;
            }
        }
        if (!elaborationPointsCycleGrid(isAnomaly, true))
        {
            return false;
        }
        meteoGridDbHandler->meteoGrid()->setIsElabValue(true);
    }
    else
    {
        if (saveClima)
        {
            if (!climatePointsCycle(true))
            {
                return false;
            }
            else
            {
                return true;
            }
        }
        if (!elaborationPointsCycle(isAnomaly, true))
        {
            return false;
        }

        setIsElabMeteoPointsValue(true);
    }

    return true;
}


bool PragaProject::elaborationPointsCycle(bool isAnomaly, bool showInfo)
{
    // initialize
    for (int i = 0; i < nrMeteoPoints; i++)
    {
        if (! isAnomaly)
        {
            meteoPoints[i].elaboration = NODATA;
        }
        meteoPoints[i].anomaly = NODATA;
        meteoPoints[i].anomalyPercentage = NODATA;
    }

    int infoStep = 0;
    QString infoStr;

    errorString.clear();

    Crit3DClimate* climaUsed = new Crit3DClimate();

    if (isAnomaly)
    {
        climaUsed->copyParam(referenceClima);
        if (showInfo)
        {
            infoStr = "Anomaly - Meteo Points";
            infoStep = setProgressBar(infoStr, nrMeteoPoints);
        }
    }
    else
    {
        climaUsed->copyParam(clima);
        if (showInfo)
        {
            infoStr = "Elaboration - Meteo Points";
            infoStep = setProgressBar(infoStr, nrMeteoPoints);
        }
    }

    QDate startDate;
    QDate endDate;

    if (climaUsed->nYears() > 0)
    {
        int myYearStart = climaUsed->yearStart();
        int myYearEnd = climaUsed->yearEnd() + climaUsed->nYears();
        startDate.setDate(myYearStart, climaUsed->genericPeriodDateStart().month(), climaUsed->genericPeriodDateStart().day());
        endDate.setDate(myYearEnd, climaUsed->genericPeriodDateEnd().month(), climaUsed->genericPeriodDateEnd().day());
        // check dates - leap case
        if (climaUsed->genericPeriodDateStart().month() == 2 && climaUsed->genericPeriodDateStart().day() == 29)
        {
            if (!isLeapYear(myYearStart))
            {
                if (climaUsed->periodType() != dailyPeriod)
                {
                    startDate.setDate(myYearStart, 2, 28);
                }
                else
                {
                    startDate.setDate(myYearStart, 3, 1);
                }
            }
        }
        if (climaUsed->genericPeriodDateEnd().month() == 2 && climaUsed->genericPeriodDateEnd().day() == 29)
        {
            if (!isLeapYear(myYearEnd))
            {
                if (climaUsed->periodType() != dailyPeriod)
                {
                    endDate.setDate(myYearEnd, 2, 28);
                }
                else
                {
                    endDate.setDate(myYearEnd, 3, 1);
                }
            }
        }
    }
    else if (climaUsed->nYears() < 0)
    {
        int myYearStart = climaUsed->yearStart() + climaUsed->nYears();
        int myYearEnd = climaUsed->yearEnd();
        startDate.setDate(myYearStart, climaUsed->genericPeriodDateStart().month(), climaUsed->genericPeriodDateStart().day());
        endDate.setDate(myYearEnd, climaUsed->genericPeriodDateEnd().month(), climaUsed->genericPeriodDateEnd().day());
        // check dates - leap case
        if (climaUsed->genericPeriodDateStart().month() == 2 && climaUsed->genericPeriodDateStart().day() == 29)
        {
            if (!isLeapYear(myYearStart))
            {
                if (climaUsed->periodType() != dailyPeriod)
                {
                    startDate.setDate(myYearStart, 2, 28);
                }
                else
                {
                    startDate.setDate(myYearStart, 3, 1);
                }
            }
        }
        if (climaUsed->genericPeriodDateEnd().month() == 2 && climaUsed->genericPeriodDateEnd().day() == 29)
        {
            if (!isLeapYear(myYearEnd))
            {
                if (climaUsed->periodType() != dailyPeriod)
                {
                    endDate.setDate(myYearEnd, 2, 28);
                }
                else
                {
                    endDate.setDate(myYearEnd, 3, 1);
                }
            }
        }
    }
    else
    {
        int myYearStart = climaUsed->yearStart();
        int myYearEnd = climaUsed->yearEnd();
        startDate.setDate(myYearStart, climaUsed->genericPeriodDateStart().month(), climaUsed->genericPeriodDateStart().day());
        endDate.setDate(myYearEnd, climaUsed->genericPeriodDateEnd().month(), climaUsed->genericPeriodDateEnd().day());
        // check dates - leap case
        if (climaUsed->genericPeriodDateStart().month() == 2 && climaUsed->genericPeriodDateStart().day() == 29)
        {
            if (!isLeapYear(myYearStart))
            {
                if (climaUsed->periodType() != dailyPeriod)
                {
                    startDate.setDate(myYearStart, 2, 28);
                }
                else
                {
                    startDate.setDate(myYearStart, 3, 1);
                }
            }
        }
        if (climaUsed->genericPeriodDateEnd().month() == 2 && climaUsed->genericPeriodDateEnd().day() == 29)
        {
            if (!isLeapYear(myYearEnd))
            {
                if (climaUsed->periodType() != dailyPeriod)
                {
                    endDate.setDate(myYearEnd, 2, 28);
                }
                else
                {
                    endDate.setDate(myYearEnd, 3, 1);
                }
            }
        }
    }


//    if (clima->elab1() == "phenology")
//    {
//        Then currentPheno.setPhenoPoint i;  // TODO
//    }

    int validPoints = 0;

    Crit3DMeteoPoint* meteoPointTemp = new Crit3DMeteoPoint;
    bool dataAlreadyLoaded = false;

    for (int i = 0; i < nrMeteoPoints; i++)
    {
        if (meteoPoints[i].active)
        {
            // copy data to MPTemp
            meteoPointTemp->id = meteoPoints[i].id;
            meteoPointTemp->point.utm.x = meteoPoints[i].point.utm.x;  // LC to compute distance in passingClimateToAnomaly
            meteoPointTemp->point.utm.y = meteoPoints[i].point.utm.y;  // LC to compute distance in passingClimateToAnomaly
            meteoPointTemp->point.z = meteoPoints[i].point.z;
            meteoPointTemp->latitude = meteoPoints[i].latitude;
            meteoPointTemp->elaboration = meteoPoints[i].elaboration;

            // meteoPointTemp should be init
            meteoPointTemp->nrObsDataDaysH = 0;
            meteoPointTemp->nrObsDataDaysD = 0;

            if (showInfo && (i % infoStep) == 0)
                        updateProgressBar(i);

            if (isAnomaly && climaUsed->getIsClimateAnomalyFromDb())
            {
                if ( passingClimateToAnomaly(&errorString, meteoPointTemp, climaUsed, meteoPoints, nrMeteoPoints, clima->getElabSettings()) )
                {
                    validPoints++;
                }
            }
            else
            {
                bool isMeteoGrid = false;
                if ( elaborationOnPoint(&errorString, meteoPointsDbHandler, nullptr, meteoPointTemp, climaUsed, isMeteoGrid, startDate, endDate, isAnomaly, meteoSettings, dataAlreadyLoaded))
                {
                    validPoints++;
                }
            }

            // save result to MP
            meteoPoints[i].elaboration = meteoPointTemp->elaboration;
            meteoPoints[i].anomaly = meteoPointTemp->anomaly;
            meteoPoints[i].anomalyPercentage = meteoPointTemp->anomalyPercentage;
        }

    } // end for
    if (showInfo) closeProgressBar();

    delete meteoPointTemp;
    delete climaUsed;

    if (validPoints == 0)
    {
        if (errorString.isEmpty())
        {
            errorString = "No valid points available:";
            errorString += "\ncheck Settings->Parameters->Meteo->minimum percentage of valid data [%]";
        }
        return false;
    }
    else return true;

}


bool PragaProject::elaborationPointsCycleGrid(bool isAnomaly, bool showInfo)
{

    bool isMeteoGrid = true; // grid

    std::string id;
    int validCell = 0;

    int infoStep = 1;
    QString infoStr;

    errorString.clear();

    Crit3DClimate* climaUsed = new Crit3DClimate();

    if (isAnomaly)
    {
        climaUsed->copyParam(referenceClima);
        if (showInfo)
        {
            infoStr = "Anomaly - Meteo Grid";
            infoStep = setProgressBar(infoStr, this->meteoGridDbHandler->gridStructure().header().nrRows);
        }
    }
    else
    {
        climaUsed->copyParam(clima);
        if (showInfo)
        {
            infoStr = "Elaboration - Meteo Grid";
            infoStep = setProgressBar(infoStr, this->meteoGridDbHandler->gridStructure().header().nrRows);
        }
    }

    QDate startDate;
    QDate endDate;

    if (climaUsed->nYears() > 0)
    {
        int myYearStart = climaUsed->yearStart();
        int myYearEnd = climaUsed->yearEnd() + climaUsed->nYears();
        startDate.setDate(myYearStart, climaUsed->genericPeriodDateStart().month(), climaUsed->genericPeriodDateStart().day());
        endDate.setDate(myYearEnd, climaUsed->genericPeriodDateEnd().month(), climaUsed->genericPeriodDateEnd().day());
        // check dates - leap case
        if (climaUsed->genericPeriodDateStart().month() == 2 && climaUsed->genericPeriodDateStart().day() == 29)
        {
            if (!isLeapYear(myYearStart))
            {
                if (climaUsed->periodType() != dailyPeriod)
                {
                    startDate.setDate(myYearStart, 2, 28);
                }
                else
                {
                    startDate.setDate(myYearStart, 3, 1);
                }
            }
        }
        if (climaUsed->genericPeriodDateEnd().month() == 2 && climaUsed->genericPeriodDateEnd().day() == 29)
        {
            if (!isLeapYear(myYearEnd))
            {
                if (climaUsed->periodType() != dailyPeriod)
                {
                    endDate.setDate(myYearEnd, 2, 28);
                }
                else
                {
                    endDate.setDate(myYearEnd, 3, 1);
                }
            }
        }
    }
    else if (climaUsed->nYears() < 0)
    {
        int myYearStart = climaUsed->yearStart() + climaUsed->nYears();
        int myYearEnd = climaUsed->yearEnd();
        startDate.setDate(myYearStart, climaUsed->genericPeriodDateStart().month(), climaUsed->genericPeriodDateStart().day());
        endDate.setDate(myYearEnd, climaUsed->genericPeriodDateEnd().month(), climaUsed->genericPeriodDateEnd().day());
        // check dates - leap case
        if (climaUsed->genericPeriodDateStart().month() == 2 && climaUsed->genericPeriodDateStart().day() == 29)
        {
            if (!isLeapYear(myYearStart))
            {
                if (climaUsed->periodType() != dailyPeriod)
                {
                    startDate.setDate(myYearStart, 2, 28);
                }
                else
                {
                    startDate.setDate(myYearStart, 3, 1);
                }
            }
        }
        if (climaUsed->genericPeriodDateEnd().month() == 2 && climaUsed->genericPeriodDateEnd().day() == 29)
        {
            if (!isLeapYear(myYearEnd))
            {
                if (climaUsed->periodType() != dailyPeriod)
                {
                    endDate.setDate(myYearEnd, 2, 28);
                }
                else
                {
                    endDate.setDate(myYearEnd, 3, 1);
                }
            }
        }
    }
    else
    {
        int myYearStart = climaUsed->yearStart();
        int myYearEnd = climaUsed->yearEnd();
        startDate.setDate(myYearStart, climaUsed->genericPeriodDateStart().month(), climaUsed->genericPeriodDateStart().day());
        endDate.setDate(myYearEnd, climaUsed->genericPeriodDateEnd().month(), climaUsed->genericPeriodDateEnd().day());
        // check dates - leap case
        if (climaUsed->genericPeriodDateStart().month() == 2 && climaUsed->genericPeriodDateStart().day() == 29)
        {
            if (!isLeapYear(myYearStart))
            {
                if (climaUsed->periodType() != dailyPeriod)
                {
                    startDate.setDate(myYearStart, 2, 28);
                }
                else
                {
                    startDate.setDate(myYearStart, 3, 1);
                }
            }
        }
        if (climaUsed->genericPeriodDateEnd().month() == 2 && climaUsed->genericPeriodDateEnd().day() == 29)
        {
            if (!isLeapYear(myYearEnd))
            {
                if (climaUsed->periodType() != dailyPeriod)
                {
                    endDate.setDate(myYearEnd, 2, 28);
                }
                else
                {
                    endDate.setDate(myYearEnd, 3, 1);
                }
            }
        }
    }

    bool dataAlreadyLoaded = false;

    for (int row = 0; row < meteoGridDbHandler->gridStructure().header().nrRows; row++)
    {
        if (showInfo && (row % infoStep) == 0)
            updateProgressBar(row);

        for (int col = 0; col < meteoGridDbHandler->gridStructure().header().nrCols; col++)
        {

            if (meteoGridDbHandler->meteoGrid()->getMeteoPointActiveId(row, col, &id))
            {
                Crit3DMeteoPoint* meteoPoint = meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col);

                // copy data to MPTemp
                Crit3DMeteoPoint* meteoPointTemp = new Crit3DMeteoPoint;
                meteoPointTemp->id = meteoPoint->id;
                meteoPointTemp->point.z = meteoPoint->point.z;
                meteoPointTemp->latitude = meteoPoint->latitude;
                meteoPointTemp->elaboration = meteoPoint->elaboration;

                // meteoPointTemp should be init
                meteoPointTemp->nrObsDataDaysH = 0;
                meteoPointTemp->nrObsDataDaysD = 0;

                if (isAnomaly && climaUsed->getIsClimateAnomalyFromDb())
                {
                    if ( passingClimateToAnomalyGrid(&errorString, meteoPointTemp, climaUsed))
                    {
                        validCell += 1;
                    }
                }
                else
                {
                    if  ( elaborationOnPoint(&errorString, nullptr, meteoGridDbHandler, meteoPointTemp, climaUsed, isMeteoGrid, startDate, endDate, isAnomaly, meteoSettings, dataAlreadyLoaded))
                    {
                        validCell += 1;
                    }
                }

                // save result to MP
                meteoPoint->elaboration = meteoPointTemp->elaboration;
                meteoPoint->anomaly = meteoPointTemp->anomaly;
                meteoPoint->anomalyPercentage = meteoPointTemp->anomalyPercentage;
                delete meteoPointTemp;

            }

        }
    }

    if (showInfo) closeProgressBar();

    if (validCell == 0)
    {
        if (errorString.isEmpty())
        {
            errorString = "no valid cells available";
        }
        delete climaUsed;
        return false;
    }
    else
    {
        delete climaUsed;
        return true;
    }

}


bool PragaProject::climatePointsCycle(bool showInfo)
{
    bool isMeteoGrid = false;
    int infoStep;
    QString infoStr;

    int validCell = 0;
    QDate startDate;
    QDate endDate;
    bool changeDataSet = true;

    errorString.clear();
    clima->resetCurrentValues();

    if (showInfo)
    {
        infoStr = "Climate  - Meteo Points";
        infoStep = setProgressBar(infoStr, nrMeteoPoints);
    }
    // parser all the list
    Crit3DClimateList* climateList = clima->getListElab();

    errorString = "";
    climateList->parserElaboration(errorString);
    if (! errorString.isEmpty())
    {
        logError();
        errorString = "";
    }

    Crit3DMeteoPoint* meteoPointTemp = new Crit3DMeteoPoint;
    for (int i = 0; i < nrMeteoPoints; i++)
    {
        if (meteoPoints[i].active)
        {
            if (showInfo && (i % infoStep) == 0)
            {
                updateProgressBar(i);
            }

            meteoPointTemp->id = meteoPoints[i].id;
            meteoPointTemp->point.z = meteoPoints[i].point.z;
            meteoPointTemp->latitude = meteoPoints[i].latitude;
            changeDataSet = true;

            std::vector<float> outputValues;

            for (int j = 0; j < climateList->listClimateElab().size(); j++)
            {
                clima->resetParam();
                clima->setClimateElab(climateList->listClimateElab().at(j));

                if (climateList->listClimateElab().at(j)!= nullptr)
                {
                    // copy current elaboration to clima
                    clima->setDailyCumulated(climateList->listDailyCumulated()[j]);
                    clima->setYearStart(climateList->listYearStart().at(j));
                    clima->setYearEnd(climateList->listYearEnd().at(j));
                    clima->setPeriodType(climateList->listPeriodType().at(j));
                    clima->setPeriodStr(climateList->listPeriodStr().at(j));
                    clima->setGenericPeriodDateStart(climateList->listGenericPeriodDateStart().at(j));
                    clima->setGenericPeriodDateEnd(climateList->listGenericPeriodDateEnd().at(j));
                    clima->setNYears(climateList->listNYears().at(j));
                    clima->setVariable(climateList->listVariable().at(j));
                    clima->setElab1(climateList->listElab1().at(j));
                    clima->setElab2(climateList->listElab2().at(j));
                    clima->setParam1(climateList->listParam1().at(j));
                    clima->setParam2(climateList->listParam2().at(j));
                    clima->setParam1IsClimate(climateList->listParam1IsClimate().at(j));
                    clima->setParam1ClimateField(climateList->listParam1ClimateField().at(j));

                    if (clima->periodType() == genericPeriod)
                    {
                        startDate.setDate(clima->yearStart(), clima->genericPeriodDateStart().month(), clima->genericPeriodDateStart().day());
                        endDate.setDate(clima->yearEnd() + clima->nYears(), clima->genericPeriodDateEnd().month(), clima->genericPeriodDateEnd().day());
                    }
                    else if (clima->periodType() == seasonalPeriod)
                    {
                        startDate.setDate(clima->yearStart() -1, 12, 1);
                        endDate.setDate(clima->yearEnd(), 12, 31);
                    }
                    else
                    {
                        startDate.setDate(clima->yearStart(), 1, 1);
                        endDate.setDate(clima->yearEnd(), 12, 31);
                    }
                }
                else
                {
                    errorString = "parser elaboration error";
                    delete meteoPointTemp;
                    return false;
                }

                if (climateOnPoint(&errorString, meteoPointsDbHandler, nullptr, clima, meteoPointTemp, outputValues, isMeteoGrid, startDate, endDate, changeDataSet, meteoSettings))
                {
                    validCell = validCell + 1;
                }
                changeDataSet = false;

            }

        }
    }
    if (showInfo) closeProgressBar();

    if (validCell == 0)
    {
        if (errorString.isEmpty())
        {
            errorString = "no valid cells available";
        }
        logError(errorString);
        delete meteoPointTemp;
        return false;
    }
    else
    {
        logInfo("climate saved");
        delete meteoPointTemp;
        return true;
    }
}


bool PragaProject::climatePointsCycleGrid(bool showInfo)
{

    bool isMeteoGrid = true;
    int infoStep;
    QString infoStr;

    int validCell = 0;
    QDate startDate;
    QDate endDate;
    std::string id;
    bool changeDataSet = true;

    errorString.clear();
    clima->resetCurrentValues();

    if (showInfo)
    {
        infoStr = "Climate  - Meteo Grid";
        infoStep = setProgressBar(infoStr, this->meteoGridDbHandler->gridStructure().header().nrRows);
    }

    // parser all the list
    Crit3DClimateList* climateList = clima->getListElab();

    errorString = "";
    climateList->parserElaboration(errorString);
    if (! errorString.isEmpty())
    {
        logError();
        errorString = "";
    }

    Crit3DMeteoPoint* meteoPointTemp = new Crit3DMeteoPoint;
    for (int row = 0; row < meteoGridDbHandler->gridStructure().header().nrRows; row++)
    {
        if (showInfo && (row % infoStep) == 0)
            updateProgressBar(row);

        for (int col = 0; col < meteoGridDbHandler->gridStructure().header().nrCols; col++)
        {

           if (meteoGridDbHandler->meteoGrid()->getMeteoPointActiveId(row, col, &id))
           {

               Crit3DMeteoPoint* meteoPoint = meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col);

               meteoPointTemp->id = meteoPoint->id;
               meteoPointTemp->point.z = meteoPoint->point.z;
               meteoPointTemp->latitude = meteoPoint->latitude;

               changeDataSet = true;
               std::vector<float> outputValues;

               for (int j = 0; j < climateList->listClimateElab().size(); j++)
               {

                   clima->resetParam();
                   clima->setClimateElab(climateList->listClimateElab().at(j));


                   if (climateList->listClimateElab().at(j)!= nullptr)
                   {

                       // copy current elaboration to clima
                       clima->setDailyCumulated(climateList->listDailyCumulated()[j]);
                       clima->setYearStart(climateList->listYearStart().at(j));
                       clima->setYearEnd(climateList->listYearEnd().at(j));
                       clima->setPeriodType(climateList->listPeriodType().at(j));
                       clima->setPeriodStr(climateList->listPeriodStr().at(j));
                       clima->setGenericPeriodDateStart(climateList->listGenericPeriodDateStart().at(j));
                       clima->setGenericPeriodDateEnd(climateList->listGenericPeriodDateEnd().at(j));
                       clima->setNYears(climateList->listNYears().at(j));
                       clima->setVariable(climateList->listVariable().at(j));
                       clima->setElab1(climateList->listElab1().at(j));
                       clima->setElab2(climateList->listElab2().at(j));
                       clima->setParam1(climateList->listParam1().at(j));
                       clima->setParam2(climateList->listParam2().at(j));
                       clima->setParam1IsClimate(climateList->listParam1IsClimate().at(j));
                       clima->setParam1ClimateField(climateList->listParam1ClimateField().at(j));

                       if (clima->periodType() == genericPeriod)
                       {
                           startDate.setDate(clima->yearStart(), clima->genericPeriodDateStart().month(), clima->genericPeriodDateStart().day());
                           endDate.setDate(clima->yearEnd() + clima->nYears(), clima->genericPeriodDateEnd().month(), clima->genericPeriodDateEnd().day());
                       }
                       else if (clima->periodType() == seasonalPeriod)
                       {
                           startDate.setDate(clima->yearStart() -1, 12, 1);
                           endDate.setDate(clima->yearEnd(), 12, 31);
                       }
                       else
                       {
                           startDate.setDate(clima->yearStart(), 1, 1);
                           endDate.setDate(clima->yearEnd(), 12, 31);
                       }

                   }
                   else
                   {
                       errorString = "parser elaboration error";
                       delete meteoPointTemp;
                       return false;
                   }

                   if (climateOnPoint(&errorString, nullptr, meteoGridDbHandler, clima, meteoPointTemp, outputValues, isMeteoGrid, startDate, endDate, changeDataSet, meteoSettings))
                   {
                       validCell = validCell + 1;
                   }
                   changeDataSet = false;

               }

           }
       }
   }

   if (showInfo) closeProgressBar();

   if (validCell == 0)
   {
       if (errorString.isEmpty())
       {
           errorString = "no valid cells available";
       }
       logError(errorString);
       delete meteoPointTemp;
       return false;
    }
    else
    {
       logInfo("climate saved");
        delete meteoPointTemp;
        return true;
    }

}

bool PragaProject::downloadDailyDataArkimet(QList<QString> variables, bool prec0024, QDate startDate, QDate endDate, bool showInfo)
{
    // check meteo point
    if (! meteoPointsLoaded || nrMeteoPoints == 0)
    {
        logError("No meteo points");
        return false;
    }

    QString id, dataset;
    QList<QString> datasetList;
    QList<QList<QString>> idList;

    QList<int> arkIdVar;
    Download* myDownload = new Download(meteoPointsDbHandler->getDbName());

    for( int i=0; i < variables.size(); i++ )
    {
        if ( !idArkimetDailyMap[variables[i]].isEmpty())
        {
            arkIdVar.append(idArkimetDailyMap[variables[i]]);
        }
        else
        {
            arkIdVar.append(myDownload->getDbArkimet()->getId(variables[i]));
            if (myDownload->getDbArkimet()->getErrorString() != "")
            {
                logError(myDownload->getDbArkimet()->getErrorString());
                myDownload->getDbArkimet()->setErrorString("");
            }
        }
    }

    if (arkIdVar.size() == 0)
    {
        logError("No variables to download");
        delete myDownload;
        return false;
    }

    int index = 0;
    bool isSelection = isSelectionPointsActive(meteoPoints, nrMeteoPoints);
    for( int i=0; i < nrMeteoPoints; i++ )
    {
        if (!isSelection || meteoPoints[i].selected)
        {
            id = QString::fromStdString(meteoPoints[i].id);
            dataset = QString::fromStdString(meteoPoints[i].dataset);

            // if the point doesn't have dataset, it is not downloaded from arkimet
            if (! dataset.isEmpty())
            {
                if (! datasetList.contains(dataset))
                {
                    datasetList << dataset;
                    QList<QString> myList;
                    myList << id;
                    idList.append(myList);
                }
                else
                {
                    index = datasetList.indexOf(dataset);
                    idList[index].append(id);
                }
            }
        }
    }

    int nrDays = int(startDate.daysTo(endDate) + 1);

    for( int i=0; i < datasetList.size(); i++ )
    {
        if (showInfo)
        {
            setProgressBar("Download daily data from: " + startDate.toString("yyyy-MM-dd") + " to: " + endDate.toString("yyyy-MM-dd") + " dataset:" + datasetList[i], nrDays);
        }

        QDate date1 = startDate;
        QDate date2 = std::min(date1.addDays(MAXDAYS_DOWNLOAD_DAILY), endDate);

        bool isOk = true;
        while (date1 <= endDate && isOk)
        {
            if (showInfo)
            {
                updateProgressBar(startDate.daysTo(date2) + 1);
            }

            isOk = myDownload->downloadDailyData(date1, date2, datasetList[i], idList[i], arkIdVar, prec0024, errorString);
            if (! isOk)
            {
                logError();
                errorString = "";
            }

            date1 = date2.addDays(1);
            date2 = std::min(date1.addDays(MAXDAYS_DOWNLOAD_DAILY), endDate);
        }

        if (showInfo)
            closeProgressBar();
    }

    delete myDownload;
    return true;
}


bool PragaProject::downloadHourlyDataArkimet(QList<QString> variables, QDate startDate, QDate endDate, bool showInfo)
{
    QList<int> arkIdVar;
    Download* myDownload = new Download(meteoPointsDbHandler->getDbName());

    for( int i=0; i < variables.size(); i++ )
    {
        if ( !idArkimetHourlyMap[variables[i]].isEmpty())
        {
            arkIdVar.append(idArkimetHourlyMap[variables[i]]);
        }
        else
        {
            arkIdVar.append(myDownload->getDbArkimet()->getId(variables[i]));
        }
    }

    int index = 0;
    QString id, dataset;
    QList<QString> datasetList;
    QList<QList<QString>> idList;

    bool isSelection = isSelectionPointsActive(meteoPoints, nrMeteoPoints);
    for( int i=0; i < nrMeteoPoints; i++ )
    {
        if (! isSelection || meteoPoints[i].selected)
        {
            id = QString::fromStdString(meteoPoints[i].id);
            dataset = QString::fromStdString(meteoPoints[i].dataset);

            // if the point doesn't have dataset, it is not downloaded from arkimet
            if (! dataset.isEmpty())
            {
                if (! datasetList.contains(dataset))
                {
                    datasetList << dataset;
                    QList<QString> myList;
                    myList << id;
                    idList.append(myList);
                }
                else
                {
                    index = datasetList.indexOf(dataset);
                    idList[index].append(id);
                }
            }
        }
    }

    int nrDays = int(startDate.daysTo(endDate) + 1);

    for( int i=0; i < datasetList.size(); i++ )
    {
        QDate date1 = startDate;
        QDate date2 = std::min(date1.addDays(MAXDAYS_DOWNLOAD_HOURLY), endDate);

        if (showInfo)
        {
            setProgressBar("Download hourly data from: " + startDate.toString("yyyy-MM-dd") + " to:" + endDate.toString("yyyy-MM-dd") + " dataset:" + datasetList[i], nrDays);
        }
        bool isOk = true;
        while (date1 <= endDate && isOk)
        {
            if (showInfo)
            {
                updateProgressBar(startDate.daysTo(date2) + 1);
            }

            isOk = myDownload->downloadHourlyData(date1, date2, datasetList[i], idList[i], arkIdVar, errorString);
            if (! isOk)
            {
                updateProgressBarText(errorString);
                errorString = "";
            }

            date1 = date2.addDays(1);
            date2 = std::min(date1.addDays(MAXDAYS_DOWNLOAD_HOURLY), endDate);
        }
        if (showInfo)
        {
            closeProgressBar();
        }
    }

    delete myDownload;
    return true;
}


bool PragaProject::averageSeriesOnZonesMeteoGrid(meteoVariable variable, meteoComputation elab1MeteoComp, QString aggregationString, float threshold, gis::Crit3DRasterGrid* zoneGrid, QDate startDate, QDate endDate, QString periodType, std::vector<float> &outputValues, bool showInfo)
{
    aggregationMethod spatialElab = getAggregationMethod(aggregationString.toStdString());
    std::vector <std::vector<int> > meteoGridRow(zoneGrid->header->nrRows, std::vector<int>(zoneGrid->header->nrCols, NODATA));
    std::vector <std::vector<int> > meteoGridCol(zoneGrid->header->nrRows, std::vector<int>(zoneGrid->header->nrCols, NODATA));
    meteoGridDbHandler->meteoGrid()->saveRowColfromZone(zoneGrid, meteoGridRow, meteoGridCol);

    float percValue;
    bool isMeteoGrid = true;
    std::string id;
    unsigned int zoneIndex = 0;
    int indexSeries = 0;
    float value;
    std::vector<float> outputSeries;
    std::vector <std::vector<int>> indexRowCol(meteoGridDbHandler->gridStructure().header().nrRows, std::vector<int>(meteoGridDbHandler->gridStructure().header().nrCols, NODATA));

    gis::updateMinMaxRasterGrid(zoneGrid);
    std::vector <std::vector<float> > zoneVector((unsigned int)(zoneGrid->maximum), std::vector<float>());
    std::vector <double> utmXvector;
    std::vector <double> utmYvector;
    std::vector <double> latVector;
    std::vector <double> lonVector;
    std::vector <int> count;
    for (int i = 0; i < int(zoneGrid->maximum); i++)
    {
        utmXvector.push_back(0);
        utmYvector.push_back(0);
        count.push_back(0);
    }

    for (int zoneRow = 0; zoneRow < zoneGrid->header->nrRows; zoneRow++)
    {
        for (int zoneCol = 0; zoneCol < zoneGrid->header->nrCols; zoneCol++)
        {
            float zoneValue = zoneGrid->value[zoneRow][zoneCol];
            double utmx = zoneGrid->utmPoint(zoneRow, zoneCol)->x;
            double utmy = zoneGrid->utmPoint(zoneRow, zoneCol)->y;

            if (! isEqual(zoneValue, zoneGrid->header->flag))
            {
                zoneIndex = (unsigned int)(zoneValue);

                if (zoneIndex > 0 && zoneIndex <= zoneGrid->maximum)
                {
                    utmXvector[zoneIndex-1] = utmXvector[zoneIndex-1] + utmx;
                    utmYvector[zoneIndex-1] = utmYvector[zoneIndex-1] + utmy;
                    count[zoneIndex-1] = count[zoneIndex-1] + 1;
                }
            }
        }
    }

    for (unsigned int zonePos = 0; zonePos < zoneVector.size(); zonePos++)
    {
        // average x, y
        utmXvector[zonePos] = utmXvector[zonePos] / count[zonePos];
        utmYvector[zonePos] = utmYvector[zonePos] / count[zonePos];
        double lat, lon;
        gis::getLatLonFromUtm(gisSettings, utmXvector[zonePos], utmYvector[zonePos], &lat, &lon);
        latVector.push_back(lat);
        lonVector.push_back(lon);
    }

    // save point properties
    int nrAggregationPoints = int(zoneVector.size());
    if (! aggregationDbHandler->writeAggregationPointProperties(nrAggregationPoints, aggregationString, lonVector, latVector))
    {
        errorString = aggregationDbHandler->error();
        return false;
    }

    int infoStep = 0;
    if (showInfo)
    {
        infoStep = setProgressBar("Aggregating data...", this->meteoGridDbHandler->gridStructure().header().nrRows);
    }

    Crit3DMeteoPoint* meteoPointTemp = new Crit3DMeteoPoint;

     for (int row = 0; row < meteoGridDbHandler->gridStructure().header().nrRows; row++)
     {
         if (showInfo && (row % infoStep) == 0)
             updateProgressBar(row);

         for (int col = 0; col < meteoGridDbHandler->gridStructure().header().nrCols; col++)
         {
            if (meteoGridDbHandler->meteoGrid()->getMeteoPointActiveId(row, col, &id))
            {
                Crit3DMeteoPoint* meteoPoint = meteoGridDbHandler->meteoGrid()->meteoPointPointer(row, col);

                // copy data to MPTemp
                meteoPointTemp->id = meteoPoint->id;
                meteoPointTemp->point.z = meteoPoint->point.z;
                meteoPointTemp->latitude = meteoPoint->latitude;
                meteoPointTemp->elaboration = meteoPoint->elaboration;

                // meteoPointTemp should be init
                meteoPointTemp->nrObsDataDaysH = 0;
                meteoPointTemp->nrObsDataDaysD = 0;

                outputValues.clear();
                bool dataLoaded = preElaboration(&errorString, nullptr, meteoGridDbHandler, meteoPointTemp, isMeteoGrid, variable, elab1MeteoComp, startDate, endDate, outputValues, &percValue, meteoSettings);
                if (dataLoaded)
                {
                    outputSeries.insert(outputSeries.end(), outputValues.begin(), outputValues.end());
                    indexRowCol[row][col] = indexSeries;
                    indexSeries = indexSeries + 1;
                }
            }
        }
    }
    if (showInfo) closeProgressBar();

     int nrDays = int(startDate.daysTo(endDate) + 1);
     std::vector< std::vector<float> > dailyElabAggregation(nrDays, std::vector<float>(int(zoneGrid->maximum), NODATA));

     for (int day = 0; day < nrDays; day++)
     {
         for (int zoneRow = 0; zoneRow < zoneGrid->header->nrRows; zoneRow++)
         {
             for (int zoneCol = 0; zoneCol < zoneGrid->header->nrCols; zoneCol++)
             {
                float zoneValue = zoneGrid->value[zoneRow][zoneCol];
                if (! isEqual(zoneValue, zoneGrid->header->flag))
                {
                    zoneIndex = (unsigned int)(zoneValue);
                    if (zoneIndex < 1 || zoneIndex > zoneGrid->maximum)
                    {
                        errorString = "invalid zone index: " + QString::number(zoneIndex);
                        errorString += "\nZone number has to be between 1 and " + QString::number(zoneGrid->maximum);
                        return false;
                    }

                    if (meteoGridRow[zoneRow][zoneCol] != NODATA && meteoGridCol[zoneRow][zoneCol] != NODATA)
                    {
                        if (indexRowCol[meteoGridRow[zoneRow][zoneCol]][meteoGridCol[zoneRow][zoneCol]] != NODATA)
                        {
                            value = outputSeries.at(indexRowCol[meteoGridRow[zoneRow][zoneCol]][meteoGridCol[zoneRow][zoneCol]]*outputValues.size()+day);
                            if (value != meteoGridDbHandler->gridStructure().header().flag)
                            {
                                zoneVector[zoneIndex-1].push_back(value);
                            }
                        }
                    }
                }
             }
         }

         for (unsigned int zonePos = 0; zonePos < zoneVector.size(); zonePos++)
         {
            std::vector<float> validValues;
            validValues = zoneVector[zonePos];
            if (! isEqual(threshold, NODATA))
            {
                extractValidValuesWithThreshold(validValues, threshold);
            }

            float res = NODATA;
            int size = int(validValues.size());

            switch (spatialElab)
            {
                case aggrAverage:
                    {
                        res = statistics::mean(validValues);
                        break;
                    }
                case aggrMedian:
                    {
                        res = sorting::percentile(validValues, size, 50.0, true);
                        break;
                    }
                case aggrStdDeviation:
                    {
                        res = statistics::standardDeviation(validValues, size);
                        break;
                    }
                case aggr95Perc:
                    {
                        res = sorting::percentile(validValues, size, 95.0, true);
                        break;
                    }
                default:
                    {
                        // default: average
                        res = statistics::mean(validValues);
                        break;
                    }
            }

            dailyElabAggregation[unsigned(day)][zonePos] = res;
         }
         // clear zoneVector
         for (unsigned int zonePos = 0; zonePos < zoneVector.size(); zonePos++)
         {
            zoneVector[zonePos].clear();
         }
     }

     // save dailyElabAggregation result into DB
     if (showInfo) setProgressBar("Save data...", 0);
     if (! aggregationDbHandler->saveAggrData(int(zoneGrid->maximum), aggregationString, periodType,
                                             startDate, endDate, variable, dailyElabAggregation))
     {
         errorString = aggregationDbHandler->error();
         if (showInfo) closeProgressBar();
         return false;
     }
     if (showInfo) closeProgressBar();

     return true;

}


void PragaProject::savePragaParameters()
{
    parametersSettings->beginGroup("elaboration");
        parametersSettings->setValue("anomaly_pts_max_distance", QString::number(double(clima->getElabSettings()->getAnomalyPtsMaxDistance())));
        parametersSettings->setValue("anomaly_pts_max_delta_z", QString::number(double(clima->getElabSettings()->getAnomalyPtsMaxDeltaZ())));
        parametersSettings->setValue("grid_min_coverage", QString::number(double(clima->getElabSettings()->getGridMinCoverage())));
        parametersSettings->setValue("merge_joint_stations", clima->getElabSettings()->getMergeJointStations());
    parametersSettings->endGroup();
}

QString getMapFileOutName(meteoVariable myVar, QDate myDate, int myHour)
{
    std::string myName = getMeteoVarName(myVar);
    if (myName == "") return "";

    QString name = QString::fromStdString(myName);
    name += "_" + myDate.toString(Qt::ISODate);
    if (getVarFrequency(myVar) == hourly) name += "_" + QString::number(myHour);

    return name;
}


gis::Crit3DRasterGrid* PragaProject::getPragaMapFromVar(meteoVariable myVar)
{
    gis::Crit3DRasterGrid* myGrid = nullptr;

    // search in project maps (hourlyMeteoMaps and radiationMaps)
    myGrid = getHourlyMeteoRaster(myVar);
    // search in pragaDailyMaps
    if (myGrid == nullptr) myGrid = pragaDailyMaps->getMapFromVar(myVar);
    // saerch in pragaHourlyMaps
    if (myGrid == nullptr) myGrid = pragaHourlyMaps->getMapFromVar(myVar);

    return myGrid;
}


bool PragaProject::timeAggregateGridVarHourlyInDaily(meteoVariable dailyVar, Crit3DDate dateIni, Crit3DDate dateFin)
{
    Crit3DDate myDate;
    Crit3DMeteoPoint* meteoPoint;

    for (unsigned col = 0; col < unsigned(meteoGridDbHandler->gridStructure().header().nrCols); col++)
        for (unsigned row = 0; row < unsigned(meteoGridDbHandler->gridStructure().header().nrRows); row++)
        {
            meteoPoint = meteoGridDbHandler->meteoGrid()->meteoPointPointer(row, col);
            if (meteoPoint->active)
                if (! aggregatedHourlyToDaily(dailyVar, meteoPoint, dateIni, dateFin, meteoSettings))
                    return false;
        }

    return true;
}

bool PragaProject::computeDailyVariablesPoint(Crit3DMeteoPoint *meteoPoint, QDate first, QDate last, QList <meteoVariable> variables)
{
    // check variables
    if (variables.size() == 0)
    {
        logError("No variable");
        return false;
    }

    // check meteo points
    if (! meteoPointsLoaded)
    {
        logError("No meteo points");
        return false;
    }

    // check dates
    if (first.isNull() || last.isNull() || first > last)
    {
        logError("Wrong period");
        return false;
    }
    Crit3DDate dateIni(first.day(),first.month(), first.year());
    Crit3DDate dateFin(last.day(),last.month(), last.year());

    QList<QString> listEntries;
    QDate firstTmp;
    QDate lastTmp;
    int index;

    for (int i = 0; i<variables.size(); i++)
    {
        firstTmp = first;
        lastTmp = last;
        index = 0;

        int varId = meteoPointsDbHandler->getIdfromMeteoVar(variables[i]);
        if (varId != NODATA)
        {
            std::vector<float> dailyData = aggregatedHourlyToDailyList(variables[i], meteoPoint, dateIni, dateFin, meteoSettings);
            if (!dailyData.empty())
            {
                while (firstTmp <= lastTmp)
                {
                    listEntries.push_back(QString("('%1',%2,%3)").arg(firstTmp.toString("yyyy-MM-dd")).arg(varId).arg(dailyData[index]));
                    firstTmp = firstTmp.addDays(1);
                    index = index + 1;
                }
            }
        }
    }
    if (listEntries.empty())
    {
        logError("Failed to compute daily data id point "+QString::fromStdString(meteoPoint->id));
        return false;
    }
    if (!meteoPointsDbHandler->writeDailyDataList(QString::fromStdString(meteoPoint->id), listEntries, errorString))
    {
        logError("Failed to write daily data id point "+QString::fromStdString(meteoPoint->id));
        return false;
    }
    return true;
}

bool PragaProject::timeAggregateGrid(QDate dateIni, QDate dateFin, QList <meteoVariable> variables, bool loadData, bool saveData)
{
    // check variables
    if (variables.size() == 0)
    {
        logError("No variable");
        return false;
    }

    // check meteo grid
    if (! meteoGridLoaded)
    {
        logError("No meteo grid");
        return false;
    }

    // check dates
    if (dateIni.isNull() || dateFin.isNull() || dateIni > dateFin)
    {
        logError("Wrong period");
        return false;
    }

    // now only hourly-->daily
    if (loadData)
    {
        logInfoGUI("Loading grid data: " + dateIni.toString("yyyy-MM-dd") + "-" + dateFin.toString("yyyy-MM-dd"));
        loadMeteoGridHourlyData(QDateTime(dateIni, QTime(1,0), Qt::UTC), QDateTime(dateFin.addDays(1), QTime(0,0), Qt::UTC), false);
    }

    foreach (meteoVariable myVar, variables)
        if (getVarFrequency(myVar) == daily)
            if (! timeAggregateGridVarHourlyInDaily(myVar, getCrit3DDate(dateIni), getCrit3DDate(dateFin))) return false;

    // saving hourly and daily meteo grid data to DB
    if (saveData)
    {
        QString myError;
        logInfoGUI("Saving meteo grid data");
        if (! meteoGridDbHandler->saveGridData(&myError, QDateTime(dateIni, QTime(1,0,0), Qt::UTC), QDateTime(dateFin.addDays(1), QTime(0,0,0), Qt::UTC), variables, meteoSettings)) return false;
    }

    return true;
}

bool PragaProject::hourlyDerivedVariablesGrid(QDate first, QDate last, bool loadData, bool saveData)
{

    // check meteo grid
    if (! meteoGridLoaded)
    {
        logError("No meteo grid");
        return false;
    }

    // check dates
    if (first.isNull() || last.isNull() || first > last)
    {
        logError("Wrong period");
        return false;
    }

    QDateTime firstDateTime = QDateTime(first, QTime(1,0), Qt::UTC);
    QDateTime lastDateTime = QDateTime(last.addDays(1), QTime(0,0), Qt::UTC);

    // now only hourly-->daily
    if (loadData)
    {
        logInfoGUI("Loading grid data: " + first.toString("yyyy-MM-dd") + "-" + last.toString("yyyy-MM-dd"));
        loadMeteoGridHourlyData(firstDateTime, lastDateTime, false);
    }

    while(firstDateTime <= lastDateTime)
    {
        meteoGridDbHandler->meteoGrid()->computeHourlyDerivedVariables(getCrit3DTime(firstDateTime));
        firstDateTime = firstDateTime.addSecs(3600);
    }

    firstDateTime = QDateTime(first, QTime(1,0), Qt::UTC);
    // saving hourly meteo grid data to DB
    if (saveData)
    {

        // save derived variables
        QList <meteoVariable> variables;
        variables << leafWetness << referenceEvapotranspiration;
        QString myError;
        logInfoGUI("Saving meteo grid data");
        if (! meteoGridDbHandler->saveGridData(&myError, firstDateTime, lastDateTime, variables, meteoSettings)) return false;
    }

    return true;
}


bool PragaProject::interpolationOutputPointsPeriod(QDate firstDate, QDate lastDate, QList <meteoVariable> variables)
{
    // check
    if (variables.size() == 0)
    {
        errorString = "No variables";
        return false;
    }

    if (! meteoPointsLoaded || nrMeteoPoints == 0)
    {
        errorString = "No meteo points";
        return false;
    }

    if (! outputMeteoPointsLoaded || outputPoints.empty())
    {
        errorString = "No output points";
        return false;
    }

    if (! DEM.isLoaded)
    {
        errorString = "Load a Digital Elevation Model before.";
        return false;
    }

    // check dates
    if (firstDate.isNull() || lastDate.isNull() || firstDate > lastDate)
    {
        errorString = "Wrong dates";
        return false;
    }

    // check variables
    bool isDaily = false;
    bool isHourly = false;
    meteoVariable myVar;

    foreach (myVar, variables)
    {
        frequencyType freq = getVarFrequency(myVar);

        if (freq == noFrequency)
        {
            errorString = "Unknown variable: " + QString::fromStdString(getMeteoVarName(myVar));
            return false;
        }
        else if (freq == hourly)
            isHourly = true;
        else if (freq == daily)
            isDaily = true;
    }

    // initialize maps
    if (pragaDailyMaps == nullptr) pragaDailyMaps = new Crit3DDailyMeteoMaps(DEM);
    if (pragaHourlyMaps == nullptr) pragaHourlyMaps = new PragaHourlyMeteoMaps(DEM);

    // check if topographic distance is needed
    bool useTd = false;
    if (interpolationSettings.getUseTD())
    {
        foreach (myVar, variables)
        {
            if (getUseTdVar(myVar))
            {
                useTd = true;
                break;
            }
        }
    }

    // load topographic distance
    if (useTd)
    {
        logInfoGUI("Loading topographic distance maps...");
        if (! loadTopographicDistanceMaps(true, false))
            return false;
    }

    // initialize data list
    QVector<QList<QString>> dailyDataList;
    QVector<QList<QString>> hourlyDataList;
    if (isDaily)
    {
        dailyDataList.resize(int(outputPoints.size()));
    }
    if (isHourly)
    {
        hourlyDataList.resize(int(outputPoints.size()));
    }

    int nrDays = firstDate.daysTo(lastDate) + 1;
    int nrDaysLoading = std::min(nrDays, 30);
    QDate lastLoadingDate;
    bool isOk;

    QDate myDate = firstDate;
    while (myDate <= lastDate)
    {
        // load data
        if (myDate == firstDate || myDate > lastLoadingDate)
        {
            lastLoadingDate = myDate.addDays(nrDaysLoading - 1);
            if (lastLoadingDate > lastDate)
                lastLoadingDate = lastDate;

            // load one day before (for transmissivity)
            logInfoGUI("Loading meteo points data from " + myDate.addDays(-1).toString("yyyy-MM-dd") + " to " + lastLoadingDate.toString("yyyy-MM-dd"));
            if (! loadMeteoPointsData(myDate.addDays(-1), lastLoadingDate, isHourly, isDaily, false))
                return false;
        }

        if (isHourly)
        {
            // initialize
            hourlyMeteoMaps->initialize();
            radiationMaps->initialize();
            pragaHourlyMaps->initialize();

            for (int hour = 1; hour <= 24; hour++)
            {
                QDateTime myDateTime;
                myDateTime.setDate(myDate);
                myDateTime.setTime(QTime(hour, 0, 0, 0));
                QString dateTimeStr = myDateTime.toString("yyyy-MM-dd hh:mm:ss");

                logInfoGUI("Interpolating hourly variables for " + dateTimeStr);

                foreach (myVar, variables)
                {
                    if (getVarFrequency(myVar) == hourly)
                    {
                        setComputeOnlyPoints(true);

                        // TODO special variables

                        if (myVar == airRelHumidity && interpolationSettings.getUseDewPoint())
                        {
                            if (interpolationSettings.getUseInterpolatedTForRH())
                            {
                                passInterpolatedTemperatureToHumidityPoints(getCrit3DTime(myDate, hour), meteoSettings);
                            }

                            isOk = interpolationDemMain(airDewTemperature, getCrit3DTime(myDate, hour), hourlyMeteoMaps->mapHourlyTdew);

                            if (isOk)
                            {
                                hourlyMeteoMaps->computeRelativeHumidityMap(hourlyMeteoMaps->mapHourlyRelHum);
                            }
                        }
                        else
                        {
                            isOk = interpolationDemMain(myVar, getCrit3DTime(myDate, hour), getPragaMapFromVar(myVar));
                        }

                        setComputeOnlyPoints(false);

                        if (! isOk)
                            return false;

                        int varId = meteoPointsDbHandler->getIdfromMeteoVar(myVar);
                        for (int i = 0; i < outputPoints.size(); i++)
                        {
                            float value = outputPoints[i].currentValue;
                            hourlyDataList[i].push_back(QString("('%1',%2,%3)").arg(dateTimeStr).arg(varId).arg(value));
                        }
                    }
                }
            }
        }

        if (isDaily)
        {
            // initialize
            pragaDailyMaps->initialize();
            QString dateStr = myDate.toString("yyyy-MM-dd");

            logInfoGUI("Interpolating daily variables for " + dateStr);

            foreach (myVar, variables)
            {
                if (getVarFrequency(myVar) == daily)
                {
                    setComputeOnlyPoints(true);

                    // TODO special variables

                    isOk = interpolationDemMain(myVar, getCrit3DTime(myDate, 0), getPragaMapFromVar(myVar));

                    setComputeOnlyPoints(false);

                    if (! isOk)
                        return false;

                    int varId = meteoPointsDbHandler->getIdfromMeteoVar(myVar);
                    for (int i = 0; i < outputPoints.size(); i++)
                    {
                        float value = outputPoints[i].currentValue;
                        dailyDataList[i].push_back(QString("('%1',%2,%3)").arg(dateStr).arg(varId).arg(value));
                    }
                }
            }
        }

        if (myDate == lastLoadingDate)
        {
            // save and clear datalist
            for (int i = 0; i < outputPoints.size(); i++)
            {
                if (isDaily)
                {
                    QString logStr = "";
                    QString pointCode = QString::fromStdString(outputPoints[i].id);
                    outputMeteoPointsDbHandler->writeDailyDataList(pointCode, dailyDataList[i], logStr);
                    if (! logStr.isEmpty())
                    {
                        logInfo(logStr);
                    }
                    dailyDataList[i].clear();
                }
                if (isHourly)
                {
                    QString logStr = "";
                    QString pointCode = QString::fromStdString(outputPoints[i].id);
                    outputMeteoPointsDbHandler->writeHourlyDataList(pointCode, hourlyDataList[i], logStr);
                    if (! logStr.isEmpty())
                    {
                        logInfo(logStr);
                    }
                    hourlyDataList[i].clear();
                }
            }
        }

        myDate = myDate.addDays(1);
    }

    closeLogInfo();
    return true;
}


bool PragaProject::interpolationMeteoGridPeriod(QDate dateIni, QDate dateFin, QList <meteoVariable> variables,
                                                QList <meteoVariable> aggrVariables, bool saveRasters,
                                                int nrDaysLoading, int nrDaysSaving)
{
    // check variables
    if (variables.size() == 0)
    {
        logError("No variable");
        return false;
    }

    // check meteo point
    if (! meteoPointsLoaded || nrMeteoPoints == 0)
    {
        logError("No meteo points");
        return false;
    }

    // check meteo grid
    if (! meteoGridLoaded)
    {
        logError("No meteo grid");
        return false;
    }

    // check dates
    if (dateIni.isNull() || dateFin.isNull() || dateIni > dateFin)
    {
        logError("Wrong period");
        return false;
    }

    // order variables for derived computation
    std::string errString;
    QString myError, rasterName, varName;
    int myHour;
    QDate myDate = dateIni;
    gis::Crit3DRasterGrid* myGrid;
    meteoVariable myVar;
    frequencyType freq;
    bool isDaily = false, isHourly = false;
    QList<meteoVariable> varToSave;
    int countDaysSaving = 0;

    if (pragaDailyMaps == nullptr) pragaDailyMaps = new Crit3DDailyMeteoMaps(DEM);
    if (pragaHourlyMaps == nullptr) pragaHourlyMaps = new PragaHourlyMeteoMaps(DEM);

    // find out needed frequency
    foreach (myVar, variables)
    {
        freq = getVarFrequency(myVar);

        if (freq == noFrequency)
        {
            logError("Unknown variable: " + QString::fromStdString(getMeteoVarName(myVar)));
            return false;
        }
        else if (freq == hourly)
            isHourly = true;
        else if (freq == daily)
            isDaily = true;

        // save two variables for vector wind
        varToSave.push_back(myVar);
        if (myVar == windVectorIntensity)
            varToSave.push_back(windVectorDirection);
        else if (myVar == windVectorDirection)
            varToSave.push_back(windVectorIntensity);
    }

    // find out if detrending needed
    bool useProxies = false;
    foreach (myVar, variables)
    {
        if (getUseDetrendingVar(myVar))
        {
            useProxies = true;
            break;
        }
    }

    // find out if topographic distance is needed
    bool useTd = false;
    if (interpolationSettings.getUseTD())
    {
        foreach (myVar, variables)
        {
            if (getUseTdVar(myVar))
            {
                useTd = true;
                break;
            }
        }
    }

    if (useTd)
    {
        logInfoGUI("Loading topographic distance maps...");
        if (! loadTopographicDistanceMaps(true, false))
            return false;
    }

    // save also time aggregated variables
    foreach (myVar, aggrVariables)
        varToSave.push_back(myVar);

    int currentYear = NODATA;
    QDate saveDateIni = dateIni;

    if (nrDaysLoading == NODATA)
    {
        nrDaysLoading = dateIni.daysTo(dateFin)+1;
    }

    if (nrDaysSaving == NODATA || nrDaysSaving > nrDaysLoading)
    {
        nrDaysSaving = nrDaysLoading;
    }

    QDate loadDateFin = QDate(1800, 1, 1);

    while (myDate <= dateFin)
    {
        countDaysSaving++;

        // check if load needed
        if (myDate == dateIni || myDate > loadDateFin)
        {
            loadDateFin = myDate.addDays(nrDaysLoading-1);
            if (loadDateFin > dateFin) loadDateFin = dateFin;

            logInfoGUI("Initializing meteo grid from " + myDate.addDays(-1).toString("yyyy-MM-dd") + " to " + loadDateFin.toString("yyyy-MM-dd"));
            meteoGridDbHandler->meteoGrid()->initializeData(getCrit3DDate(myDate.addDays(-1)), getCrit3DDate(loadDateFin), isHourly, isDaily, false);

            logInfoGUI("Loading meteo points data from " + myDate.addDays(-1).toString("yyyy-MM-dd") + " to " + loadDateFin.toString("yyyy-MM-dd"));

            // load one day before (for transmissivity)
            if (! loadMeteoPointsData(myDate.addDays(-1), loadDateFin, isHourly, isDaily, false))
                return false;
        }

        // check proxy grid series
        if (useProxies && currentYear != myDate.year())
        {
            logInfoGUI("Interpolating proxy grid series...");

            if (! checkProxyGridSeries(&interpolationSettings, DEM, proxyGridSeries, myDate, errorString))
                return false;

            if (! readProxyValues())
                return false;

            currentYear = myDate.year();
        }

        if (isHourly)
        {
            for (myHour = 1; myHour <= 24; myHour++)
            {
                logInfoGUI("Interpolating hourly variables for " + myDate.toString("yyyy-MM-dd") + " " + QString("%1").arg(myHour, 2, 10, QChar('0')) + ":00");

                foreach (myVar, variables)
                {
                    if (getVarFrequency(myVar) == hourly)
                    {
                        varName = QString::fromStdString(getMeteoVarName(myVar));
                        logInfo(varName);

                        if (myVar == airRelHumidity && interpolationSettings.getUseDewPoint())
                        {
                            if (interpolationSettings.getUseInterpolatedTForRH())
                                passInterpolatedTemperatureToHumidityPoints(getCrit3DTime(myDate, myHour), meteoSettings);
                            if (! interpolationDemMain(airDewTemperature, getCrit3DTime(myDate, myHour), hourlyMeteoMaps->mapHourlyTdew)) return false;
                            hourlyMeteoMaps->computeRelativeHumidityMap(hourlyMeteoMaps->mapHourlyRelHum);

                            if (saveRasters)
                            {
                                myGrid = getPragaMapFromVar(airDewTemperature);
                                rasterName = getMapFileOutName(airDewTemperature, myDate, myHour);
                                if (rasterName != "") gis::writeEsriGrid(getProjectPath().toStdString() + rasterName.toStdString(), myGrid, errString);
                            }
                        }
                        else if (myVar == windVectorDirection || myVar == windVectorIntensity) {
                            if (! interpolationDemMain(windVectorX, getCrit3DTime(myDate, myHour), getPragaMapFromVar(windVectorX))) return false;
                            if (! interpolationDemMain(windVectorY, getCrit3DTime(myDate, myHour), getPragaMapFromVar(windVectorY))) return false;
                            if (! pragaHourlyMaps->computeWindVector()) return false;
                        }
                        else if (myVar == leafWetness) {
                            hourlyMeteoMaps->computeLeafWetnessMap() ;
                        }
                        else if (myVar == referenceEvapotranspiration) {
                            hourlyMeteoMaps->computeET0PMMap(DEM, radiationMaps);
                        }
                        else {
                            if (! interpolationDemMain(myVar, getCrit3DTime(myDate, myHour), getPragaMapFromVar(myVar))) return false;
                        }

                        myGrid = getPragaMapFromVar(myVar);
                        if (myGrid == nullptr) return false;

                        //save raster
                        if (saveRasters)
                        {
                            rasterName = getMapFileOutName(myVar, myDate, myHour);
                            if (rasterName != "") gis::writeEsriGrid(getProjectPath().toStdString() + rasterName.toStdString(), myGrid, errString);
                        }

                        if (myVar == windVectorDirection || myVar == windVectorIntensity)
                        {
                            meteoGridDbHandler->meteoGrid()->spatialAggregateMeteoGrid(windVectorX, hourly, getCrit3DDate(myDate), myHour, 0, &DEM, getPragaMapFromVar(windVectorX), interpolationSettings.getMeteoGridAggrMethod());
                            meteoGridDbHandler->meteoGrid()->spatialAggregateMeteoGrid(windVectorY, hourly, getCrit3DDate(myDate), myHour, 0, &DEM, getPragaMapFromVar(windVectorY), interpolationSettings.getMeteoGridAggrMethod());
                            meteoGridDbHandler->meteoGrid()->computeWindVectorHourly(getCrit3DDate(myDate), myHour);
                        }
                        else
                            meteoGridDbHandler->meteoGrid()->spatialAggregateMeteoGrid(myVar, hourly, getCrit3DDate(myDate), myHour, 0, &DEM, myGrid, interpolationSettings.getMeteoGridAggrMethod());
                    }
                }
            }
        }

        if (isDaily)
        {
            logInfoGUI("Interpolating daily variables for " + myDate.toString("yyyy-MM-dd"));

            foreach (myVar, variables)
            {
                if (getVarFrequency(myVar) == daily)
                {
                    varName = QString::fromStdString(getMeteoVarName(myVar));
                    logInfo(varName);

                    if (myVar == dailyReferenceEvapotranspirationHS) {
                        pragaDailyMaps->computeHSET0Map(&gisSettings, getCrit3DDate(myDate));
                    }
                    else {
                        if (! interpolationDemMain(myVar, getCrit3DTime(myDate, 0), getPragaMapFromVar(myVar))) return false;
                    }

                    // fix daily temperatures consistency
                    if (myVar == dailyAirTemperatureMax || myVar == dailyAirTemperatureMin) {
                        if (! pragaDailyMaps->fixDailyThermalConsistency()) return false;
                    }

                    myGrid = getPragaMapFromVar(myVar);
                    if (myGrid == nullptr) return false;

                    //save raster
                    if (saveRasters)
                    {
                        rasterName = getMapFileOutName(myVar, myDate, 0);
                        gis::writeEsriGrid(getProjectPath().toStdString() + rasterName.toStdString(), myGrid, errString);
                    }

                    meteoGridDbHandler->meteoGrid()->spatialAggregateMeteoGrid(myVar, daily, getCrit3DDate(myDate), 0, 0, &DEM, myGrid, interpolationSettings.getMeteoGridAggrMethod());
                }
            }
        }

        if (countDaysSaving == nrDaysSaving || myDate == dateFin || myDate == loadDateFin)
        {
            if (aggrVariables.count() > 0)
            {
                logInfoGUI("Time integration from " + saveDateIni.toString("yyyy-MM-dd") + " to " + myDate.toString("yyyy-MM-dd"));
                if (! timeAggregateGrid(saveDateIni, myDate, aggrVariables, false, false)) return false;
            }

            // saving hourly and daily meteo grid data to DB
            logInfoGUI("Saving meteo grid data from " + saveDateIni.toString("yyyy-MM-dd") + " to " + myDate.toString("yyyy-MM-dd"));
            meteoGridDbHandler->saveGridData(&myError, QDateTime(saveDateIni, QTime(1,0,0), Qt::UTC), QDateTime(myDate.addDays(1), QTime(0,0,0), Qt::UTC), varToSave, meteoSettings);

            meteoGridDbHandler->meteoGrid()->emptyGridData(getCrit3DDate(saveDateIni), getCrit3DDate(myDate));

            countDaysSaving = 0;
            saveDateIni = myDate.addDays(1);
        }

        myDate = myDate.addDays(1);
    }

    // restore original proxy grids
    logInfoGUI("Restoring proxy grids");
    if (! loadProxyGrids())
        return false;

    return true;
}

bool PragaProject::interpolationCrossValidationPeriod(QDate dateIni, QDate dateFin, meteoVariable myVar, QString filename)
{
    // check meteo point
    if (! meteoPointsLoaded || nrMeteoPoints == 0)
    {
        logError("No meteo points");
        return false;
    }

    // check dates
    if (dateIni.isNull() || dateFin.isNull() || dateIni > dateFin)
    {
        logError("Wrong period");
        return false;
    }

    // order variables for derived computation
    std::string errString;
    QString myError;
    int myHour;
    QDate myDate = dateIni;
    frequencyType myFreq = getVarFrequency(myVar);

    if (getUseTdVar(myVar))
    {
        logInfoGUI("Loading topographic distance maps...");
        if (! loadTopographicDistanceMaps(true, false))
            return false;
    }

    // loading point data
    logInfoGUI("Loading meteo points data from " + dateIni.addDays(-1).toString("yyyy-MM-dd") + " to " + dateFin.toString("yyyy-MM-dd"));
    // load one day before (for transmissivity)
    if (! loadMeteoPointsData(dateIni, dateFin, myFreq == hourly, myFreq == daily, false))
        return false;

    crossValidationStatistics stats;
    QFile file(filename);
    if (! file.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        myError = "Error writing to file " + filename;
        return false;
    }

    Crit3DTime myTime;

    QTextStream cvOutput(&file);
    cvOutput << "Time,MAE,MBE,RMSE,CRE,R2";

    while (myDate <= dateFin)
    {
        if (getVarFrequency(myVar) == hourly)
        {
            for (myHour = 1; myHour <= 24; myHour++)
            {
                myTime = getCrit3DTime(myDate, myHour);

                if (interpolationCv(myVar, myTime, &stats))
                {
                    cvOutput << getQDateTime(myTime).toString() << '\n';
                    cvOutput << stats.getMeanAbsoluteError() << '\n';
                    cvOutput << stats.getMeanBiasError() << '\n';
                    cvOutput << stats.getRootMeanSquareError() << '\n';
                    cvOutput << stats.getCompoundRelativeError() << '\n';
                    cvOutput << stats.getR2() << '\n';
                }
            }





                /*if (myVar == airRelHumidity && interpolationSettings.getUseDewPoint())
                {
                    if (interpolationSettings.getUseInterpolatedTForRH())
                        passInterpolatedTemperatureToHumidityPoints(getCrit3DTime(myDate, myHour), meteoSettings);
                    if (! interpolationDemMain(airDewTemperature, getCrit3DTime(myDate, myHour), hourlyMeteoMaps->mapHourlyTdew)) return false;
                    hourlyMeteoMaps->computeRelativeHumidityMap(hourlyMeteoMaps->mapHourlyRelHum);
                }
                else if (myVar == windVectorDirection || myVar == windVectorIntensity) {
                    if (! interpolationDemMain(windVectorX, getCrit3DTime(myDate, myHour), getPragaMapFromVar(windVectorX))) return false;
                    if (! interpolationDemMain(windVectorY, getCrit3DTime(myDate, myHour), getPragaMapFromVar(windVectorY))) return false;
                    if (! pragaHourlyMaps->computeWindVector()) return false;
                }
                else if (myVar == leafWetness) {
                    hourlyMeteoMaps->computeLeafWetnessMap() ;
                }
                else if (myVar == referenceEvapotranspiration) {
                    hourlyMeteoMaps->computeET0PMMap(DEM, radiationMaps);
                }
                else {
                    if (! interpolationDemMain(myVar, getCrit3DTime(myDate, myHour), getPragaMapFromVar(myVar))) return false;
                }

                if (myVar == windVectorDirection || myVar == windVectorIntensity)
                {
                    meteoGridDbHandler->meteoGrid()->spatialAggregateMeteoGrid(windVectorX, hourly, getCrit3DDate(myDate), myHour, 0, &DEM, getPragaMapFromVar(windVectorX), interpolationSettings.getMeteoGridAggrMethod());
                    meteoGridDbHandler->meteoGrid()->spatialAggregateMeteoGrid(windVectorY, hourly, getCrit3DDate(myDate), myHour, 0, &DEM, getPragaMapFromVar(windVectorY), interpolationSettings.getMeteoGridAggrMethod());
                    meteoGridDbHandler->meteoGrid()->computeWindVectorHourly(getCrit3DDate(myDate), myHour);
                }
                else
                    meteoGridDbHandler->meteoGrid()->spatialAggregateMeteoGrid(myVar, hourly, getCrit3DDate(myDate), myHour, 0, &DEM, myGrid, interpolationSettings.getMeteoGridAggrMethod());

                */

        }
        else
        {
            if (interpolationCv(myVar, myTime, &stats))
            {
                cvOutput << getQDateTime(myTime).date().toString() << '\n';
                cvOutput << stats.getMeanAbsoluteError() << '\n';
                cvOutput << stats.getMeanBiasError() << '\n';
                cvOutput << stats.getRootMeanSquareError() << '\n';
                cvOutput << stats.getCompoundRelativeError() << '\n';
                cvOutput << stats.getR2() << '\n';
            }
        }

        myDate = myDate.addDays(1);
    }

    return true;
}

bool PragaProject::interpolationMeteoGrid(meteoVariable myVar, frequencyType myFrequency, const Crit3DTime& myTime)
{
    if (meteoGridDbHandler != nullptr)
    {
        if (interpolationSettings.getMeteoGridUpscaleFromDem())
        {
            gis::Crit3DRasterGrid *myRaster = new gis::Crit3DRasterGrid;
            if (!interpolationDemMain(myVar, myTime, myRaster)) return false;

            meteoGridDbHandler->meteoGrid()->spatialAggregateMeteoGrid(myVar, myFrequency, myTime.date, myTime.getHour(),
                            myTime.getMinutes(), &DEM, myRaster, interpolationSettings.getMeteoGridAggrMethod());
        }
        else
        {
            if (! interpolationGrid(myVar, myTime))
                return false;
        }

        meteoGridDbHandler->meteoGrid()->fillMeteoRaster();
    }
    else
    {
        errorString = "Open a Meteo Grid before.";
        return false;
    }

    return true;
}


bool PragaProject::dbMeteoPointDataCount(QDate myFirstDate, QDate myLastDate, meteoVariable myVar, QString dataset, std::vector<int> &myCounter)
{
    frequencyType myFreq = getVarFrequency(myVar);

    if (dataset == "")
    {
        if (! loadMeteoPointsData(myFirstDate, myLastDate, myFreq == hourly, myFreq == daily, true))
            return false;
    }
    else
    {
        if (! loadMeteoPointsData(myFirstDate, myLastDate, myFreq == hourly, myFreq == daily, dataset, true))
            return false;
    }

    QDate myDate = myFirstDate;
    short myHour;
    int counter;
    int i;

    if (modality == MODE_GUI)
        setProgressBar("Counting " + QString::fromStdString(getVariableString(myVar)) + " values...", myFirstDate.daysTo(myLastDate));

    myCounter.clear();

    while (myDate <= myLastDate)
    {
        if (modality == MODE_GUI)
            updateProgressBar(myFirstDate.daysTo(myDate));

        if (myFreq == daily)
        {
            counter = 0;
            for (i = 0; i < nrMeteoPoints; i++)
                if (dataset == "" || meteoPoints[i].dataset == dataset.toStdString())
                    if (! isEqual(meteoPoints[i].getMeteoPointValueD(getCrit3DDate(myDate), myVar, meteoSettings), NODATA)) counter++;

            myCounter.push_back(counter);
        }
        else if (myFreq == hourly)
        {
            for (myHour = 1; myHour <= 24; myHour++)
            {
                counter = 0;
                for (i = 0; i < nrMeteoPoints; i++)
                    if (dataset == "" || meteoPoints[i].dataset == dataset.toStdString())
                        if (! isEqual(meteoPoints[i].getMeteoPointValueH(getCrit3DDate(myDate), myHour, 0, myVar), NODATA)) counter++;

                myCounter.push_back(counter);
            }
        }

        myDate = myDate.addDays(1);
    }

    this->cleanMeteoPointsData();

    if (modality == MODE_GUI) closeProgressBar();

    return true;
}


bool PragaProject::dbMeteoGridMissingData(QDate myFirstDate, QDate myLastDate, meteoVariable myVar, QList <QDate> &dateList, QList <QString> &idList)
{
    frequencyType myFreq = getVarFrequency(myVar);

    QDate myDate;
    short myHour;

    std::string id;

    if (modality == MODE_GUI)
        setProgressBar("Finding missing data for " + QString::fromStdString(getVariableString(myVar)), meteoGridDbHandler->gridStructure().header().nrRows);

    int infoStep = 1;

    for (int row = 0; row < this->meteoGridDbHandler->gridStructure().header().nrRows; row++)
    {
        if (modality == MODE_GUI && (row % infoStep) == 0) updateProgressBar(row);

        for (int col = 0; col < this->meteoGridDbHandler->gridStructure().header().nrCols; col++)
        {
            if (meteoGridDbHandler->meteoGrid()->getMeteoPointActiveId(row, col, &id))
            {
                if (!meteoGridDbHandler->gridStructure().isFixedFields())
                {
                    if (myFreq == daily)
                        meteoGridDbHandler->loadGridDailyData(errorString, QString::fromStdString(id), myFirstDate, myLastDate);
                    else if (myFreq == hourly)
                        meteoGridDbHandler->loadGridHourlyData(errorString, QString::fromStdString(id), QDateTime(myFirstDate,QTime(0,0,0),Qt::UTC), QDateTime(myLastDate,QTime(23,0,0),Qt::UTC));
                }
                else
                {
                    if (myFreq == daily)
                        meteoGridDbHandler->loadGridDailyDataFixedFields(errorString, QString::fromStdString(id), myFirstDate, myLastDate);
                    else if (myFreq ==hourly)
                        meteoGridDbHandler->loadGridHourlyDataFixedFields(errorString, QString::fromStdString(id), QDateTime(myFirstDate, QTime(0,0,0), Qt::UTC), QDateTime(myLastDate,QTime(23,0,0), Qt::UTC));
                }

                for (myDate = myFirstDate; myDate <= myLastDate; myDate = myDate.addDays(1))
                {
                    if (myFreq == daily)
                    {
                        if (isEqual(meteoGridDbHandler->meteoGrid()->meteoPoint(row, col).getMeteoPointValueD(getCrit3DDate(myDate), myVar, meteoSettings), NODATA))
                        {
                            if (dateList.indexOf(myDate) == -1)
                            {
                                dateList.push_back(myDate);
                                idList.push_back(QString::fromStdString(id));
                            }
                        }
                    }
                    else if (myFreq == hourly)
                    {
                        for (myHour = 1; myHour <= 24; myHour++)
                        {
                            if (isEqual(meteoGridDbHandler->meteoGrid()->meteoPoint(row, col).getMeteoPointValueH(getCrit3DDate(myDate), myHour, 0, myVar), NODATA))
                            {
                                if (dateList.indexOf(myDate) == -1)
                                {
                                    dateList.push_back(myDate);
                                    idList.push_back(QString::fromStdString(id));
                                }

                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    if (modality == MODE_GUI) closeProgressBar();

    return true;
}

void PragaProject::showPointStatisticsWidgetPoint(std::string idMeteoPoint)
{
    logInfoGUI("Loading data...");

    // check dates
    QDate firstDaily = meteoPointsDbHandler->getFirstDate(daily, idMeteoPoint).date();
    QDate lastDaily = meteoPointsDbHandler->getLastDate(daily, idMeteoPoint).date();
    bool hasDailyData = !(firstDaily.isNull() || lastDaily.isNull());

    QDateTime firstHourly = meteoPointsDbHandler->getFirstDate(hourly, idMeteoPoint);
    QDateTime lastHourly = meteoPointsDbHandler->getLastDate(hourly, idMeteoPoint);
    bool hasHourlyData = !(firstHourly.isNull() || lastHourly.isNull());

    if (!hasDailyData && !hasHourlyData)
    {
        logInfoGUI("No data.");
        return;
    }

    Crit3DMeteoPoint mp;
    meteoPointsDbHandler->getPropertiesGivenId(QString::fromStdString(idMeteoPoint), mp, gisSettings, errorString);
    logInfoGUI("Loading daily data...");
    meteoPointsDbHandler->loadDailyData(getCrit3DDate(firstDaily), getCrit3DDate(lastDaily), mp);
    logInfoGUI("Loading hourly data...");
    meteoPointsDbHandler->loadHourlyData(getCrit3DDate(firstHourly.date()), getCrit3DDate(lastHourly.date()), mp);
    QList<QString> jointStationsMyMp = meteoPointsDbHandler->getJointStations(QString::fromStdString(idMeteoPoint));
    for (int j = 0; j<jointStationsMyMp.size(); j++)
    {
        QDate lastDateNew = meteoPointsDbHandler->getLastDate(daily, jointStationsMyMp[j].toStdString()).date();
        if (lastDateNew > lastDaily)
        {
            lastDaily = lastDateNew;
        }
    }
    closeLogInfo();
    QList<Crit3DMeteoPoint> meteoPointsWidgetList;
    meteoPointsWidgetList.append(mp);
    double mpUtmX = mp.point.utm.x;
    double mpUtmY = mp.point.utm.y;
    for (int i = 0; i < nrMeteoPoints; i++)
    {
        if (meteoPoints[i].id != idMeteoPoint)
        {
            if (meteoPoints[i].nrObsDataDaysD != 0 || meteoPoints[i].nrObsDataDaysH != 0)
            {
                double utmX = meteoPoints[i].point.utm.x;
                double utmY = meteoPoints[i].point.utm.y;
                float currentDist = gis::computeDistance(mpUtmX, mpUtmY, utmX, utmY);
                if (currentDist < clima->getElabSettings()->getAnomalyPtsMaxDistance())
                {
                    meteoPointsWidgetList.append(meteoPoints[i]);
                }
            }
        }
    }
    bool isGrid = false;
    pointStatisticsWidget = new Crit3DPointStatisticsWidget(isGrid, meteoPointsDbHandler, nullptr, meteoPointsWidgetList, firstDaily, lastDaily, firstHourly, lastHourly,
                                                            meteoSettings, pragaDefaultSettings, &climateParameters, quality);
    return;
}


void PragaProject::showHomogeneityTestWidgetPoint(std::string idMeteoPoint)
{
    logInfoGUI("Loading data...");

    // check dates
    QDate firstDaily = meteoPointsDbHandler->getFirstDate(daily, idMeteoPoint).date();
    QDate lastDaily = meteoPointsDbHandler->getLastDate(daily, idMeteoPoint).date();
    bool hasDailyData = !(firstDaily.isNull() || lastDaily.isNull());

    if (! hasDailyData)
    {
        logInfoGUI("No daily data.");
        return;
    }

    Crit3DMeteoPoint mp;
    meteoPointsDbHandler->getPropertiesGivenId(QString::fromStdString(idMeteoPoint), mp, gisSettings, errorString);

    logInfoGUI("Loading daily data...");
    meteoPointsDbHandler->loadDailyData(getCrit3DDate(firstDaily), getCrit3DDate(lastDaily), mp);
    QList<QString> jointStationsMyMp = meteoPointsDbHandler->getJointStations(QString::fromStdString(idMeteoPoint));
    for (int j = 0; j<jointStationsMyMp.size(); j++)
    {
        QDate lastDateNew = meteoPointsDbHandler->getLastDate(daily, jointStationsMyMp[j].toStdString()).date();
        if (lastDateNew > lastDaily)
        {
            lastDaily = lastDateNew;
        }
    }
    closeLogInfo();
    QList<Crit3DMeteoPoint> meteoPointsNearDistanceList;
    std::vector<float> myDistances;
    std::vector<int> myIndeces;
    QList<std::string> myId;
    meteoPointsNearDistanceList.append(mp);
    double mpUtmX = mp.point.utm.x;
    double mpUtmY = mp.point.utm.y;
    for (int i = 0; i < nrMeteoPoints; i++)
    {
        if (meteoPoints[i].id != idMeteoPoint)
        {
            if (meteoPoints[i].nrObsDataDaysD != 0 || meteoPoints[i].nrObsDataDaysH != 0)
            {
                double utmX = meteoPoints[i].point.utm.x;
                double utmY = meteoPoints[i].point.utm.y;
                float currentDist = gis::computeDistance(mpUtmX, mpUtmY, utmX, utmY);
                if (currentDist < clima->getElabSettings()->getAnomalyPtsMaxDistance() || jointStationsMyMp.contains(QString::fromStdString(meteoPoints[i].id)))
                {
                    meteoPointsNearDistanceList.append(meteoPoints[i]);
                }
                if (meteoPoints[i].lapseRateCode != supplemental)
                {
                    myDistances.push_back(currentDist);
                    myIndeces.push_back(i);
                }
            }
        }
    }

    if (myIndeces.empty())
    {
        logInfoGUI("There are no meteo points as reference...");
        return;
    }

    sorting::quicksortAscendingIntegerWithParameters(myIndeces, myDistances, 0, unsigned(myIndeces.size()-1));
    for (int i = 0; i < myIndeces.size(); i++)
    {
        myId << meteoPoints[myIndeces[i]].id;
    }

    homogeneityWidget = new Crit3DHomogeneityWidget(meteoPointsDbHandler, meteoPointsNearDistanceList, myId, myDistances, jointStationsMyMp, firstDaily, lastDaily,
                                                            meteoSettings, pragaDefaultSettings, &climateParameters, quality);
}


void PragaProject::showSynchronicityTestWidgetPoint(std::string idMeteoPoint)
{
    logInfoGUI("Loading data...");

    // check dates
    QDate firstDaily = meteoPointsDbHandler->getFirstDate(daily, idMeteoPoint).date();
    QDate lastDaily = meteoPointsDbHandler->getLastDate(daily, idMeteoPoint).date();
    bool hasDailyData = !(firstDaily.isNull() || lastDaily.isNull());

    if (!hasDailyData)
    {
        logInfoGUI("No daily data.");
        return;
    }
    closeLogInfo();

    Crit3DMeteoPoint* otherMeteoPoints = new Crit3DMeteoPoint[unsigned(nrMeteoPoints-1)];
    int j = 0;
    int indexMp = 0;
    for (int i=0; i < nrMeteoPoints; i++)
    {
        if (meteoPoints[i].id != idMeteoPoint)
        {
            otherMeteoPoints[j] = meteoPoints[i];
            j = j + 1;
        }
        else
        {
            indexMp = i;
        }
    }

    synchronicityWidget = new Crit3DSynchronicityWidget(meteoPointsDbHandler, meteoPoints[indexMp], gisSettings, firstDaily, lastDaily, meteoSettings, pragaDefaultSettings,
                                                        &climateParameters, quality, interpolationSettings, qualityInterpolationSettings, checkSpatialQuality, otherMeteoPoints, nrMeteoPoints-1);
    connect(synchronicityWidget, &Crit3DSynchronicityWidget::closeSynchWidget,[=]() { this->deleteSynchWidget(); });
    if (synchReferencePoint != "")
    {
        synchronicityWidget->setReferencePointId(synchReferencePoint);
    }
    return;
}

void PragaProject::setSynchronicityReferencePoint(std::string idMeteoPoint)
{
    // check dates
    QDate firstDaily = meteoPointsDbHandler->getFirstDate(daily, idMeteoPoint).date();
    QDate lastDaily = meteoPointsDbHandler->getLastDate(daily, idMeteoPoint).date();
    bool hasDailyData = !(firstDaily.isNull() || lastDaily.isNull());

    if (!hasDailyData)
    {
        logInfoGUI("No daily data.");
        return;
    }
    synchReferencePoint = idMeteoPoint;
    if (synchronicityWidget != nullptr)
    {
        synchronicityWidget->setReferencePointId(synchReferencePoint);
    }
}

void PragaProject::deleteSynchWidget()
{
    synchronicityWidget = nullptr;
}


void PragaProject::showPointStatisticsWidgetGrid(std::string id)
{
    logInfoGUI("Loading data...");

    // check dates
    QDate firstDaily = meteoGridDbHandler->getFirstDailyDate();
    QDate lastDaily = meteoGridDbHandler->getLastDailyDate();
    bool hasDailyData = !(firstDaily.isNull() || lastDaily.isNull());

    QDate firstHourly = meteoGridDbHandler->getFirstHourlyDate();
    QDate lastHourly = meteoGridDbHandler->getLastHourlyDate();
    bool hasHourlyData = !(firstHourly.isNull() || lastHourly.isNull());

    if (!hasDailyData && !hasHourlyData)
    {
        logInfoGUI("No data.");
        return;
    }
    QDateTime firstDateTime = QDateTime(firstHourly, QTime(1,0), Qt::UTC);
    QDateTime lastDateTime = QDateTime(lastHourly.addDays(1), QTime(0,0), Qt::UTC);

    if (!meteoGridDbHandler->gridStructure().isFixedFields())
    {
        logInfoGUI("Loading daily data...");
        meteoGridDbHandler->loadGridDailyData(errorString, QString::fromStdString(id), firstDaily, lastDaily);
        logInfoGUI("Loading hourly data...");
        meteoGridDbHandler->loadGridHourlyData(errorString, QString::fromStdString(id), firstDateTime, lastDateTime);
    }
    else
    {
        logInfoGUI("Loading daily data...");
        meteoGridDbHandler->loadGridDailyDataFixedFields(errorString, QString::fromStdString(id), firstDaily, lastDaily);
        logInfoGUI("Loading hourly data...");
        meteoGridDbHandler->loadGridHourlyDataFixedFields(errorString, QString::fromStdString(id), firstDateTime, lastDateTime);
    }
    closeLogInfo();

    unsigned row;
    unsigned col;
    Crit3DMeteoPoint mp;
    if (meteoGridDbHandler->meteoGrid()->findMeteoPointFromId(&row,&col,id))
    {
        mp = meteoGridDbHandler->meteoGrid()->meteoPoint(row,col);
    }
    else
    {
        return;
    }
    QList<Crit3DMeteoPoint> meteoPointsWidgetList;
    meteoPointsWidgetList.append(mp);
    bool isGrid = true;
    pointStatisticsWidget = new Crit3DPointStatisticsWidget(isGrid, nullptr, meteoGridDbHandler, meteoPointsWidgetList, firstDaily, lastDaily, firstDateTime, lastDateTime,
                                                            meteoSettings, pragaDefaultSettings, &climateParameters, quality);
   return;
}

#ifdef NETCDF
    bool PragaProject::exportMeteoGridToNetCDF(QString fileName, QString title, QString variableName, std::string variableUnit, Crit3DDate myDate, int nDays, int refYearStart, int refYearEnd)
    {
        if (! checkMeteoGridForExport()) return false;

        NetCDFHandler* netcdf = new NetCDFHandler();

        if (! netcdf->createNewFile(fileName.toStdString()))
        {
            logError("Wrong filename: " + fileName);
            return false;
        }

        if (! netcdf->writeMetadata(meteoGridDbHandler->gridStructure().header(), title.toStdString(),
                                    variableName.toStdString(), variableUnit, myDate, nDays, refYearStart, refYearEnd))
        {
            logError("Error in writing geo dimensions.");
            return false;
        }

        if (! netcdf->writeData_NoTime(meteoGridDbHandler->meteoGrid()->dataMeteoGrid))
        {
            logError("Error in writing data.");
            return false;
        }

        netcdf->close();
        delete netcdf;

        return true;
    }

    bool PragaProject::exportXMLElabGridToNetcdf(QString xmlName)
    {
        QString xmlPath = QFileInfo(xmlName).absolutePath()+"/";
        if (meteoGridDbHandler == nullptr)
        {
            return false;
        }
        Crit3DElabList *listXMLElab = new Crit3DElabList();
        Crit3DAnomalyList *listXMLAnomaly = new Crit3DAnomalyList();
        Crit3DDroughtList *listXMLDrought = new Crit3DDroughtList();
        Crit3DPhenologyList *listXMLPhenology = new Crit3DPhenologyList();

        if (xmlName == "")
        {
            errorString = "Empty XML name";
            delete listXMLElab;
            delete listXMLAnomaly;
            delete listXMLDrought;
            delete listXMLPhenology;
            return false;
        }
        if (!parseXMLElaboration(listXMLElab, listXMLAnomaly, listXMLDrought, listXMLPhenology, xmlName, &errorString))
        {
            delete listXMLElab;
            delete listXMLAnomaly;
            delete listXMLDrought;
            delete listXMLPhenology;
            return false;
        }
        if (!listXMLElab->listAll().isEmpty() && listXMLElab->isMeteoGrid() == false)
        {
            errorString = "Datatype is not Grid";
            delete listXMLElab;
            delete listXMLAnomaly;
            delete listXMLDrought;
            delete listXMLPhenology;
            return false;
        }
        if (!listXMLAnomaly->listAll().isEmpty() && listXMLAnomaly->isMeteoGrid() == false)
        {
            errorString = "Datatype is not Grid";
            delete listXMLElab;
            delete listXMLAnomaly;
            delete listXMLDrought;
            delete listXMLPhenology;
            return false;
        }
        if (listXMLDrought->listAll().size() != 0 && listXMLDrought->isMeteoGrid() == false)
        {
            errorString = "Datatype is not Grid";
            delete listXMLElab;
            delete listXMLAnomaly;
            delete listXMLDrought;
            delete listXMLPhenology;
            return false;
        }
        if (listXMLPhenology->listAll().size() != 0 && listXMLPhenology->isMeteoGrid() == false)
        {
            errorString = "Datatype is not Grid";
            delete listXMLElab;
            delete listXMLAnomaly;
            delete listXMLDrought;
            delete listXMLPhenology;
            return false;
        }
        if (listXMLElab->listAll().isEmpty() && listXMLAnomaly->listAll().isEmpty() && listXMLDrought->listAll().size() == 0 && listXMLPhenology->listAll().size() == 0)
        {
            errorString = "There are not valid Elaborations or Anomalies or Drought";
            delete listXMLElab;
            delete listXMLAnomaly;
            delete listXMLDrought;
            delete listXMLPhenology;
            return false;
        }
        if (clima == nullptr)
        {
            clima = new Crit3DClimate();
        }
        if (referenceClima == nullptr && !listXMLAnomaly->listAll().isEmpty())
        {
            referenceClima = new Crit3DClimate();
        }

        for (int i = 0; i<listXMLElab->listAll().size(); i++)
        {
            clima->setVariable(listXMLElab->listVariable()[i]);
            clima->setYearStart(listXMLElab->listYearStart()[i]);
            clima->setYearEnd(listXMLElab->listYearEnd()[i]);
            clima->setPeriodStr(listXMLElab->listPeriodStr()[i]);
            clima->setPeriodType(listXMLElab->listPeriodType()[i]);

            clima->setGenericPeriodDateStart(listXMLElab->listDateStart()[i]);
            clima->setGenericPeriodDateEnd(listXMLElab->listDateEnd()[i]);
            clima->setNYears(listXMLElab->listNYears()[i]);
            clima->setElab1(listXMLElab->listElab1()[i]);

            if (!listXMLElab->listParam1IsClimate()[i])
            {
                clima->setParam1IsClimate(false);
                clima->setParam1(listXMLElab->listParam1()[i]);
            }
            else
            {
                clima->setParam1IsClimate(true);
                clima->setParam1ClimateField(listXMLElab->listParam1ClimateField()[i]);
                int climateIndex = getClimateIndexFromElab(listXMLElab->listDateStart()[i], listXMLElab->listParam1ClimateField()[i]);
                clima->setParam1ClimateIndex(climateIndex);

            }
            clima->setElab2(listXMLElab->listElab2()[i]);
            clima->setParam2(listXMLElab->listParam2()[i]);

            elaborationPointsCycleGrid(false, false);
            meteoGridDbHandler->meteoGrid()->fillMeteoRasterElabValue();

            QString netcdfName;
            QString netcdfTitle;
            if(listXMLElab->listFileName().size() <= i)
            {
                netcdfTitle = "ELAB_"+listXMLElab->listAll()[i];
                netcdfName = xmlPath + "ELAB_"+listXMLElab->listAll()[i]+".nc";
            }
            else
            {
                netcdfTitle = listXMLElab->listFileName()[i];
                netcdfName = xmlPath + listXMLElab->listFileName()[i]+".nc";
            }
            QDate dateEnd = QDate(listXMLElab->listYearEnd()[i] + listXMLElab->listNYears()[i], listXMLElab->listDateEnd()[i].month(), listXMLElab->listDateEnd()[i].day());
            QDate dateStart = QDate(listXMLElab->listYearStart()[i], listXMLElab->listDateStart()[i].month(), listXMLElab->listDateStart()[i].day());
            int nDays = dateStart.daysTo(dateEnd);
            exportMeteoGridToNetCDF(netcdfName, netcdfTitle, QString::fromStdString(MapDailyMeteoVarToString.at(listXMLElab->listVariable()[i])), getUnitFromVariable(listXMLElab->listVariable()[i]), getCrit3DDate(dateStart), nDays, 0, 0);
            // reset param
            clima->resetParam();
            // reset current values
            clima->resetCurrentValues();
        }

        for (int i = 0; i<listXMLAnomaly->listAll().size(); i++)
        {
            clima->setVariable(listXMLAnomaly->listVariable()[i]);
            clima->setYearStart(listXMLAnomaly->listYearStart()[i]);
            clima->setYearEnd(listXMLAnomaly->listYearEnd()[i]);
            clima->setPeriodStr(listXMLAnomaly->listPeriodStr()[i]);
            clima->setPeriodType(listXMLAnomaly->listPeriodType()[i]);

            clima->setGenericPeriodDateStart(listXMLAnomaly->listDateStart()[i]);
            clima->setGenericPeriodDateEnd(listXMLAnomaly->listDateEnd()[i]);
            clima->setNYears(listXMLAnomaly->listNYears()[i]);
            clima->setElab1(listXMLAnomaly->listElab1()[i]);

            if (!listXMLAnomaly->listParam1IsClimate()[i])
            {
                clima->setParam1IsClimate(false);
                clima->setParam1(listXMLAnomaly->listParam1()[i]);
            }
            else
            {
                clima->setParam1IsClimate(true);
                clima->setParam1ClimateField(listXMLAnomaly->listParam1ClimateField()[i]);
                int climateIndex = getClimateIndexFromElab(listXMLAnomaly->listDateStart()[i], listXMLElab->listParam1ClimateField()[i]);
                clima->setParam1ClimateIndex(climateIndex);

            }
            clima->setElab2(listXMLAnomaly->listElab2()[i]);
            clima->setParam2(listXMLAnomaly->listParam2()[i]);

            referenceClima->setVariable(listXMLAnomaly->listVariable()[i]);
            referenceClima->setYearStart(listXMLAnomaly->listRefYearStart()[i]);
            referenceClima->setYearEnd(listXMLAnomaly->listRefYearEnd()[i]);
            referenceClima->setPeriodStr(listXMLAnomaly->listRefPeriodStr()[i]);
            referenceClima->setPeriodType(listXMLAnomaly->listRefPeriodType()[i]);

            referenceClima->setGenericPeriodDateStart(listXMLAnomaly->listRefDateStart()[i]);
            referenceClima->setGenericPeriodDateEnd(listXMLAnomaly->listRefDateEnd()[i]);
            referenceClima->setNYears(listXMLAnomaly->listRefNYears()[i]);
            referenceClima->setElab1(listXMLAnomaly->listRefElab1()[i]);

            if (!listXMLAnomaly->listRefParam1IsClimate()[i])
            {
                referenceClima->setParam1IsClimate(false);
                referenceClima->setParam1(listXMLAnomaly->listRefParam1()[i]);
            }
            else
            {
                referenceClima->setParam1IsClimate(true);
                referenceClima->setParam1ClimateField(listXMLAnomaly->listRefParam1ClimateField()[i]);
                int climateIndex = getClimateIndexFromElab(listXMLAnomaly->listRefDateStart()[i], listXMLAnomaly->listRefParam1ClimateField()[i]);
                referenceClima->setParam1ClimateIndex(climateIndex);
            }
            referenceClima->setElab2(listXMLAnomaly->listRefElab2()[i]);
            referenceClima->setParam2(listXMLAnomaly->listRefParam2()[i]);

            elaborationPointsCycleGrid(false, false);
            qDebug() << "--------------------------------------------------";
            elaborationPointsCycleGrid(true, false);

            if (!listXMLAnomaly->isPercentage()[i])
            {
                meteoGridDbHandler->meteoGrid()->fillMeteoRasterAnomalyValue();
            }
            else
            {
                meteoGridDbHandler->meteoGrid()->fillMeteoRasterAnomalyPercValue();
            }
            QString netcdfName;
            QString netcdfTitle;
            if (listXMLAnomaly->listFileName().size() <= i)
            {
                netcdfTitle = "ANOMALY_"+listXMLAnomaly->listAll()[i];
                netcdfName = xmlPath + "ANOMALY_"+listXMLAnomaly->listAll()[i]+".nc";
            }
            else
            {
                netcdfTitle = listXMLAnomaly->listFileName()[i];
                netcdfName = xmlPath + listXMLAnomaly->listFileName()[i]+".nc";
            }

            QDate dateEnd = QDate(listXMLAnomaly->listYearEnd()[i] + listXMLAnomaly->listNYears()[i], listXMLAnomaly->listDateEnd()[i].month(), listXMLAnomaly->listDateEnd()[i].day());
            QDate dateStart = QDate(listXMLAnomaly->listYearStart()[i], listXMLAnomaly->listDateStart()[i].month(), listXMLAnomaly->listDateStart()[i].day());

            int nDays = dateStart.daysTo(dateEnd);
            exportMeteoGridToNetCDF(netcdfName, netcdfTitle, QString::fromStdString(MapDailyMeteoVarToString.at(listXMLAnomaly->listVariable()[i])), getUnitFromVariable(listXMLAnomaly->listVariable()[i]), getCrit3DDate(dateStart),
                                    nDays, listXMLAnomaly->listRefYearStart()[i], listXMLAnomaly->listRefYearEnd()[i]);
            // reset param
            clima->resetParam();
            referenceClima->resetParam();
            // reset current values
            clima->resetCurrentValues();
            referenceClima->resetCurrentValues();
        }

        for (unsigned int i = 0; i < listXMLDrought->listAll().size(); i++)
        {
            computeDroughtIndexGrid(listXMLDrought->listIndex()[i], listXMLDrought->listYearStart()[i], listXMLDrought->listYearEnd()[i], listXMLDrought->listDate()[i], listXMLDrought->listTimescale()[i], listXMLDrought->listVariable()[i]);
            meteoGridDbHandler->meteoGrid()->fillMeteoRasterElabValue();

            QString netcdfName;
            if(listXMLDrought->listFileName().size() <= i)
            {
                netcdfName = xmlPath + listXMLDrought->listAll()[i]+".nc";
            }
            else
            {
                netcdfName = xmlPath + listXMLDrought->listFileName()[i]+".nc";
            }
            if (listXMLDrought->listIndex()[i] == INDEX_SPI)
            {
                int firstMonth = listXMLDrought->listDate()[i].month() - listXMLDrought->listTimescale()[i]+1;
                QDate dateStart;
                if (firstMonth <= 0 && firstMonth >= -11)
                {
                        firstMonth = 12 + firstMonth;
                        dateStart.setDate(listXMLDrought->listDate()[i].year()-1, firstMonth, 1);
                }
                else if (firstMonth < -11)
                {
                        firstMonth = 24 + firstMonth;
                        dateStart.setDate(listXMLDrought->listDate()[i].year()-2, firstMonth, 1);
                }
                else
                {
                    dateStart.setDate(listXMLDrought->listDate()[i].year(), firstMonth, 1);
                }
                int lastDay = listXMLDrought->listDate()[i].daysInMonth();
                QDate dateEnd(listXMLDrought->listDate()[i].year(),listXMLDrought->listDate()[i].month(),lastDay);
                int nDays = dateStart.daysTo(dateEnd);
                exportMeteoGridToNetCDF(netcdfName, "Standardized Precipitation Index", "SPI at "+QString::number(listXMLDrought->listTimescale()[i])+" month scale", "", getCrit3DDate(dateStart), nDays, listXMLDrought->listYearStart()[i], listXMLDrought->listYearEnd()[i]);
            }
            else if (listXMLDrought->listIndex()[i] == INDEX_SPEI )
            {
                int fistMonth = listXMLDrought->listDate()[i].month() - listXMLDrought->listTimescale()[i]+1;
                QDate dateStart;
                if (fistMonth <= 0 && fistMonth >= -11)
                {
                        fistMonth = 12 + fistMonth;
                        dateStart.setDate(listXMLDrought->listDate()[i].year()-1, fistMonth, 1);
                }
                else if (fistMonth < -11)
                {
                        fistMonth = 24 + fistMonth;
                        dateStart.setDate(listXMLDrought->listDate()[i].year()-2, fistMonth, 1);
                }
                else
                {
                    dateStart.setDate(listXMLDrought->listDate()[i].year(), fistMonth, 1);
                }
                int lastDay = listXMLDrought->listDate()[i].daysInMonth();
                QDate dateEnd(listXMLDrought->listDate()[i].year(),listXMLDrought->listDate()[i].month(),lastDay);
                int nDays = dateStart.daysTo(dateEnd);
                exportMeteoGridToNetCDF(netcdfName, "Standardized Precipitation Evapotranspiration Index", "SPEI at "+QString::number(listXMLDrought->listTimescale()[i])+" month scale", "", getCrit3DDate(dateStart), nDays, listXMLDrought->listYearStart()[i], listXMLDrought->listYearEnd()[i]);
            }
            else if (listXMLDrought->listIndex()[i] == INDEX_DECILES)
            {
                QDate dateStart(listXMLDrought->listDate()[i].year(), listXMLDrought->listDate()[i].month(), 1);
                int lastDay = listXMLDrought->listDate()[i].daysInMonth();
                QDate dateEnd(listXMLDrought->listDate()[i].year(),listXMLDrought->listDate()[i].month(),lastDay);
                int nDays = dateStart.daysTo(dateEnd);
                exportMeteoGridToNetCDF(netcdfName, "Deciles Index", "precipitation sum percentile rank", getUnitFromVariable(listXMLDrought->listVariable()[i]), getCrit3DDate(dateStart), nDays, listXMLDrought->listYearStart()[i], listXMLDrought->listYearEnd()[i]);
            }
        }

        for (unsigned int i = 0; i<listXMLPhenology->listAll().size(); i++)
        {
            // TO DO
        }

        delete listXMLElab;
        delete listXMLAnomaly;
        delete listXMLDrought;
        delete listXMLPhenology;
        return true;
    }

#endif

/*
bool PragaProject::loadForecastToGrid(QString fileName, bool overWrite, bool checkTables)
{
    if (! QFile(fileName).exists() || ! QFileInfo(fileName).isFile())
    {
        logError("Missing file: " + fileName);
        return false;
    }
    if (meteoGridDbHandler == nullptr)
    {
        logError("Open a Meteo Grid before.");
        return false;
    }
    ForecastDataset dataset;
    dataset.importForecastData(fileName);
    return true;
}
*/

bool PragaProject::parserXMLImportExportData(QString xmlName, bool isGrid)
{
    if (! QFile(xmlName).exists() || ! QFileInfo(xmlName).isFile())
    {
        logError("Missing file: " + xmlName);
        return false;
    }
    if (isGrid)
    {
        inOutData = new InOutDataXML(isGrid, nullptr, meteoGridDbHandler, xmlName);
    }
    else
    {
        inOutData = new InOutDataXML(isGrid, meteoPointsDbHandler, nullptr, xmlName);
    }
    errorString = "";
    if (!inOutData->parserXML(&errorString))
    {
        logError(errorString);
        delete inOutData;
        return false;
    }
    return true;
}


bool PragaProject::loadXMLImportData(QString fileName)
{
    if (! QFile(fileName).exists() || ! QFileInfo(fileName).isFile())
    {
        logError("Missing file: " + fileName);
        return false;
    }

    errorString = "";
    if (! inOutData->importDataMain(fileName, errorString))
    {
        logError();
        return false;
    }

    return true;
}


bool PragaProject::loadXMLExportData(QString code, QDateTime myFirstTime, QDateTime myLastTime)
{
    errorString = "";
    QString filename = inOutData->parseXMLFilename(code);
    if (filename.isEmpty())
    {
        errorString = "Invalid filename" ;
        return false;
    }
    QString variable = inOutData->getVariableExport();
    meteoVariable meteoVar = getMeteoVar(variable.toStdString());
    if (meteoVar == noMeteoVar)
    {
        errorString = "Unknown meteo variable: " + variable;
        return false;
    }
    QString fixedString = "";
    int pointCodeFirstChar = inOutData->getPointCodeFirstChar();
    if (pointCodeFirstChar != NODATA)
    {
        fixedString = code;
        for (int i = 0; i<pointCodeFirstChar-1; i++)
        {
                fixedString.insert(0, " ");
        }
    }

    int variableCodeFirstChar = inOutData->getVariableCodeFirstChar();
    int whiteSpaces = variableCodeFirstChar - (fixedString.length()+1);
    for (int i = 0; i < whiteSpaces; i++)
    {
        fixedString.append(" ");
    }

    QString attribute = inOutData->getVariableCodeAttribute();
    if (! attribute.isEmpty())
    {
        fixedString = fixedString + attribute;
    }

    int timeFirstChar = inOutData->getTimeFirstChar();
    whiteSpaces = timeFirstChar - (fixedString.length()+1);
    for (int i = 0; i < whiteSpaces; i++)
    {
        fixedString.append(" ");
    }

    QString variableAlign = inOutData->getVariableAlign();
    int variableFirstChar = inOutData->getVariableFirstChar();
    int variableNrChar = inOutData->getVariableNrChar();
    QString variableFormat = inOutData->getVariableFormat();
    QChar charFormat = variableFormat[variableFormat.length()-1];
    int nDecimals = variableFormat.mid(variableFormat.length()-2,1).toInt();
    if (variableAlign.isEmpty())
    {
        variableAlign = "right"; //default
    }
    else if (variableAlign != "right" && variableAlign != "left")
    {
        errorString = "Invalid alignment: " + variableAlign;
        return false;
    }
    QString flagAccepted = inOutData->getVariableFlagAccepted();
    int flagFirstChar = 0;
    if (!flagAccepted.isEmpty())
    {
        flagFirstChar = inOutData->getVariableFlagFirstChar();
    }
    QString missingValueStr = QString::number(inOutData->getFormatMissingValue());
    QString timeType = inOutData->getTimeType();
    frequencyType freq;
    if (timeType == "daily" || timeType == "DAILY" || timeType == "D")
    {
        freq = daily;
    } else if (timeType == "hourly" || timeType == "HOURLY" || timeType == "H")
    {
        freq = hourly;
    }
    else
    {
        errorString = "Invalid time type: " + timeType;
        return false;
    }


    std::vector<QString> dateStr;
    std::vector<float> values = meteoPointsDbHandler->exportAllDataVar(&errorString, freq, meteoVar, code, myFirstTime, myLastTime, dateStr);
    if (values.size() == 0)
    {
        // non ci sono dati per quella variabile, non produce alcun file e va avanti con eventuali altri meteo points
        return true;
    }
    QFile file(filename);
    file.open(QIODevice::WriteOnly | QIODevice::Text);
    QTextStream out(&file);
    QDate myDate;
    QTime myTime;
    QDateTime myDateTime;
    QString myDateStr;
    for (int i = 0; i<values.size(); i++)
    {
        if (freq == daily)
        {
            myDate = QDate::fromString(dateStr[i], "yyyy-MM-dd");
            myDateStr = myDate.toString(inOutData->getTimeFormat());
        }
        else if (freq == hourly)
        {
            myDate = QDate::fromString(dateStr[i].mid(0,10), "yyyy-MM-dd");
            myTime = QTime::fromString(dateStr[i].mid(11,8), "hh:mm");
            myDateTime = QDateTime(myDate, myTime, Qt::UTC);
            myDateStr = myDateTime.toString(inOutData->getTimeFormat());
        }

        if (!dateStr[i].isEmpty() && myDateStr.isEmpty())
        {
            errorString = "Invalid date format: " + inOutData->getTimeFormat();
            file.close();
            return false;
        }
        QString fixedStringAddDate = fixedString+myDateStr;
        out << fixedStringAddDate;
        whiteSpaces = variableFirstChar - (fixedStringAddDate.length()+1);
        for (int i = 0; i<whiteSpaces; i++)
        {
            out << " " ;
        }

        QString myValue;
        if (values[i] == NODATA)
        {
            myValue = missingValueStr;
        }
        else
        {
            if (charFormat == 'f')
            {
                myValue = QString::number(values[i],'f',nDecimals);
            }
            else if (charFormat == 'd')
            {
                myValue = QString::number(values[i],'d',nDecimals);
            }
            else if (charFormat == 'd'|| charFormat == 'i')
            {
                myValue = QString::number(values[i],'f',nDecimals);
            }
            else if (charFormat == 'e')
            {
                myValue = QString::number(values[i],'e',nDecimals);
            }
            else if (charFormat == 'g')
            {
                myValue = QString::number(values[i],'g',nDecimals);
            }
            else
            {
                myValue = QString::number(values[i],'f',nDecimals);  // default
            }
        }

        if (variableAlign == "left")
        {
            myValue = myValue.leftJustified(variableNrChar, ' ');
            out << myValue;
        }
        else
        {
            myValue = myValue.rightJustified(variableNrChar, ' ');
            out << myValue;
        }
        if (values[i] != NODATA && !flagAccepted.isEmpty())
        {
            whiteSpaces = flagFirstChar - (variableFirstChar + variableNrChar);
            for (int i = 0; i<whiteSpaces; i++)
            {
                    out << " " ;
            }
            out << flagAccepted;
        }
        out <<"\n";
    }
    file.close();

    return true;
}

// LC 2 funzioni separate per gliglie e punti per eventualmente diversificare anche i dati da esportare (es. le griglie hanno anche i mensili)
bool PragaProject::loadXMLExportDataGrid(QString code, QDateTime myFirstTime, QDateTime myLastTime)
{
    errorString = "";
    QString filename = inOutData->parseXMLFilename(code);
    if (filename.isEmpty())
    {
        errorString = "Invalid filename" ;
        return false;
    }

    QString variable = inOutData->getVariableExport();
    meteoVariable meteoVar = getMeteoVar(variable.toStdString());
    if (meteoVar == noMeteoVar)
    {
        errorString = "Unknown meteo variable: " + variable;
        return false;
    }

    QString fixedString = "";
    int pointCodeFirstChar = inOutData->getPointCodeFirstChar();
    if (pointCodeFirstChar != NODATA)
    {
        fixedString = code;
        for (int i = 0; i < pointCodeFirstChar-1; i++)
        {
            fixedString.insert(0, " ");
        }
    }

    int variableCodeFirstChar = inOutData->getVariableCodeFirstChar();
    int whiteSpaces = variableCodeFirstChar - (fixedString.length()+1);
    for (int i = 0; i < whiteSpaces; i++)
    {
        fixedString.append(" ");
    }

    QString attribute = inOutData->getVariableCodeAttribute();
    if (! attribute.isEmpty())
    {
        fixedString = fixedString + attribute;
    }

    int timeFirstChar = inOutData->getTimeFirstChar();
    whiteSpaces = timeFirstChar - (fixedString.length()+1);
    for (int i = 0; i < whiteSpaces; i++)
    {
        fixedString.append(" ");
    }

    QString variableAlign = inOutData->getVariableAlign();
    int variableFirstChar = inOutData->getVariableFirstChar();
    int variableNrChar = inOutData->getVariableNrChar();
    QString variableFormat = inOutData->getVariableFormat();
    QChar charFormat = variableFormat[variableFormat.length()-1];
    int nDecimals = variableFormat.mid(variableFormat.length()-2,1).toInt();

    if (variableAlign.isEmpty())
    {
        variableAlign = "right"; //default
    }
    else if (variableAlign != "right" && variableAlign != "left")
    {
        errorString = "Invalid alignment: " + variableAlign;
        return false;
    }

    QString flagAccepted = inOutData->getVariableFlagAccepted();
    int flagFirstChar = 0;
    if (!flagAccepted.isEmpty())
    {
        flagFirstChar = inOutData->getVariableFlagFirstChar();
    }
    QString missingValueStr = QString::number(inOutData->getFormatMissingValue());
    QString timeType = inOutData->getTimeType();
    frequencyType freq;
    if (timeType == "daily" || timeType == "DAILY" || timeType == "D")
    {
        freq = daily;
    } else if (timeType == "hourly" || timeType == "HOURLY" || timeType == "H")
    {
        freq = hourly;
    }
    else
    {
        errorString = "Invalid time type: " + timeType;
        return false;
    }


    std::vector<QString> dateStr;
    std::vector<float> values = meteoGridDbHandler->exportAllDataVar(&errorString, freq, meteoVar, code, myFirstTime, myLastTime, dateStr);
    if (values.size() == 0)
    {
        errorString = code + " has no data for variable: " + variable;
        return false;
    }
    QFile file(filename);
    file.open(QIODevice::WriteOnly | QIODevice::Text);
    QTextStream out(&file);
    QDate myDate;
    QTime myTime;
    QDateTime myDateTime;
    QString myDateStr;
    for (int i = 0; i<values.size(); i++)
    {
        if (freq == daily)
        {
            myDate = QDate::fromString(dateStr[i], "yyyy-MM-dd");
            myDateStr = myDate.toString(inOutData->getTimeFormat());
        }
        else if (freq == hourly)
        {
            myDate = QDate::fromString(dateStr[i].mid(0,10), "yyyy-MM-dd");
            myTime = QTime::fromString(dateStr[i].mid(11,8), "hh:mm");
            myDateTime = QDateTime(myDate, myTime, Qt::UTC);
            myDateStr = myDateTime.toString(inOutData->getTimeFormat());
        }

        if (!dateStr[i].isEmpty() && myDateStr.isEmpty())
        {
            errorString = "Invalid date format: " + inOutData->getTimeFormat();
            file.close();
            return false;
        }
        QString fixedStringAddDate = fixedString+myDateStr;
        out << fixedStringAddDate;
        whiteSpaces = variableFirstChar - (fixedStringAddDate.length()+1);
        for (int i = 0; i<whiteSpaces; i++)
        {
            out << " " ;
        }

        QString myValue;
        if (values[i] == NODATA)
        {
            myValue = missingValueStr;
        }
        else
        {
            if (charFormat == 'f')
            {
                    myValue = QString::number(values[i],'f',nDecimals);
            }
            else if (charFormat == 'd')
            {
                    myValue = QString::number(values[i],'d',nDecimals);
            }
            else if (charFormat == 'd'|| charFormat == 'i')
            {
                    myValue = QString::number(values[i],'f',nDecimals);
            }
            else if (charFormat == 'e')
            {
                    myValue = QString::number(values[i],'e',nDecimals);
            }
            else if (charFormat == 'g')
            {
                    myValue = QString::number(values[i],'g',nDecimals);
            }
            else
            {
                    myValue = QString::number(values[i],'f',nDecimals);  // default
            }
        }

        if (variableAlign == "left")
        {
            myValue = myValue.leftJustified(variableNrChar, ' ');
            out << myValue;
        }
        else
        {
            myValue = myValue.rightJustified(variableNrChar, ' ');
            out << myValue;
        }
        if (values[i] != NODATA && !flagAccepted.isEmpty())
        {
            whiteSpaces = flagFirstChar - (variableFirstChar + variableNrChar);
            for (int i = 0; i<whiteSpaces; i++)
            {
                    out << " " ;
            }
            out << flagAccepted;
        }
        out <<"\n";
    }
    file.close();

    return true;
}

bool PragaProject::monthlyAggregateVariablesGrid(const QDate &firstDate, const QDate &lastDate, QList<meteoVariable> &variablesList)
{
    // check meteo grid
    if (! meteoGridLoaded)
    {
        logError("No meteo grid");
        return false;
    }

    // check dates
    if (firstDate.isNull() || lastDate.isNull() || firstDate > lastDate)
    {
        logError("Wrong period");
        return false;
    }

    std::vector <meteoVariable> dailyMeteoVar;
    for (int i = 0; i < variablesList.size(); i++)
    {
        meteoVariable dailyVar = updateMeteoVariable(variablesList[i], daily);
        if (dailyVar != noMeteoVar)
        {
            dailyMeteoVar.push_back(dailyVar);
        }
    }

    if (! monthlyAggregateDataGrid(meteoGridDbHandler, firstDate, lastDate, dailyMeteoVar, meteoSettings,
                                  quality, &climateParameters, errorString))
        return false;

    meteoGridDbHandler->updateMeteoGridDate(errorString);

    return true;
}


bool PragaProject::computeDroughtIndexGrid(droughtIndex index, int firstYear, int lastYear, QDate date, int timescale, meteoVariable myVar)
{
    // check meteo grid
    if (! meteoGridLoaded)
    {
        logError("No meteo grid");
        return false;
    }

    // check dates
    if (firstYear > lastYear)
    {
        logError("Wrong years");
        return false;
    }

    QDate firstDate(firstYear, 1, 1);
    QDate lastDate;
    int maxYear = std::max(lastYear, date.year());
    if (maxYear == QDate::currentDate().year())
    {
        lastDate.setDate(maxYear, QDate::currentDate().month(), 1);
    }
    else
    {
        lastDate.setDate(maxYear,12,1);
    }

    logInfoGUI("Load monthly grid data...");
    qApp->processEvents();
    bool isOk = meteoGridDbHandler->loadGridAllMonthlyData(errorString, firstDate, lastDate);
    closeLogInfo();

    if (! isOk)
    {
        if (! errorString.isEmpty())
            logError();
        else
            logError("No monthly data.");

        return false;
    }

    isOk = false;
    setProgressBar("Drought Index - Meteo Grid", meteoGridDbHandler->meteoGrid()->gridStructure().header().nrRows);
    for (unsigned row = 0; row < unsigned(meteoGridDbHandler->meteoGrid()->gridStructure().header().nrRows); row++)
    {
        updateProgressBar(row);

        for (unsigned col = 0; col < unsigned(meteoGridDbHandler->meteoGrid()->gridStructure().header().nrCols); col++)
        {
            if (meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col)->active)
            {
                Drought mydrought(index, firstYear, lastYear, getCrit3DDate(date), meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col), meteoSettings);
                if (timescale > 0)
                {
                    mydrought.setTimeScale(timescale);
                }
                meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col)->elaboration = NODATA;

                if (index == INDEX_DECILES)
                {
                    if (myVar != noMeteoVar)
                    {
                        mydrought.setMyVar(myVar);
                    }
                    if (mydrought.computePercentileValuesCurrentDay())
                    {
                        meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col)->elaboration = mydrought.getCurrentPercentileValue();
                    }
                }
                else if (index == INDEX_SPI || index == INDEX_SPEI)
                {
                    meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col)->elaboration = mydrought.computeDroughtIndex();
                }

                if (meteoGridDbHandler->meteoGrid()->meteoPointPointer(row,col)->elaboration != NODATA)
                {
                    isOk = true;
                }
            }
        }
    }
    closeProgressBar();

    if (! isOk)
        logError("Missing data.");

    return isOk;
}


// assume che sia stato caricato un progetto con solo aggregationDb
bool PragaProject::computeDroughtIndexPoint(droughtIndex index, int timescale, int refYearStart, int refYearEnd)
{
    if (! aggregationDbHandler)
    {
        logError("No db aggregation");
        return false;
    }

    // check meteo point
    if (! meteoPointsLoaded)
    {
        logError("No meteo point");
        return false;
    }

    // check ref years
    if (refYearStart > refYearEnd)
    {
        logError("Wrong reference years");
        return false;
    }

    QDate firstDate = meteoPointsDbHandler->getFirstDate(daily).date();
    QDate lastDate = meteoPointsDbHandler->getLastDate(daily).date();
    QDate myDate;
    bool loadHourly = false;
    bool loadDaily = true;
    bool showInfo = true;
    float value = NODATA;
    QString indexStr;
    QList<QString> listEntries;

    if (index == INDEX_SPI)
    {
        indexStr = "SPI";
    }
    else if (index == INDEX_SPEI)
    {
        indexStr = "SPEI";
    }
    else if (index == INDEX_DECILES)
    {
        indexStr = "DECILES";
    }
    else
    {
        logError("Unknown index");
        return false;
    }

    if (! loadMeteoPointsData(firstDate, lastDate, loadHourly, loadDaily, showInfo))
    {
        logError("There are no data");
        return false;
    }

    int step = 0;
    if (showInfo)
    {
        QString infoStr = "Compute drought - Meteo Points";
        step = setProgressBar(infoStr, nrMeteoPoints);
    }

    std::vector<meteoVariable> dailyMeteoVar;
    dailyMeteoVar.push_back(dailyPrecipitation);
    dailyMeteoVar.push_back(dailyReferenceEvapotranspirationHS);
    int nrMonths = (lastDate.year()-firstDate.year())*12+lastDate.month()-(firstDate.month()-1);

    for (int i=0; i < nrMeteoPoints; i++)
    {
        myDate = firstDate;
        if (showInfo && (i % step) == 0)
        {
            updateProgressBar(i);
        }

        // compute monthly data
        meteoPoints[i].initializeObsDataM(nrMonths, firstDate.month(), firstDate.year());
        for(int j = 0; j < dailyMeteoVar.size(); j++)
        {
            meteoPoints[i].computeMonthlyAggregate(getCrit3DDate(firstDate), getCrit3DDate(lastDate), dailyMeteoVar[j], meteoSettings, quality, &climateParameters);
        }

        while(myDate <= lastDate)
        {
            Drought myDrought(index, refYearStart, refYearEnd, getCrit3DDate(myDate), &(meteoPoints[i]), meteoSettings);
            if (timescale > 0)
            {
                myDrought.setTimeScale(timescale);
            }
            if (index == INDEX_DECILES)
            {
                if (myDrought.computePercentileValuesCurrentDay())
                {
                    value = myDrought.getCurrentPercentileValue();
                }
            }
            else if (index == INDEX_SPI || index == INDEX_SPEI)
            {
                value = myDrought.computeDroughtIndex();
            }

            listEntries.push_back(QString("(%1,%2,'%3',%4,%5,'%6',%7,%8)").arg(QString::number(myDate.year()), QString::number(myDate.month()),
                                                                               QString::fromStdString(meteoPoints[i].id),
                                                                               QString::number(refYearStart), QString::number(refYearEnd),
                                                                               indexStr, QString::number(timescale), QString::number(value)));
            myDate = myDate.addMonths(1);
        }
    }

    if (listEntries.empty())
    {
        logError("Failed to compute droughtIndex ");
        return false;
    }

    if (! aggregationDbHandler->writeDroughtDataList(listEntries, &errorString))
    {
        logError("Failed to write droughtIndex "+errorString);
        return false;
    }

    if (showInfo)
    {
        logInfo("droughtIndex saved");
    }

    return true;
}


bool PragaProject::computeDroughtIndexPointGUI(droughtIndex index, int timescale, int refYearStart, int refYearEnd, QDate myDate)
{
    // check meteo point
    if (! meteoPointsLoaded)
    {
        logError("No meteo point");
        return false;
    }

    // check ref years
    if (refYearStart > refYearEnd)
    {
        logError("Wrong reference years");
        return false;
    }

    QDate firstDate(refYearStart,1,1);
    QDate lastDate;
    int maxYear = std::max(refYearEnd,myDate.year());
    if (maxYear == QDate::currentDate().year())
    {
        lastDate.setDate(maxYear, QDate::currentDate().month(),1);
    }
    else
    {
        lastDate.setDate(maxYear,12,1);
    }

    QString indexStr;
    if (index == INDEX_SPI)
    {
        indexStr = "SPI";
    }
    else if (index == INDEX_SPEI)
    {
        indexStr = "SPEI";
    }
    else if (index == INDEX_DECILES)
    {
        indexStr = "DECILES";
    }
    else
    {
        logError("Unknown index");
        return false;
    }

    bool loadHourly = false;
    bool loadDaily = true;
    bool showInfo = true;
    if (! loadMeteoPointsData(firstDate, lastDate, loadHourly, loadDaily, showInfo))
    {
        logError("No data.");
        return false;
    }

    int step = 0;
    if (showInfo)
    {
        QString infoStr = "Compute drought - Meteo Points";
        step = setProgressBar(infoStr, nrMeteoPoints);
    }

    std::vector<meteoVariable> dailyMeteoVar;
    dailyMeteoVar.push_back(dailyPrecipitation);
    dailyMeteoVar.push_back(dailyReferenceEvapotranspirationHS);

    bool isOk = false;
    float value = NODATA;

    int nrMonths = (lastDate.year()-firstDate.year())*12 + lastDate.month() - (firstDate.month()-1);
    for (int i = 0; i < nrMeteoPoints; i++)
    {
        if (showInfo && (i % step) == 0)
        {
            updateProgressBar(i);
        }

        // compute monthly data
        meteoPoints[i].initializeObsDataM(nrMonths, firstDate.month(), firstDate.year());
        for(int j = 0; j < dailyMeteoVar.size(); j++)
        {
            meteoPoints[i].computeMonthlyAggregate(getCrit3DDate(firstDate), getCrit3DDate(lastDate), dailyMeteoVar[j], meteoSettings, quality, &climateParameters);
        }

        Drought myDrought(index, refYearStart, refYearEnd, getCrit3DDate(myDate), &(meteoPoints[i]), meteoSettings);

        if (timescale > 0)
        {
            myDrought.setTimeScale(timescale);
        }

        if (index == INDEX_DECILES)
        {
            if (myDrought.computePercentileValuesCurrentDay())
            {
                value = myDrought.getCurrentPercentileValue();
            }
            else
            {
                value = NODATA;
            }
        }
        else if (index == INDEX_SPI || index == INDEX_SPEI)
        {
            value = myDrought.computeDroughtIndex();
        }

        meteoPoints[i].elaboration = value;

        if (value != NODATA)
        {
            isOk = true;
        }
    }

    return isOk;
}


bool PragaProject::activeMeteoGridCellsWithDEM()
{

    if (modality == MODE_GUI)
        setProgressBar("Active cells... ", meteoGridDbHandler->gridStructure().header().nrRows);

    int infoStep = 1;
    bool excludeNoData = true;
    QList<QString> idActiveList;
    QList<QString> idNotActiveList;
    float minCoverage = clima->getElabSettings()->getGridMinCoverage();

    for (int row = 0; row < this->meteoGridDbHandler->gridStructure().header().nrRows; row++)
    {
        if (modality == MODE_GUI && (row % infoStep) == 0) updateProgressBar(row);

        for (int col = 0; col < this->meteoGridDbHandler->gridStructure().header().nrCols; col++)
        {
            meteoGridDbHandler->meteoGrid()->assignCellAggregationPoints(row, col, &DEM, excludeNoData);
            if (meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->aggregationPointsMaxNr == 0)
            {
                idNotActiveList.append(QString::fromStdString(meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->id));
                meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->active = false;
            }
            else
            {
                if ((float)meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->aggregationPoints.size() / (float)meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->aggregationPointsMaxNr > minCoverage/100.0)
                {
                    idActiveList.append(QString::fromStdString(meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->id));
                    meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->active = true;
                }
                else
                {
                    idNotActiveList.append(QString::fromStdString(meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->id));
                    meteoGridDbHandler->meteoGrid()->meteoPoints()[row][col]->active = false;
                }
            }
        }
    }

    if (modality == MODE_GUI) closeProgressBar();

    logInfoGUI("Update meteo grid db");

    bool ok = true;

    if (!meteoGridDbHandler->setActiveStateCellsInList(&errorString, idActiveList, true))
    {
        ok = false;
    }
    if (!meteoGridDbHandler->setActiveStateCellsInList(&errorString, idNotActiveList, false))
    {
        ok = false;
    }

    return ok;
}

bool PragaProject::planGriddingTask(QDate dateIni, QDate dateFin, QString user, QString notes)
{
    if (meteoGridDbHandler == nullptr)
    {
        logError(ERROR_STR_MISSING_GRID);
        return false;
    }

    QSqlQuery qry(meteoGridDbHandler->db());
    QString table = "gridding_tasks";
    QString statement = QString("CREATE TABLE IF NOT EXISTS `%1` (date_creation DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP, date_start DATE NOT NULL, date_end DATE NOT NULL, praga_user TEXT NOT NULL, notes TEXT, ts_end DATETIME, ts_start DATETIME)").arg(table);

    if( !qry.exec(statement) )
    {
        errorString = qry.lastError().text();
        return false;
    }

    QString myQuery = QString("REPLACE INTO `%1` (`praga_user`,`date_start`,`date_end`,`notes`) VALUES ('%2','%3','%4','%5')").arg(table).arg(user).arg(dateIni.toString("yyyy-MM-dd")).arg(dateFin.toString("yyyy-MM-dd")).arg(notes);

    if( !qry.exec(myQuery))
    {
        errorString = qry.lastError().text();
        myQuery.clear();
        return false;
    }
    else
    {
        myQuery.clear();
        return true;
    }

    return true;

}


bool PragaProject::getGriddingTasks(std::vector <QDateTime> &timeCreation, std::vector <QDate> &dateStart, std::vector <QDate> &dateEnd,
                                           std::vector <QString> &users, std::vector <QString> &notes)
{
    if (meteoGridDbHandler == nullptr)
    {
        logError(ERROR_STR_MISSING_GRID);
        return false;
    }

    QSqlQuery qry(meteoGridDbHandler->db());
    QString table = "gridding_tasks";
    QString myQuery = QString("SELECT * FROM `%1` ORDER BY `date_creation`,`praga_user`,`date_start`,`date_end`").arg(table);

    QDateTime myTime;
    QDate myDateStart, myDateEnd;
    QString user, note;

    if( !qry.exec(myQuery))
    {
        errorString = qry.lastError().text();
        myQuery.clear();
        return false;
    }
    else
    {
        while (qry.next())
        {
            if (getValue(qry.value("praga_user"), &user) && getValue(qry.value("date_creation"), &myTime)
                    && getValue(qry.value("date_start"), &myDateStart) && getValue(qry.value("date_end"), &myDateEnd))
            {
                timeCreation.push_back(myTime);
                dateStart.push_back(myDateStart);
                dateEnd.push_back(myDateEnd);
                users.push_back(user);

                note = "";
                getValue(qry.value("notes"), &note);
                notes.push_back(note);
            }
            else
            {
                errorString = "Error reading table " + table ;
                return false;
            }
        }
    }

    return true;
}

bool PragaProject::removeGriddingTask(QDateTime dateCreation, QString user, QDate dateStart, QDate dateEnd)
{
    if (meteoGridDbHandler == nullptr)
    {
        logError(ERROR_STR_MISSING_GRID);
        return false;
    }

    QSqlQuery qry(meteoGridDbHandler->db());
    QString table = "gridding_tasks";
    QString myQuery = QString("DELETE FROM `%1` WHERE `date_creation`='%2' AND `praga_user`='%3' AND `date_start`='%4' AND `date_end`='%5'")
            .arg(table).arg(dateCreation.toString("yyyy-MM-dd hh:mm:ss")).arg(user).arg(dateStart.toString("yyyy-MM-dd")).arg(dateEnd.toString("yyyy-MM-dd"));

    if( !qry.exec(myQuery))
    {
        errorString = qry.lastError().text();
        myQuery.clear();
        return false;
    }

    return true;
}

bool PragaProject::cleanClimatePoint()
{

    QSqlDatabase db;

    if (this->meteoPointsDbHandler == nullptr)
    {
        errorString = "Load meteo points";
        return false;
    }
    db = this->meteoPointsDbHandler->getDb();

    QList<QString> climateTables;
    QList<QString> climateFields;

    if (getClimateTables(db, &errorString, &climateTables) )
    {
        for (int i = 0; i < climateTables.size(); i++)
        {
            climateFields.clear();
            getClimateFieldsFromTable(db, &errorString, climateTables[i], &climateFields);

            if (! climateFields.isEmpty())
            {
                for (int j = 0; j < climateFields.size(); j++)
                    if (! deleteElab(db, &errorString, climateTables[i].toLower(), climateFields[j]))
                        return false;
            }
        }
    }

    return true;
}

bool PragaProject::computeClimatePointXML(QString xmlName)
{

    Crit3DElabList *listXMLElab = new Crit3DElabList();
    Crit3DAnomalyList *listXMLAnomaly = new Crit3DAnomalyList();
    Crit3DDroughtList *listXMLDrought = new Crit3DDroughtList();
    Crit3DPhenologyList *listXMLPhenology = new Crit3DPhenologyList();

    if (xmlName == "")
    {
        errorString = "Empty XML name";
        delete listXMLElab;
        delete listXMLAnomaly;
        delete listXMLDrought;
        delete listXMLPhenology;
        return false;
    }
    if (!parseXMLElaboration(listXMLElab, listXMLAnomaly, listXMLDrought, listXMLPhenology, xmlName, &errorString))
    {
        delete listXMLElab;
        delete listXMLAnomaly;
        delete listXMLDrought;
        delete listXMLPhenology;
        return false;
    }
    if (!elaborationCheck(listXMLElab->isMeteoGrid(), false))
    {
        errorString = "Elaboration check return false";
        delete listXMLElab;
        delete listXMLAnomaly;
        delete listXMLDrought;
        delete listXMLPhenology;
        return false;
    }
    if (listXMLElab->listAll().isEmpty())
    {
        errorString = "There are not valid Elaborations";
        delete listXMLElab;
        delete listXMLAnomaly;
        delete listXMLDrought;
        delete listXMLPhenology;
        return false;
    }

    clima->getListElab()->setListClimateElab(listXMLElab->listAll());
    int validCell = 0;
    bool changeDataSet = true;
    QDate startDate;
    QDate endDate;
    Crit3DMeteoPoint* meteoPointTemp = new Crit3DMeteoPoint;

    for (int i = 0; i < nrMeteoPoints; i++)
    {
        if (meteoPoints[i].active)
        {
            meteoPointTemp->id = meteoPoints[i].id;
            meteoPointTemp->point.z = meteoPoints[i].point.z;
            meteoPointTemp->latitude = meteoPoints[i].latitude;
            changeDataSet = true;

            std::vector<float> outputValues;
            for (int i = 0; i<listXMLElab->listAll().size(); i++)
            {
                clima->setDailyCumulated(listXMLElab->listDailyCumulated()[i]);
                clima->setClimateElab(listXMLElab->listAll()[i]);
                clima->setVariable(listXMLElab->listVariable()[i]);
                clima->setYearStart(listXMLElab->listYearStart()[i]);
                clima->setYearEnd(listXMLElab->listYearEnd()[i]);
                clima->setPeriodStr(listXMLElab->listPeriodStr()[i]);
                clima->setPeriodType(listXMLElab->listPeriodType()[i]);

                clima->setGenericPeriodDateStart(listXMLElab->listDateStart()[i]);
                clima->setGenericPeriodDateEnd(listXMLElab->listDateEnd()[i]);
                clima->setNYears(listXMLElab->listNYears()[i]);
                clima->setElab1(listXMLElab->listElab1()[i]);

                if (!listXMLElab->listParam1IsClimate()[i])
                {
                    clima->setParam1IsClimate(false);
                    clima->setParam1(listXMLElab->listParam1()[i]);
                }
                else
                {
                    clima->setParam1IsClimate(true);
                    clima->setParam1ClimateField(listXMLElab->listParam1ClimateField()[i]);
                    int climateIndex = getClimateIndexFromElab(listXMLElab->listDateStart()[i], listXMLElab->listParam1ClimateField()[i]);
                    clima->setParam1ClimateIndex(climateIndex);

                }
                clima->setElab2(listXMLElab->listElab2()[i]);
                clima->setParam2(listXMLElab->listParam2()[i]);

                if (clima->periodType() == genericPeriod)
                {
                    startDate.setDate(clima->yearStart(), clima->genericPeriodDateStart().month(), clima->genericPeriodDateStart().day());
                    endDate.setDate(clima->yearEnd() + clima->nYears(), clima->genericPeriodDateEnd().month(), clima->genericPeriodDateEnd().day());
                }
                else if (clima->periodType() == seasonalPeriod)
                {
                    startDate.setDate(clima->yearStart() -1, 12, 1);
                    endDate.setDate(clima->yearEnd(), 12, 31);
                }
                else
                {
                    startDate.setDate(clima->yearStart(), 1, 1);
                    endDate.setDate(clima->yearEnd(), 12, 31);
                }

                if (climateOnPoint(&errorString, meteoPointsDbHandler, nullptr, clima, meteoPointTemp, outputValues, listXMLElab->isMeteoGrid(), startDate, endDate, changeDataSet, meteoSettings))
                {
                    validCell = validCell + 1;
                }
                changeDataSet = false;

                // reset param
                clima->resetParam();
                // reset current values
                clima->resetCurrentValues();
            }
        }
    }

    delete listXMLElab;
    delete listXMLAnomaly;
    delete listXMLDrought;
    delete listXMLPhenology;

    if (validCell == 0)
    {
        if (errorString.isEmpty())
        {
            errorString = "no valid cells available";
        }
        logError(errorString);
        delete meteoPointTemp;
        return false;
     }
     else
     {
        logInfo("climate saved");
         delete meteoPointTemp;
         return true;
     }
}

bool PragaProject::saveLogProceduresGrid(QString nameProc, QDate date)
{

    // check meteo grid
    if (! meteoGridLoaded)
    {
        logError("No meteo grid");
        return false;
    }

    // check dates
    if (date.isNull() || !date.isValid())
    {
        logError("Wrong date");
        return false;
    }

    QString myError;
    logInfoGUI("Saving procedure last date");
    if (! meteoGridDbHandler->saveLogProcedures(&myError, nameProc, date))
    {
        logError(myError);
        return false;
    }
    return true;
}


// --------------------------- OUTPUT METEO POINTS ----------------------------------

void PragaProject::closeOutputMeteoPointsDB()
{
    if (outputMeteoPointsDbHandler != nullptr)
    {
        delete outputMeteoPointsDbHandler;
        outputMeteoPointsDbHandler = nullptr;
    }

    outputMeteoPointsLoaded = false;
    outputPoints.clear();
}


bool PragaProject::loadOutputMeteoPointsDB(const QString &fileName)
{
    if (fileName.isEmpty())
        return false;

    closeOutputMeteoPointsDB();
    errorString = "";
    logInfo("Load output meteo points = " + fileName);

    QString outputDbName = getCompleteFileName(fileName, PATH_METEOPOINT);
    if (! QFile(outputDbName).exists())
    {
        errorString = "Output meteo points DB does not exists:\n" + outputDbName;
        return false;
    }

    outputMeteoPointsDbHandler = new Crit3DMeteoPointsDbHandler(outputDbName);
    if (! outputMeteoPointsDbHandler->getErrorString().isEmpty())
    {
        errorString = "Error in opening:\n" + outputDbName
                      + "\n" + outputMeteoPointsDbHandler->getErrorString();
        return false;
    }

    QList<Crit3DMeteoPoint> listMeteoPoints;
    if (! outputMeteoPointsDbHandler->getPropertiesFromDb(listMeteoPoints, gisSettings, errorString))
    {
        errorString = "Error in reading table 'point_properties'\n" + errorString;
        return false;
    }

    if (listMeteoPoints.empty())
    {
        errorString = "Missing data in the table 'point_properties'";
        return false;
    }
    // set output points
    outputPoints.resize(listMeteoPoints.size());
    for (int i=0; i < listMeteoPoints.size(); i++)
    {
        outputPoints[i].id = listMeteoPoints[i].id;
        outputPoints[i].latitude = listMeteoPoints[i].latitude;
        outputPoints[i].longitude = listMeteoPoints[i].longitude;
        outputPoints[i].utm = listMeteoPoints[i].point.utm;
        outputPoints[i].z = listMeteoPoints[i].point.z;
        outputPoints[i].active = listMeteoPoints[i].active;
    }
    listMeteoPoints.clear();

    outputMeteoPointsLoaded = true;
    logInfo("Output meteo points DB = " + outputDbName);

    return true;
}


bool PragaProject::writeMeteoPointsProperties(const QList<QString> &joinedPropertiesList, const QList<QString> &csvFields,
                                              const QList<QList<QString>> &csvData, bool isOutputPoints)
{
    QList<QString> propertiesList;
    QList<int> posValues;

    for (int i = 0; i < joinedPropertiesList.size(); i++)
    {
        QList<QString> couple = joinedPropertiesList[i].split("-->");
        QString pragaProperty = couple[0];
        QString csvProperty = couple[1];
        int pos = csvFields.indexOf(csvProperty);
        if (pos != -1)
        {
            propertiesList << pragaProperty;
            posValues << pos;
        }
    }

    for (int row = 0; row < csvData.size(); row++)
    {
        QList<QString> csvDataList;

        for (int j = 0; j < posValues.size(); j++)
        {
            csvDataList << csvData[row][posValues[j]];
        }

        if (isOutputPoints)
        {
            if (! outputMeteoPointsDbHandler->updatePointProperties(propertiesList, csvDataList))
            {
                errorString = outputMeteoPointsDbHandler->getErrorString();
                return false;
            }
        }
        else
        {
            if (! meteoPointsDbHandler->updatePointProperties(propertiesList, csvDataList))
            {
                errorString = meteoPointsDbHandler->getErrorString();
                return false;
            }
        }
    }

    return true;
}
