/*!
    \copyright 2018 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by ARPAE Emilia-Romagna

    CRITERIA3D is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CRITERIA3D is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with CRITERIA3D.  If not, see <http://www.gnu.org/licenses/>.

    contacts:
    fausto.tomei@gmail.com
    ftomei@arpae.it
    gantolini@arpae.it
*/


#include "commonConstants.h"
#include "formInfo.h"
#include "utilities.h"
#include "criteria3DProject.h"
#include "soilDbTools.h"
#include "gis.h"
#include "statistics.h"

#include <QtSql>
#include <QPaintEvent>


Crit3DProject::Crit3DProject() : Project3D()
{
}


bool Crit3DProject::loadCriteria3DProject(QString myFileName)
{
    if (myFileName == "") return(false);

    clearCriteria3DProject();

    initializeProject();
    initializeProject3D();

    if (! loadProjectSettings(myFileName))
        return false;

    if (! loadCriteria3DSettings())
        return false;

    if (! loadProject())
        return false;

    // soil map and data
    if (soilMapFileName != "") loadSoilMap(soilMapFileName);
    if (soilDbFileName != "") loadSoilDatabase(soilDbFileName);

    // soiluse map and data
    // TODO soilUse map
    if (cropDbFileName != "") loadCropDatabase(cropDbFileName);

    if (projectName != "")
    {
        logInfo("Project " + projectName + " loaded");
    }

    isProjectLoaded = true;
    return isProjectLoaded;
}


bool Crit3DProject::loadCriteria3DSettings()
{
    projectSettings->beginGroup("project");

    soilDbFileName = projectSettings->value("soil_db").toString();
    if (soilDbFileName == "")
        soilDbFileName = projectSettings->value("db_soil").toString();

    cropDbFileName = projectSettings->value("crop_db").toString();
    if (cropDbFileName == "")
        cropDbFileName = projectSettings->value("db_crop").toString();

    soilMapFileName = projectSettings->value("soil_map").toString();

    projectSettings->endGroup();

    return true;
}


bool Crit3DProject::loadSoilMap(QString fileName)
{
    if (fileName == "")
    {
        logError("Missing soil map filename");
        return false;
    }

    soilMapFileName = fileName;
    fileName = getCompleteFileName(fileName, PATH_GEO);

    std::string myError;
    std::string myFileName = fileName.left(fileName.length()-4).toStdString();

    if (! gis::readEsriGrid(myFileName, &soilMap, &myError))
    {
        logError("Load soil map failed: " + fileName);
        return false;
    }

    logInfo("Soil map = " + fileName);
    return true;
}


bool Crit3DProject::setSoilIndexMap()
{
    // check
    if (!DEM.isLoaded || !soilMap.isLoaded || soilList.size() == 0)
    {
        if (!DEM.isLoaded)
            logError("Missing Digital Elevation Model.");
        else if (!soilMap.isLoaded)
            logError("Missing soil map.");
        else if (soilList.size() == 0)
            logError("Missing soil properties.");
        return false;
    }

    int soilIndex;
    double x, y;
    soilIndexMap.initializeGrid(*(DEM.header));
    for (int row = 0; row < DEM.header->nrRows; row++)
    {
        for (int col = 0; col < DEM.header->nrCols; col++)
        {
            if (int(DEM.value[row][col]) != int(DEM.header->flag))
            {
                gis::getUtmXYFromRowCol(DEM, row, col, &x, &y);
                soilIndex = getCrit3DSoilIndex(x, y);
                if (soilIndex != NODATA)
                    soilIndexMap.value[row][col] = float(soilIndex);
            }
        }
    }

    soilIndexMap.isLoaded = true;
    return true;
}


int Crit3DProject::getCrit3DSoilId(double x, double y)
{
    if (! soilMap.isLoaded)
        return NODATA;

    int idSoil = int(gis::getValueFromXY(soilMap, x, y));

    if (idSoil == int(soilMap.header->flag))
    {
        return NODATA;
    }
    else
    {
        return idSoil;
    }
}


int Crit3DProject::getCrit3DSoilIndex(double x, double y)
{
    int idSoil = getCrit3DSoilId(x, y);
    if (idSoil == NODATA) return NODATA;

    for (unsigned int index = 0; index < soilList.size(); index++)
    {
        if (soilList[index].id == idSoil)
        {
            return signed(index);
        }
    }

    return NODATA;
}


QString Crit3DProject::getCrit3DSoilCode(double x, double y)
{
    int idSoil = getCrit3DSoilId(x, y);
    if (idSoil == NODATA) return "";

    for (unsigned int i = 0; i < soilList.size(); i++)
    {
        if (soilList[i].id == idSoil)
        {
            return QString::fromStdString(soilList[i].code);
        }
    }

    return "NOT FOUND";
}


double Crit3DProject::getSoilVar(int soilIndex, int layerIndex, soil::soilVariable myVar)
{
    unsigned int horizonIndex = unsigned(soil::getHorizonIndex(&(soilList[unsigned(soilIndex)]),
                                                               layerDepth[unsigned(layerIndex)]));
    if (myVar == soil::soilWaterPotentialWP)
        return soilList[unsigned(soilIndex)].horizon[horizonIndex].wiltingPoint;
    else if (myVar == soil::soilWaterPotentialFC)
        return soilList[unsigned(soilIndex)].horizon[horizonIndex].fieldCapacity;
    else if (myVar == soil::soilWaterContentFC)
        return soilList[unsigned(soilIndex)].horizon[horizonIndex].waterContentFC;
    else if (myVar == soil::soilWaterContentSat)
        return soilList[unsigned(soilIndex)].horizon[horizonIndex].vanGenuchten.thetaS;
    else if (myVar == soil::soilWaterContentWP)
    {
        double signPsiLeaf = -160;      //[m]
        return soil::thetaFromSignPsi(signPsiLeaf, &(soilList[unsigned(soilIndex)].horizon[horizonIndex]));
    }
    else
        return NODATA;
}


void Crit3DProject::clearCriteria3DProject()
{
    soilUseMap.clear();
    soilMap.clear();

    clearProject3D();
}


bool Crit3DProject::computeAllMeteoMaps(const QDateTime& myTime, bool showInfo)
{
    if (! this->DEM.isLoaded)
    {
        errorString = "Load a Digital Elevation Model before.";
        return false;
    }
    if (this->hourlyMeteoMaps == nullptr)
    {
        errorString = "Meteo maps not initialized.";
        return false;
    }

    this->hourlyMeteoMaps->setComputed(false);

    FormInfo myInfo;
    if (showInfo)
    {
        myInfo.start("Computing air temperature...", 6);
    }

    if (! interpolateHourlyMeteoVar(airTemperature, myTime, false))
        return false;

    if (showInfo)
    {
        myInfo.setText("Computing air relative humidity...");
        myInfo.setValue(1);
    }

    if (! interpolateHourlyMeteoVar(airRelHumidity, myTime, false))
        return false;

    if (showInfo)
    {
        myInfo.setText("Computing precipitation...");
        myInfo.setValue(2);
    }

    if (! interpolateHourlyMeteoVar(precipitation, myTime, false))
        return false;

    if (showInfo)
    {
        myInfo.setText("Computing wind intensity...");
        myInfo.setValue(3);
    }

    if (! interpolateHourlyMeteoVar(windScalarIntensity, myTime, false))
        return false;

    if (showInfo)
    {
        myInfo.setText("Computing global irradiance...");
        myInfo.setValue(4);
    }

    if (! interpolateHourlyMeteoVar(globalIrradiance, myTime, false))
        return false;

    if (showInfo)
    {
        myInfo.setText("Computing ET0...");
        myInfo.setValue(5);
    }

    if (! this->hourlyMeteoMaps->computeET0PMMap(this->DEM, this->radiationMaps))
        return false;

    if (showInfo) myInfo.close();

    this->hourlyMeteoMaps->setComputed(true);

    return true;
}


bool Crit3DProject::initializeCriteria3DModel()
{
    // check data
    if (! this->DEM.isLoaded)
    {
        logError("Missing Digital Elevation Model.");
        return false;
    }
    else if (! this->soilMap.isLoaded)
    {
        logError("Missing soil map.");
        return false;
    }
    else if (this->soilList.size() == 0)
    {
        logError("Missing soil properties.");
        return false;
    }

    clearWaterBalance3D();

    if (!setSoilIndexMap()) return false;

    // TODO set soilUseMap()

    /* TODO initialize root density
    // andrebbe rifatto per ogni tipo di suolo (ora considera solo suolo 0)
    int nrSoilLayersWithoutRoots = 2;
    int soilLayerWithRoot = this->nrSoilLayers - nrSoilLayersWithoutRoots;
    double depthModeRootDensity = 0.35*this->soilDepth;     //[m] depth of mode of root density
    double depthMeanRootDensity = 0.5*this->soilDepth;      //[m] depth of mean of root density
    initializeRootProperties(&(this->soilList[0]), this->nrSoilLayers, this->soilDepth,
                         this->layerDepth.data(), this->layerThickness.data(),
                         nrSoilLayersWithoutRoots, soilLayerWithRoot,
                         GAMMA_DISTRIBUTION, depthModeRootDensity, depthMeanRootDensity);
    */

    if (! initializeWaterBalance3D())
    {
        clearWaterBalance3D();
        logError("Criteria3D model not initialized.");
        return false;
    }

    isCriteria3DInitialized = true;
    logInfoGUI("Criteria3D model initialized");

    return true;
}


void Crit3DProject::setAllHourlyMeteoMapsComputed(bool value)
{
    if (radiationMaps != nullptr)
        radiationMaps->setComputed(value);

    if (hourlyMeteoMaps != nullptr)
        hourlyMeteoMaps->setComputed(value);
}


bool Crit3DProject::saveDailyOutput(QDate myDate, const QString& hourlyPath)
{
    QString dailyPath = getProjectPath() + "OUTPUT/daily/" + myDate.toString("yyyy/MM/dd/");
    QDir myDir;

    if (! myDir.mkpath(dailyPath))
    {
        logError("Creation daily output directory failed." );
        return false;
    }
    else
    {
        logInfo("Aggregate daily meteo data");
        Crit3DDate crit3DDate = getCrit3DDate(myDate);

        aggregateAndSaveDailyMap(airTemperature, aggrMin, crit3DDate, dailyPath, hourlyPath, "");
        aggregateAndSaveDailyMap(airTemperature, aggrMax, crit3DDate, dailyPath, hourlyPath, "");
        aggregateAndSaveDailyMap(airTemperature, aggrAverage, crit3DDate, dailyPath,hourlyPath, "");
        aggregateAndSaveDailyMap(precipitation, aggrSum, crit3DDate, dailyPath, hourlyPath, "");
        aggregateAndSaveDailyMap(referenceEvapotranspiration, aggrSum, crit3DDate, dailyPath, hourlyPath, "");
        aggregateAndSaveDailyMap(airRelHumidity, aggrMin, crit3DDate, dailyPath, hourlyPath, "");
        aggregateAndSaveDailyMap(airRelHumidity, aggrMax, crit3DDate, dailyPath, hourlyPath, "");
        aggregateAndSaveDailyMap(airRelHumidity, aggrAverage, crit3DDate, dailyPath, hourlyPath, "");
        aggregateAndSaveDailyMap(globalIrradiance, aggrSum, crit3DDate, dailyPath, hourlyPath, "");

        removeDirectory(hourlyPath);

        // save crop output

        // save water balance output
    }

    return true;
}

