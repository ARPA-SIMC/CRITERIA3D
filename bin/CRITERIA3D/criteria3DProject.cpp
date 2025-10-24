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
    ftomei@arpae.it
    gantolini@arpae.it
*/


#include "commonConstants.h"
#include "basicMath.h"
#include "utilities.h"
#include "criteria3DProject.h"
#include "gis.h"
#include "meteo.h"
#include "color.h"
#include "statistics.h"
#include "project.h"
#include "project3D.h"
#include "soilFluxes3D.h"
#include "hydrall.h"
#include "physics.h"
#include "shell.h"

#include <iostream>
#include <QVector3D>
#include <cfloat>
#include <omp.h>


Crit3DProject::Crit3DProject() : Project3D()
{
    _saveOutputRaster = false;
    _saveOutputPoints = false;
    _saveDailyState = false;
    _saveEndOfRunState = false;
    _saveYearlyState = false;
    _saveMonthlyState = false;

    modelFirstTime.setTimeSpec(Qt::UTC);
    modelLastTime.setTimeSpec(Qt::UTC);
}


bool Crit3DProject::initializeCriteria3DModel()
{
    if (! check3DProject())
        return false;

    clearWaterBalance3D();

    if (! waterFluxesParameters.computeOnlySurface)
    {
        // it is necessary to reload the soils db (the fitting options may have changed)
        if (! loadSoilDatabase(soilDbFileName))
            return false;

        if (! setSoilIndexMap())
            return false;
    }

    if (! initialize3DModel())
    {
        clearWaterBalance3D();
        errorString += "\nCriteria3D model is not initialized!";
        return false;
    }

    isCriteria3DInitialized = true;
    return true;
}


void Crit3DProject::clearCropMaps()
{
    laiMap.clear();
    degreeDaysMap.clear();
    dailyTminMap.clear();
    dailyTmaxMap.clear();

    isCropInitialized = false;
}

void Crit3DProject::clearHydrallMaps()
{
    mapLast30DaysTAvg.clear();

    isHydrallInitialized = false;
}

void Crit3DProject::clearRothCMaps()
{
    monthlyET0.clear();
    monthlyPrec.clear();

    rothCModel.map.clear();

    isRothCInitialized = false;
}

bool Crit3DProject::initializeHydrall()
{
    if (! DEM.isLoaded)
    {
        errorString = ERROR_STR_MISSING_DEM;
        return false;
    }

    if (! treeCoverMap.isLoaded)
    {
        errorString = "Load a tree cover map in order to compute Hydrall model.";
        return false;
    }
    logInfo("Initialize hydrall model...");


    //TODO fare anche clear hydrall maps
    mapLast30DaysTAvg.initializeGrid(DEM);

    //inizializzare un vettore che rimandi ai valori dei parametri ecofisiologici per hydrall che attraverso gli indici della croplist

    //approccio alternativo sarebbe avere nelle tabelle tipo tableEcophysiologicalParameters un id che corrisponde all'indice letto da .flt di copertura forestaoe
    //si inizializza conversionTablVector in modo che all'indice letto da .flt corrisponda l'indice da utilizzare invece nella tabella tableEcophysiologicalParameters (e le altre)
    //a quel punto, quando viene letta la currentCrop dentro assignETreal, viene letta come foresta secondo la mappa di landUse, poi viene letto il valore della specifica foresta
    //e infine si aggiorna la crop in base all'indice letto grazie al conversiontablevector. TODO parlarne con Anto

    hydrallModel.conversionTableVector.resize(cropList.size(), NODATA);

    for (unsigned int i = 0; i < cropList.size(); i++)
    {
        for (unsigned int j = 0; j < hydrallModel.plant.tableEcophysiologicalParameters.size(); j++)
        {
            if (cropList[i].idCrop == hydrallModel.plant.tableEcophysiologicalParameters[j].name)
            {
                hydrallModel.conversionTableVector[i] = j;
            }
        }
    }

    for (int row = 0; row < DEM.header->nrRows; row++)
    {
        for (int col = 0; col < DEM.header->nrCols; col++)
        {
            float height = DEM.value[row][col];
            if (! isEqual(height, DEM.header->flag))
            {
                mapLast30DaysTAvg.value[row][col] = 15.f; // initialize to 15°C
            }
        }
    }

    isHydrallInitialized = true;

    return true;
}

bool Crit3DProject::initializeRothC()
{
    rothCModel.initialize();
    rothCModel.map.initialize(DEM);

    monthlyET0.initializeGrid(*(DEM.header));
    monthlyPrec.initializeGrid(*(DEM.header));

    if (! mapLast30DaysTAvg.isLoaded)
        mapLast30DaysTAvg.initializeGrid(*DEM.header);

    if (! processes.computeCrop)
    {
        dailyTminMap.initializeGrid(*(DEM.header));
        dailyTmaxMap.initializeGrid(*(DEM.header));
    }

    if (! soilIndexMap.isLoaded)
        setSoilIndexMap();

    rothCModel.conversionTableVector.resize(cropList.size(), NODATA);

    for (unsigned int i = 0; i < cropList.size(); i++)
    {
        for (unsigned int j = 0; j < rothCModel.tableYield.size(); j++)
        {
            if (cropList[i].idCrop == rothCModel.tableYield[j].name)
            {
                rothCModel.conversionTableVector[i] = j;
            }
        }
    }


    logInfo("Initializing RothC maps...");

    for (int row = 0; row < DEM.header->nrRows; row ++)
    {
        for (int col = 0; col < DEM.header->nrCols; col++)
        {
            if (! isEqual(DEM.value[row][col], DEM.header->flag))
            {
                int soilIndex = int(soilIndexMap.value[row][col]);
                if (soilIndex != NODATA)
                {
                    rothCModel.map.setDepth(soilList[soilIndex].totalDepth, row, col);

                    double clayContent = 0;
                    unsigned int i;

                    if (! processes.computeWater)
                    {
                        rothCModel.map.setClay(getRothCClayContent(soilIndex), row, col);
                    }
                    else
                    {
                        for (i = 0; ((i < nrLayers) && (soilList[soilIndex].getHorizonIndex(layerDepth[i]))!= NODATA); i++)
                        {
                            clayContent += soilList[soilIndex].horizon[soilList[soilIndex].getHorizonIndex(layerDepth[i])].texture.clay;
                        }

                        if (i > 0)
                            rothCModel.map.setClay(clayContent/i, row, col);
                    }

                    if (! processes.computeHydrall)
                    {
                        float height = DEM.value[row][col];
                        if (! isEqual(height, DEM.header->flag))
                        {
                            mapLast30DaysTAvg.value[row][col] = 15.f; // initialize to 15°C
                        }
                    }

                    rothCModel.map.decomposablePlantMaterial->value[row][col] = rothCModel.getDPM();
                    rothCModel.map.resistantPlantMaterial->value[row][col] = rothCModel.getRPM();
                    rothCModel.map.humifiedOrganicMatter->value[row][col] = rothCModel.getHUM();
                    rothCModel.map.microbialBiomass->value[row][col] = rothCModel.getBIO();
                    rothCModel.map.inertOrganicMatter->value[row][col] = rothCModel.getIOM();
                    rothCModel.map.soilOrganicMatter->value[row][col] = rothCModel.getSOC();

                }
            }
        }

    }

    if (! processes.computeWater) //TODO
    {
        rothCModel.isInitializing = true;

        std::string errorStr;
        gis::Crit3DRasterGrid raster;

        QDir myDir = QDir(QString::fromStdString(rothCModel.BICMapFolderName));
        myDir.setNameFilters(QStringList("*.flt"));
        QList<QString> fileList = myDir.entryList();
        std::string fileNamePath;

        if (fileList.size() != 12)
        {
            errorStr = "Insufficient number of files.";
            logError("Average BIC maps load from directory " + QString::fromStdString(rothCModel.BICMapFolderName) + " failed.\n" + QString::fromStdString(errorStr));
            return false;
        }

        for (unsigned int i = 0; i < 12; i++)
        {
            fileNamePath = rothCModel.BICMapFolderName + "/" + fileList[i].toStdString();
            if (! gis::openRaster(fileNamePath, &raster, gisSettings.utmZone, errorStr))
            {
                logError("Average BIC map load failed: " + fileList[i] + "\n" + QString::fromStdString(errorStr));
                return false;
            }

            gis::resampleGrid(raster, rothCModel.map.avgBIC[i], DEM.header, aggrPrevailing, 0);
        }
        logInfo("Average BIC maps loaded from directory " + QString::fromStdString(rothCModel.BICMapFolderName));

    }

    isRothCInitialized = true;
    //todo

    return true;
}

//initializing soil carbon content without interpolating meteo data. using data of temperature and BIC averaged over the last 24 years
bool Crit3DProject::initializeRothCSoilCarbonContent()
{
    for (int row = 0; row < DEM.header->nrRows; row ++)
    {
        for (int col = 0; col < DEM.header->nrCols; col++)
        {
            if (! isEqual(DEM.value[row][col], DEM.header->flag))
            {
                int soilIndex = int(soilIndexMap.value[row][col]);
                if (soilIndex != NODATA)
                {
                    rothCModel.setClay(rothCModel.map.getClay(row, col));

                    rothCModel.setStateVariables(row, col);
                    rothCModel.initializeRothCSoilCarbonContent();
                    rothCModel.getStateVariables(row, col);
                }
            }
        }
    }

    return true;
}

double Crit3DProject::getRothCClayContent(int soilIndex)
{
    std::vector<soil::Crit3DHorizon> horizonVector = soilList[soilIndex].horizon;
    double weightSum = 0;
    double clayContent = 0;
    double upperDepth = 0;
    double lowerDepth = 0;

    if (! horizonVector.empty())
        lowerDepth = horizonVector.front().lowerDepth;
    else
        return NODATA;


    for (unsigned int i = 0; i < horizonVector.size(); i++)
    {
        clayContent += horizonVector[i].texture.clay * (lowerDepth - upperDepth);
        weightSum += lowerDepth - upperDepth;
    }

    return clayContent/weightSum;
}

bool Crit3DProject::initializeCropMaps()
{
    if (! DEM.isLoaded)
    {
        errorString = ERROR_STR_MISSING_DEM;
        return false;
    }

    logInfo("Initialize crop...");

    // initialize LAI and degree days map to NODATA
    laiMap.initializeGrid(*(DEM.header));
    degreeDaysMap.initializeGrid(*(DEM.header));

    dailyTminMap.initializeGrid(*(DEM.header));
    dailyTmaxMap.initializeGrid(*(DEM.header));

    return true;
}


bool Crit3DProject::initializeCropWithClimateData()
{
    if (! processes.computeCrop)
        return false;

    if (! initializeCropMaps())
        return false;

    if (landUnitList.empty() || cropList.empty())
    {
        errorString = "missing crop db or land use map";
        return false;
    }

    QDate currentDate = getCurrentDate();

    for (int row = 0; row < DEM.header->nrRows; row++)
    {
        for (int col = 0; col < DEM.header->nrCols; col++)
        {
            float height = DEM.value[row][col];
            if (! isEqual(height, DEM.header->flag))
            {
                int index = getLandUnitIndexRowCol(row, col);
                if (isCrop(index))
                {
                    double degreeDays = 0;
                    int firstDoy = 1;
                    int lastDoy = currentDate.dayOfYear();

                    if (gisSettings.startLocation.latitude >= 0)
                    {
                        // Northern hemisphere
                        firstDoy = 1;
                    }
                    else
                    {
                        // Southern hemisphere
                        if (currentDate.dayOfYear() >= 182)
                        {
                            firstDoy = 182;
                        }
                        else
                        {
                            firstDoy = -183;
                        }
                    }

                    // daily cycle
                    for (int doy = firstDoy; doy <= lastDoy; doy++)
                    {
                        int currentDoy = doy;
                        int currentYear = currentDate.year();
                        if (currentDoy <= 0)
                        {
                            currentYear--;
                            currentDoy += 365;
                        }
                        Crit3DDate myDate = getDateFromDoy(currentYear, currentDoy);

                        float tmin = climateParameters.getClimateVar(dailyAirTemperatureMin, myDate.month,
                                                                     height, quality->getReferenceHeight());
                        float tmax = climateParameters.getClimateVar(dailyAirTemperatureMax, myDate.month,
                                                                     height, quality->getReferenceHeight());

                        double currentDD = cropList[index].getDailyDegreeIncrease(tmin, tmax, currentDoy);
                        if (! isEqual(currentDD, NODATA))
                        {
                            degreeDays += currentDD;
                        }
                    }

                    degreeDaysMap.value[row][col] = float(degreeDays);
                    laiMap.value[row][col] = cropList[index].computeSimpleLAI(degreeDays, gisSettings.startLocation.latitude,
                                                                              currentDate.dayOfYear());
                }
            }
        }
    }

    logInfo("LAI initialized with climate data - doy: " + QString::number(currentDate.dayOfYear()));
    isCropInitialized = true;

    return true;
}


bool Crit3DProject::initializeCropFromDegreeDays(gis::Crit3DRasterGrid &myDegreeMap)
{
    initializeCropMaps();

    if (! myDegreeMap.isLoaded)
    {
        errorString = "Wrong degree days map: crop cannot be initialized.";
        processes.setComputeCrop(false);
        return false;
    }

     if (! landUseMap.isLoaded || landUnitList.empty())
    {
        errorString = "Crop db or land use map is missing: crop cannot be initialized.";
        processes.setComputeCrop(false);
        return false;
    }

    int currentDoy = getCurrentDate().dayOfYear();
    for (int row = 0; row < DEM.header->nrRows; row++)
    {
        for (int col = 0; col < DEM.header->nrCols; col++)
        {
            float height = DEM.value[row][col];
            if (! isEqual(height, DEM.header->flag))
            {
                // field unit list and crop list have the same index
                int index = getLandUnitIndexRowCol(row, col);
                if (isCrop(index))
                {
                    double x, y;
                    gis::getUtmXYFromRowCol(*(DEM.header), row, col, &x, &y);
                    float currentDegreeDay = gis::getValueFromXY(myDegreeMap, x, y);
                    if (! isEqual(currentDegreeDay, myDegreeMap.header->flag))
                    {
                        degreeDaysMap.value[row][col] = currentDegreeDay;
                        laiMap.value[row][col] = cropList[index].computeSimpleLAI(degreeDaysMap.value[row][col],
                                                         gisSettings.startLocation.latitude, currentDoy);
                    }
                }
            }
        }
    }

    logInfo("LAI initialized with degree days map.");
    isCropInitialized = true;
    return true;
}


void Crit3DProject::dailyUpdateCropMaps(const QDate &myDate)
{
    int firstDoy = 1;
    if (gisSettings.startLocation.latitude < 0)
    {
        // Southern hemisphere
        firstDoy = 182;
    }

    // reset the crop at the beginning of the new year
    if (myDate.dayOfYear() == firstDoy)
    {
        logInfo("Reset crop...");

        laiMap.emptyGrid();
        degreeDaysMap.emptyGrid();
    }

    int currentDoy = getCurrentDate().dayOfYear();
    for (int row = 0; row < DEM.header->nrRows; row++)
    {
        for (int col = 0; col < DEM.header->nrCols; col++)
        {
            // is valid point
            float height = DEM.value[row][col];
            if (! isEqual(height, DEM.header->flag))
            {
                // landUnit list and crop list have the same index
                int index = getLandUnitIndexRowCol(row, col);
                if (isCrop(index))
                {
                    float tmin = dailyTminMap.value[row][col];
                    float tmax = dailyTmaxMap.value[row][col];
                    if (! isEqual(tmin, dailyTminMap.header->flag) && ! isEqual(tmax, dailyTmaxMap.header->flag))
                    {
                        double dailyDD = cropList[index].getDailyDegreeIncrease(tmin, tmax, currentDoy);
                        if (! isEqual(dailyDD, NODATA))
                        {
                            if (isEqual(degreeDaysMap.value[row][col], degreeDaysMap.header->flag))
                            {
                                degreeDaysMap.value[row][col] = float(dailyDD);
                            }
                            else
                            {
                                degreeDaysMap.value[row][col] += float(dailyDD);
                            }

                            laiMap.value[row][col] = cropList[index].computeSimpleLAI(degreeDaysMap.value[row][col],
                                                            gisSettings.startLocation.latitude, currentDoy);
                        }
                    }
                }
            }
        }
    }

    // cleans daily temperature maps
    dailyTminMap.emptyGrid();
    dailyTmaxMap.emptyGrid();
}


bool Crit3DProject::dailyUpdateHydrall(const QDate &myDate)
{
    //set daily variables like temp, co2
    if (myDate.dayOfYear() == 1)
    {
        for (int row = 0; row < indexMap.at(0).header->nrRows; row++)
        {
            for (int col = 0; col < indexMap.at(0).header->nrCols; col++)
            {
                int surfaceIndex = indexMap.at(0).value[row][col];
                if (surfaceIndex != indexMap.at(0).header->flag)
                {
                    //se firstDayOfYear, scrivi le mappe (mensili?) di LAI, biomassa, etc
                    //TODO, togliere?
                    std::string myError;
                    if (! gis::writeEsriGrid(getCompleteFileName("treeNPP_"+myDate.toString("yyyyMMdd"), PATH_OUTPUT).toStdString(), hydrallMaps.treeNetPrimaryProduction, myError))
                    {
                        errorString = QString::fromStdString(myError);
                        return false;
                    }

                    if (! gis::writeEsriGrid(getCompleteFileName("understoreyNPP_"+myDate.toString("yyyyMMdd"), PATH_OUTPUT).toStdString(), hydrallMaps.understoreyNetPrimaryProduction, myError))
                    {
                        errorString = QString::fromStdString(myError);
                        return false;
                    }


                    hydrallModel.setYear(myDate.year());
                    hydrallModel.weatherVariable.setYearlyET0(hydrallMaps.yearlyET0->getValueFromRowCol(row, col));
                    hydrallModel.weatherVariable.setYearlyPrec(hydrallMaps.yearlyPrec->getValueFromRowCol(row, col));

                    hydrallModel.simplifiedGrowthStand(); // TODO quit this line - temporary position to prompt check

                    hydrallModel.resetStandVariables();

                    hydrallMaps.yearlyPrec->value[row][col] = 0;
                    hydrallMaps.yearlyET0->value[row][col] = 0;

                    if (myDate.month() == hydrallModel.firstMonthVegetativeSeason) //TODO
                    {
                        /* in case of the first day of the year
                         * the algorithms devoted to allocate dry matter
                         * into the biomass pools (foliage, sapwood and fine roots)
                         * */
                        //hydrallModel.growthStand();
                        //hydrallModel.resetStandVariables();
                        //grtree

                    }
                }
            }
        }
    }

    return true;
}


bool Crit3DProject::updateRothC(const QDate &myDate)
{
    rothCModel.isInitializing = false;

    if (! processes.computeWater)
        rothCModel.isInitializing = true;

    if (myDate.day() == 1)
    {
        for (int row = 0; row < DEM.header->nrRows; row++)
        {
            for (int col = 0; col < DEM.header->nrCols; col++)
            {
                // is valid point
                float height = DEM.value[row][col];
                if (! isEqual(height, DEM.header->flag))
                {
                    setRothCVariables(row, col, myDate.month());

                    rothCModel.setStateVariables(row, col);

                    if (! rothCModel.checkCell())
                    {
                        //chiamata a rothC
                        computeRothCModel();

                        rothCModel.getStateVariables(row, col);

                        //reset meteo maps
                        monthlyPrec.value[row][col] = 0;
                        monthlyET0.value[row][col] = 0;

                        //reset meteo variables and C input
                        rothCModel.resetInputVariables();
                    }
                }
            }
        }
    }

    return true;
}

void Crit3DProject::setRothCVariables(int row, int col, int month)
{
    //soil variables
    rothCModel.setDepth(rothCModel.map.getDepth(row, col));
    rothCModel.setClay(rothCModel.map.getClay(row, col));
    rothCModel.setPlantCover(1); //understorey

    //meteo variables
    rothCModel.meteoVariable.setPrecipitation(monthlyPrec.getValueFromRowCol(row, col));
    rothCModel.meteoVariable.setWaterLoss(monthlyET0.getValueFromRowCol(row, col));
    rothCModel.meteoVariable.setBIC(monthlyPrec.getValueFromRowCol(row, col) - monthlyET0.getValueFromRowCol(row, col));
    rothCModel.meteoVariable.setTemperature(mapLast30DaysTAvg.value[row][col]);

    rothCModel.meteoVariable.setAvgBIC(rothCModel.map.getAvgBIC(row, col, month));

    double inputCTable = NODATA;

    if (! processes.computeHydrall || ! isEqual(hydrallModel.getOutputC(),NODATA)) //yield table must be used when initialising rothC or for first year of simulation of hydrall+rothC (hydrall still hasn't produced its first carbon output)
    {
        int treeCoverIndex = getTreeCoverIndexRowCol(row,col);
        int managementIndex, forestIndex;

        if (treeCoverIndex != NODATA && treeCoverIndex > 9)
        {
            std::string indexString = std::to_string(treeCoverIndex);
            if (indexString.size() >= 2)
            {
                managementIndex = std::stoi(indexString.substr(indexString.size()-1, indexString.size()-1));
                forestIndex = (treeCoverIndex - managementIndex) / 10 - 1;
                inputCTable = rothCModel.tableYield[rothCModel.conversionTableVector[forestIndex]].carbon;
            }
        }
    }
    //carbon input is taken from hydrall, otherwise TODO
    if (processes.computeHydrall && ! isEqual(hydrallModel.getOutputC(),NODATA))
        rothCModel.setInputC(hydrallModel.getOutputC()/12.0); //read from hydrall (eventually from crop too?). output used is from previous year and divided by 12 months
    else
    {
        rothCModel.setInputC(inputCTable/12.0);
    }

    //swc comes from water model. during initialization phase, it is not used
    double SWC = NODATA;
    if (! rothCModel.isInitializing && processes.computeWater && processes.computeHydrall)
    {
        SWC = 0;
        for (int i = 0; i < (int)hydrallModel.soil.waterContent.size(); i++)
        {
            SWC += hydrallModel.soil.waterContent[i]*hydrallModel.soil.nodeThickness[i]*1000;
        }
    }
    rothCModel.setSWC(SWC);


    return;
}

/*!
 * \brief assignETreal
 * assigns soil evaporation and crop transpiration for the whole domain
 */
void Crit3DProject::assignETreal()
{
    totalEvaporation = 0;               // [m3 h-1]
    totalTranspiration = 0;             // [m3 h-1]

    double area = DEM.header->cellSize * DEM.header->cellSize;

    Crit3DHydrall myHydrallModel = hydrallModel;

    #pragma omp parallel for if (isParallelComputing()) firstprivate(myHydrallModel)
    for (int row = 0; row < indexMap.at(0).header->nrRows; row++)
    {
        for (int col = 0; col < indexMap.at(0).header->nrCols; col++)
        {
            int surfaceIndex = indexMap.at(0).value[row][col];
            if (surfaceIndex != indexMap.at(0).header->flag)
            {
                double utmX, utmY;
                DEM.getXY(row, col, utmX, utmY);
                int soilIndex = getSoilListIndex(utmX, utmY);

                float currentLAI = 0;          // Leaf Area Index [m3 m-3]
                if (laiMap.isLoaded)
                {
                    float laiMapValue = laiMap.getValueFromRowCol(row, col);
                    if (! isEqual(laiMapValue, laiMap.header->flag))
                        currentLAI = laiMapValue;
                }

                // assigns actual evaporation
                double actualEvap = assignEvaporation(row, col, currentLAI, soilIndex);     // [mm h-1]
                double evapFlow = area * (actualEvap / 1000.);                              // [m3 h-1]
                totalEvaporation += evapFlow;                                               // [m3 h-1]

                int forestIndex = NODATA;
                int managementIndex = NODATA;
                //hydrall only
                if (processes.computeHydrall)
                {
                    int treeCoverIndex = getTreeCoverIndexRowCol(row,col);

                    if (treeCoverIndex != NODATA && treeCoverIndex > 9)
                    {
                        std::string indexString = std::to_string(treeCoverIndex);
                        if (indexString.size() >= 2)
                        {
                            managementIndex = std::stoi(indexString.substr(indexString.size()-1, indexString.size()-1));
                            forestIndex = (treeCoverIndex - managementIndex) / 10 - 1;
                        }
                    }
                    myHydrallModel.plant.management = managementIndex;
                }

                int cropIndex = getLandUnitIndexRowCol(row, col);
                if ((cropIndex != NODATA && (int)cropList.size() > cropIndex) || (forestIndex != NODATA && forestIndex >= 0))
                {
                    Crit3DCrop currentCrop;
                    if (forestIndex != NODATA  && forestIndex >= 0 && (int)cropList.size() > forestIndex)
                        currentCrop = cropList[forestIndex];
                    else if (cropIndex != NODATA && (int)cropList.size() > cropIndex)
                        currentCrop = cropList[cropIndex];
                    else
                        continue;

                    double actualTransp = 0;

                    // assigns actual transpiration
                    if (currentLAI > 0)
                    {
                        float degreeDays = degreeDaysMap.value[row][col];
                        actualTransp = assignTranspiration(row, col, currentCrop, currentLAI, degreeDays);          // [mm h-1]
                        // TODO verificare che la traspirazione ottenuta da hydrall sia confrontabile e nel caso mettere un if che decida come computare la traspirazione
                        double traspFlow = area * (actualTransp / 1000.);                                           // [m3 h-1] flux
                        totalTranspiration += traspFlow;                                                            // [m3 h-1] flux
                    }

                    if (processes.computeHydrall && forestIndex != NODATA)
                    {
                        if (currentCrop.roots.rootDensity.empty())
                        {
                            // compute root lenght
                            currentCrop.computeRootLength3D(degreeDaysMap.value[row][col], soilList[soilIndex].totalDepth);

                            // compute root density
                            root::computeRootDensity3D(currentCrop, soilList[soilIndex], nrLayers, layerDepth, layerThickness);
                        }

                        myHydrallModel.soil.setRootDensity(currentCrop.roots.rootDensity);
                        myHydrallModel.plant.setLAICanopy(MAXVALUE(0, currentLAI));
                        myHydrallModel.plant.setLAICanopyMin(currentCrop.LAImin);
                        myHydrallModel.plant.setLAICanopyMax(currentCrop.LAImax);

                        int soilIndex = int(soilIndexMap.value[row][col]);
                        if (soilIndex != NODATA)
                            computeHydrallModel(myHydrallModel, row, col, forestIndex);
                    }

                    if (processes.computeRothC)
                    {
                        // TODO FT non è corretto (trasformare in mappa)
                        rothCModel.setPlantCover(currentLAI / currentCrop.LAImax);
                    }
                }
            }
        }
    }
}


void Crit3DProject::assignPrecipitation()
{
    // initialize
    totalPrecipitation = 0;                 // [m3]

    double area = DEM.header->cellSize * DEM.header->cellSize;

    gis::Crit3DRasterGrid *snowFallMap, *snowMeltMap;
    bool isSnowOk = false;
    if (processes.computeSnow)
    {
        snowFallMap = snowMaps.getSnowFallMap();
        snowMeltMap = snowMaps.getSnowMeltMap();
        if (snowFallMap != nullptr && snowMeltMap != nullptr)
            isSnowOk = true;
    }

    // precipitation
    for (long row = 0; row < indexMap.at(0).header->nrRows; row++)
    {
        for (long col = 0; col < indexMap.at(0).header->nrCols; col++)
        {
            int surfaceIndex = indexMap.at(0).value[row][col];
            if (surfaceIndex != indexMap.at(0).header->flag)
            {
                float prec = hourlyMeteoMaps->mapHourlyPrec->value[row][col];
                if (! isEqual(prec, hourlyMeteoMaps->mapHourlyPrec->header->flag))
                {
                    float liquidWater = prec;
                    if (processes.computeSnow && isSnowOk)
                    {
                        float currentSnowFall = snowFallMap->value[row][col];
                        float currentSnowMelt = snowMeltMap->value[row][col];
                        if (! isEqual(currentSnowFall, snowFallMap->header->flag)
                            && ! isEqual(currentSnowMelt, snowMeltMap->header->flag) )
                        {
                            liquidWater = prec - currentSnowFall + currentSnowMelt;
                        }
                    }
                    if (liquidWater > 0)
                    {
                        double flow = area * (liquidWater / 1000.);                         // [m3 h-1]
                        totalPrecipitation += flow;                                         // [m3 h-1]

                        float precSurfaceWater = computeSoilCracking(row, col, liquidWater);
                        double surfaceFlow = area * (precSurfaceWater / 1000.);             // [m3 h-1]
                        if ((surfaceFlow / 3600.) > 0.)
                        {
                            waterSinkSource[surfaceIndex] += surfaceFlow / 3600.;           // [m3 s-1]
                        }
                    }
                }
            }
        }
    }
}


// Water infiltration into soil cracks
// input: precipitation [mm]
// output: water remaining on the surface [mm]
// after infiltration into soil cracks
// use waterSinkSource to set the infiltration
float Crit3DProject::computeSoilCracking(int row, int col, float precipitation)
{
    const double MAX_CRACKING_DEPTH = 0.6;              // [m]
    const double MIN_VOID_VOLUME = 0.15;                // [m3 m-3]
    const double MAX_VOID_VOLUME = 0.20;                // [m3 m-3]
    const double MIN_FINE_FRACTION = 0.5;               // [m3 m-3]
    const double MIN_FINE_LAYER_DEPTH = 0.2;            // [m]

    // check soil
    int soilIndex = getSoilIndex(row, col);
    if (soilIndex == NODATA)
        return precipitation;

    // check surface
    long surfaceNodeIndex = indexMap.at(0).value[row][col];
    if (surfaceNodeIndex == NODATA)
        return precipitation;

    // check pond
    double currentPond = getCriteria3DVar(surfacePond, surfaceNodeIndex);       // [mm]
    double minimumPond = currentPond;                                           // [mm]
    if (precipitation <= minimumPond)
        return precipitation;

    // check soil depth
    double soilDepth = std::min(computationSoilDepth, soilList[soilIndex].totalDepth);
    if (soilDepth <= MIN_FINE_LAYER_DEPTH)
        return precipitation;

    // check fine fraction (clay)
    bool isFineFraction = true;
    int lastFineHorizon = NODATA;
    unsigned int h = 0;
    double maxDepth = std::min(soilDepth, MAX_CRACKING_DEPTH);   // [m]
    while (h < soilList[soilIndex].nrHorizons && soilList[soilIndex].horizon[h].upperDepth < maxDepth && isFineFraction)
    {
        soil::Crit3DHorizon horizon = soilList[soilIndex].horizon[h];

        double fineFraction = (horizon.texture.clay + horizon.texture.silt * 0.5) / 100
                              * (1 - horizon.coarseFragments) * (1 - horizon.organicMatter);
        if (fineFraction < MIN_FINE_FRACTION)
        {
            isFineFraction = false;
        }
        else
        {
            lastFineHorizon = h;
            h++;
        }
    }

    if (lastFineHorizon == NODATA)
        return precipitation;

    maxDepth = std::min(soilList[soilIndex].horizon[lastFineHorizon].lowerDepth, MAX_CRACKING_DEPTH);
    maxDepth = std::min(maxDepth, computationSoilDepth);

    // clay horizon is too thin
    if (maxDepth < MIN_FINE_LAYER_DEPTH)
        return precipitation;

    // compute the volume of voids
    double stepDepth = 0.05;                // [m]
    double currentDepth = stepDepth;        // [m]
    double voidsVolumeSum = 0;              // [m3 m-3]
    int nrData = 0;
    while (currentDepth <= maxDepth)
    {
        int layerIndex = getSoilLayerIndex(currentDepth);
        if (layerIndex == NODATA)
            break;

        long nodeIndex = indexMap.at(layerIndex).value[row][col];
        if (nodeIndex == NODATA)
            break;

        int horizonIndex = soilList[soilIndex].getHorizonIndex(currentDepth);
        if (horizonIndex == NODATA)
            break;

        double soilFraction = 1.0 - soilList[soilIndex].horizon[horizonIndex].coarseFragments;
        double VWC = getCriteria3DVar(volumetricWaterContent, nodeIndex);               // [m3 m-3]
        double maxVWC = getCriteria3DVar(maximumVolumetricWaterContent, nodeIndex);     // [m3 m-3]

        voidsVolumeSum += (maxVWC - VWC) * soilFraction;
        nrData++;

        currentDepth += stepDepth;
    }

    if (nrData == 0)
        return precipitation;

    double avgVoidVolume = voidsVolumeSum / nrData;              // [m3 m-3]
    if (avgVoidVolume <= MIN_VOID_VOLUME)
        return precipitation;

    // THERE IS A SOIL CRACK
    double crackRatio = std::min(1.0, (avgVoidVolume - MIN_VOID_VOLUME) / (MAX_VOID_VOLUME - MIN_VOID_VOLUME));

    double maxInfiltration = precipitation * crackRatio;        // [mm]
    double surfaceWater = precipitation - maxInfiltration;      // [mm]
    surfaceWater = std::max(surfaceWater, minimumPond);
    double downWater = precipitation - surfaceWater;            // [mm]

    int lastLayer = getSoilLayerIndex(maxDepth);
    double area = DEM.header->cellSize * DEM.header->cellSize;  // [m2]

    // accumulation on the crack bottom (0.5 mm of water for each soil cm)
    for (int l = lastLayer; l > 0; l--)
    {
        if (downWater <= 0)
            break;

        long nodeIndex = indexMap.at(l).value[row][col];
        if (nodeIndex != indexMap.at(l).header->flag)
        {
            double layerThick_cm = layerThickness[l] * 100;         // [cm]
            double layerWater = layerThick_cm * 0.5;                // [mm]
            layerWater = std::min(layerWater, downWater);

            double flow = area * (layerWater / 1000.);              // [m3 h-1]
            if ((flow / 3600.) > 0.)
            {
                waterSinkSource[nodeIndex] += flow / 3600.;         // [m3 s-1]
                downWater -= layerWater;                            // [mm]
            }
        }
    }

    if (downWater > 0)
    {
        // remaining water
        return surfaceWater + downWater;
    }
    else
    {
        return surfaceWater;
    }
}


bool Crit3DProject::startModels(const QDateTime &firstTime, const QDateTime &lastTime)
{
    if (! checkProcesses())
        return false;

    if (processes.computeSnow && ! snowMaps.isInitialized)
    {
        errorString = "Initialize Snow model or load a state before.";
        return false;
    }

    if (processes.computeWater && ! isCriteria3DInitialized)
    {
        errorString = "Initialize 3D water fluxes or load a state before.";
        return false;
    }

    if (processes.computeCrop && landUnitList.size() == 0)
    {
        errorString = "load land units map before.";
        return false;
    }

    std::cout << "First time: " << firstTime.date().toString("yyyy-MM-dd").toStdString() << " H" << firstTime.time().hour() << std::endl;
    std::cout << "Last time: " << lastTime.date().toString("yyyy-MM-dd").toStdString() << " H" << lastTime.time().hour() << std::endl;

    if (lastTime < firstTime)
    {
        errorString = "Wrong Time: lastTime < firstTime";
        return false;
    }

    // todo: struttura dati per memorizzare i tempi

    logInfoGUI("Loading meteo data...");
    bool loadHourly = true;
    bool isdataLoaded = loadMeteoPointsData(firstTime.date().addDays(-1), lastTime.date().addDays(+1), loadHourly, false, false);
    closeLogInfo();
    if (! isdataLoaded)
    {
        logError("No data available for this period.");
        return false;
    }

    // initialize
    modelFirstTime = firstTime;
    modelLastTime = lastTime;
    isModelPaused = false;
    isModelStopped = false;

    return runModels(firstTime, lastTime);
}


bool Crit3DProject::runModels(const QDateTime &firstTime, const QDateTime &lastTime, bool isRestart)
{
    if (! isRestart)
    {
        // create tables for output points
        if (isSaveOutputPoints())
        {
            if (! writeOutputPointsTables())
            {
                logError();
                return false;
            }
        }

        // initialize meteo maps
        if (processes.computeMeteo)
        {
            hourlyMeteoMaps->initialize();

            // load td maps if needed
            if (interpolationSettings.getUseTD())
            {
                logInfoGUI("Loading topographic distance maps...");
                if (! loadTopographicDistanceMaps(true, false))
                    return false;
            }
        }

        // initialize radiation maps
        if (processes.computeRadiation)
        {
            radiationMaps->initialize();
        }

        isModelRunning = true;
    }

    QDate firstDate = firstTime.date();
    QDate lastDate = lastTime.date();
    int hour1 = firstTime.time().hour();
    int hour2 = lastTime.time().hour();

    // cycle on days
    QString currentOutputPath;
    for (QDate myDate = firstDate; myDate <= lastDate; myDate = myDate.addDays(1))
    {
        setCurrentDate(myDate);

        // update crop at last hour of each day
        // TODO FT check
        if (processes.computeCrop && getCurrentHour() == 23)
        {
            if (! isRestart || (currentSeconds == 0 || currentSeconds == 3600))
            {
                dailyUpdateCropMaps(myDate);
            }
        }

        if (processes.computeWater)
        {
            dailyUpdatePond();
        }

        if (processes.computeRothC)
        {
            if (myDate.day() == 1)
            {
                updateRothC(myDate);
            }
        }

        if (processes.computeHydrall)
            dailyUpdateHydrall(myDate);

        if (isSaveOutputRaster())
        {
            // create directory for hourly raster output
            currentOutputPath = getProjectPath() + PATH_OUTPUT + myDate.toString("yyyy/MM/dd/");
            if (!QDir().mkpath(currentOutputPath))
            {
                logError("Creation of directory for hourly raster output failed:" + currentOutputPath);
                setSaveOutputRaster(false);
            }
        }

        // cycle on hours
        int firstHour = (myDate == firstDate) ? hour1 : 0;
        int lastHour = (myDate == lastDate) ? hour2 : 23;

        for (int hour = firstHour; hour <= lastHour; hour++)
        {
            setCurrentHour(hour);
            if (currentSeconds == 0 || currentSeconds == 3600)
                isRestart = false;

            if (!runModelHour(currentOutputPath, isRestart))
            {
                isModelRunning = false;
                logError();
                return false;
            }

            //rothC maps update must be done hourly, otherwise ETReal data are not stored
            if (processes.computeRothC || processes.computeHydrall)
                updateETAndPrecMaps();

            // output points
            if (isSaveOutputPoints() && currentSeconds == 3600)
            {
                if (!writeOutputPointsData())
                {
                    isModelRunning = false;
                    logError();
                    return false;
                }
            }

            if (isModelPaused || isModelStopped)
            {
                return true;
            }
        }

        if (processes.computeHydrall || processes.computeRothC)
        {
            updateLast30DaysTavg();

        }

        if (isSaveDailyState() || (isSaveYearlyState() && myDate.dayOfYear() == 1) || (isSaveMonthlyState() && myDate.day() == 1))
        {
            QString dirName;
            saveModelsState(dirName);
        }
    }

    if (getCurrentTime() == modelLastTime)
    {
        if (isSaveEndOfRunState())
        {
            QString dirName;
            saveModelsState(dirName);
        }

        logInfoGUI("Computation is finished.");
    }

    isModelRunning = false;

    return true;
}

void Crit3DProject::updateETAndPrecMaps()
{
    int nrRows = DEM.header->nrRows;
    int nrCols = DEM.header->nrCols;

    for (int row = 0; row < nrRows; row++) //valuta se usare surfaceIndex o cosa
    {
        for (int col = 0; col < nrCols; col++)
        {
            if (processes.computeHydrall)
            {
                if (! isEqual(hydrallMaps.yearlyET0->value[row][col], NODATA))
                    hydrallMaps.yearlyET0->value[row][col] += hourlyMeteoMaps->mapHourlyET0->value[row][col];
                else
                    hydrallMaps.yearlyET0->value[row][col] = hourlyMeteoMaps->mapHourlyET0->value[row][col];

                if (! isEqual(hydrallMaps.yearlyPrec->value[row][col], NODATA))
                    hydrallMaps.yearlyPrec->value[row][col] += hourlyMeteoMaps->mapHourlyPrec->value[row][col];
                else
                    hydrallMaps.yearlyPrec->value[row][col] = hourlyMeteoMaps->mapHourlyPrec->value[row][col];
            }

            if (processes.computeRothC)
            {
                if (! isEqual(monthlyET0.value[row][col], NODATA))
                    monthlyET0.value[row][col] += hourlyMeteoMaps->mapHourlyET0->value[row][col];
                else
                    monthlyET0.value[row][col] = hourlyMeteoMaps->mapHourlyET0->value[row][col];

                if (! isEqual(monthlyPrec.value[row][col], NODATA))
                    monthlyPrec.value[row][col] += hourlyMeteoMaps->mapHourlyPrec->value[row][col];
                else
                    monthlyPrec.value[row][col] = hourlyMeteoMaps->mapHourlyPrec->value[row][col];
            }

        }
    }
}

void Crit3DProject::setSaveOutputRaster(bool isSave)
{
    _saveOutputRaster = isSave;
}

bool Crit3DProject::isSaveOutputRaster()
{
    return _saveOutputRaster;
}

void Crit3DProject::setSaveOutputPoints(bool isSave)
{
    _saveOutputPoints = isSave;
}

// true if at least one point is active
bool Crit3DProject::isSaveOutputPoints()
{
    if (! _saveOutputPoints || outputPoints.empty())
        return false;

    for (unsigned int i = 0; i < outputPoints.size(); i++)
    {
        if (outputPoints[i].active)
            return true;
    }

    return false;
}


bool Crit3DProject::loadCriteria3DProject(const QString &fileName)
{
    if (fileName.isEmpty())
        return false;

    clear3DProject();
    initializeProject3D();

    if (! loadProjectSettings(fileName))
        return false;

    if (! loadProject3DSettings())
        return false;

    if (! loadProject())
    {
        if (errorType != ERROR_DBGRID && errorType != ERROR_OUTPUTPOINTLIST)
            return false;
    }

    //TO DO: integrate with a specific flag in the UI and/or a specific shell command
    soilFluxes3D::initializeLog(getFilePath(getCompleteFileName(logFileName, PATH_LOG)).toStdString(), getProjectName().toStdString());

    if (meteoPointsLoaded)
    {
        logInfoGUI("Check meteopoints first date...");
        meteoPointsDbFirstTime = findDbPointFirstTime();
    }

    if (! loadCriteria3DParameters())
    {
        return false;
    }

    // soil map and data
    if (soilMapFileName != "")
    {
        logInfoGUI("Load soil map...");
        loadSoilMap(soilMapFileName);
    }

    if (soilDbFileName != "")
        loadSoilDatabase(soilDbFileName);

    // land use map, crop data and tree cover map
    if (landUseMapFileName != "")
    {
        logInfoGUI("Load landuse map...");
        loadLandUseMap(landUseMapFileName);
    }

    if (! cropDbFileName.isEmpty())
        loadCropDatabase(cropDbFileName);

    if (! treeCoverMapFileName.isEmpty())
        loadTreeCoverMap(treeCoverMapFileName);

    QString projectName = getProjectName();
    if (projectName != "" && projectName != "default")
    {
        logInfo("Project " + projectName + " loaded");
    }

    closeLogInfo();

    setProjectLoaded(true);
    return true;
}


bool Crit3DProject::loadCriteria3DParameters()
{
    QString fileName = getCompleteFileName(parametersFileName, PATH_SETTINGS);
    if (! QFile(fileName).exists() || ! QFileInfo(fileName).isFile())
    {
        logError("Missing parameters file: " + fileName);
        return false;
    }
    if (parametersSettings == nullptr)
    {
        logError("parametersSettings is null");
        return false;
    }
    Q_FOREACH (QString group, parametersSettings->childGroups())
    {
        if (group == "snow")
        {
            parametersSettings->beginGroup(group);
            if (parametersSettings->contains("tempMaxWithSnow") && !parametersSettings->value("tempMaxWithSnow").toString().isEmpty())
            {
                snowModel.snowParameters.tempMaxWithSnow = parametersSettings->value("tempMaxWithSnow").toDouble();
            }
            if (parametersSettings->contains("tempMinWithRain") && !parametersSettings->value("tempMinWithRain").toString().isEmpty())
            {
                snowModel.snowParameters.tempMinWithRain = parametersSettings->value("tempMinWithRain").toDouble();
            }
            if (parametersSettings->contains("snowWaterHoldingCapacity") && !parametersSettings->value("snowWaterHoldingCapacity").toString().isEmpty())
            {
                snowModel.snowParameters.snowWaterHoldingCapacity = parametersSettings->value("snowWaterHoldingCapacity").toDouble();
            }
            if (parametersSettings->contains("skinThickness") && !parametersSettings->value("skinThickness").toString().isEmpty())
            {
                snowModel.snowParameters.skinThickness = parametersSettings->value("skinThickness").toDouble();
            }
            if (parametersSettings->contains("snowVegetationHeight") && !parametersSettings->value("snowVegetationHeight").toString().isEmpty())
            {
                snowModel.snowParameters.snowVegetationHeight = parametersSettings->value("snowVegetationHeight").toDouble();
            }
            if (parametersSettings->contains("soilAlbedo") && !parametersSettings->value("soilAlbedo").toString().isEmpty())
            {
                snowModel.snowParameters.soilAlbedo = parametersSettings->value("soilAlbedo").toDouble();
            }
            if (parametersSettings->contains("snowSurfaceDampingDepth") && !parametersSettings->value("snowSurfaceDampingDepth").toString().isEmpty())
            {
                snowModel.snowParameters.snowSurfaceDampingDepth = parametersSettings->value("snowSurfaceDampingDepth").toDouble();
            }
            parametersSettings->endGroup();
        }

        if (group == "soilWaterFluxes")
        {
            parametersSettings->beginGroup(group);

            if (parametersSettings->contains("isInitialWaterPotential") && ! parametersSettings->value("isInitialWaterPotential").toString().isEmpty())
            {
                waterFluxesParameters.isInitialWaterPotential = parametersSettings->value("isInitialWaterPotential").toBool();
            }

            if (parametersSettings->contains("initialWaterPotential") && ! parametersSettings->value("initialWaterPotential").toString().isEmpty())
            {
                waterFluxesParameters.initialWaterPotential = parametersSettings->value("initialWaterPotential").toDouble();
            }

            if (parametersSettings->contains("initialDegreeOfSaturation") && ! parametersSettings->value("initialDegreeOfSaturation").toString().isEmpty())
            {
                waterFluxesParameters.initialDegreeOfSaturation = parametersSettings->value("initialDegreeOfSaturation").toDouble();
            }

            if (parametersSettings->contains("computeOnlySurface") && ! parametersSettings->value("computeOnlySurface").toString().isEmpty())
            {
                waterFluxesParameters.computeOnlySurface = parametersSettings->value("computeOnlySurface").toBool();
            }

            if (parametersSettings->contains("computeAllSoilDepth") && ! parametersSettings->value("computeAllSoilDepth").toString().isEmpty())
            {
                waterFluxesParameters.computeAllSoilDepth = parametersSettings->value("computeAllSoilDepth").toBool();
            }

            if (parametersSettings->contains("imposedComputationDepth") && ! parametersSettings->value("imposedComputationDepth").toString().isEmpty())
            {
                waterFluxesParameters.imposedComputationDepth = parametersSettings->value("imposedComputationDepth").toDouble();
            }

            if (parametersSettings->contains("conductivityHorizVertRatio") && ! parametersSettings->value("conductivityHorizVertRatio").toString().isEmpty())
            {
                waterFluxesParameters.conductivityHorizVertRatio = parametersSettings->value("conductivityHorizVertRatio").toDouble();
            }

            if (parametersSettings->contains("freeCatchmentRunoff") && ! parametersSettings->value("freeCatchmentRunoff").toString().isEmpty())
            {
                waterFluxesParameters.freeCatchmentRunoff = parametersSettings->value("freeCatchmentRunoff").toBool();
            }

            if (parametersSettings->contains("freeBottomDrainage") && ! parametersSettings->value("freeBottomDrainage").toString().isEmpty())
            {
                waterFluxesParameters.freeBottomDrainage = parametersSettings->value("freeBottomDrainage").toBool();
            }

            if (parametersSettings->contains("freeLateralDrainage") && ! parametersSettings->value("freeLateralDrainage").toString().isEmpty())
            {
                waterFluxesParameters.freeLateralDrainage = parametersSettings->value("freeLateralDrainage").toBool();
            }

            if (parametersSettings->contains("modelAccuracy") && ! parametersSettings->value("modelAccuracy").toString().isEmpty())
            {
                waterFluxesParameters.modelAccuracy = parametersSettings->value("modelAccuracy").toInt();
            }

            if (parametersSettings->contains("numberOfThreads") && ! parametersSettings->value("numberOfThreads").toString().isEmpty())
            {
                waterFluxesParameters.numberOfThreads = parametersSettings->value("numberOfThreads").toInt();
            }

            parametersSettings->endGroup();
        }

        if (group == "soilCracking")
        {
            parametersSettings->beginGroup(group);

            // TODO parametri soil crack

            parametersSettings->endGroup();

        }
    }
    return true;
}


bool Crit3DProject::writeCriteria3DParameters(bool isSnow, bool isWater, bool isSoilCrack)
{
    QString fileName = getCompleteFileName(parametersFileName, PATH_SETTINGS);
    if (! QFile(fileName).exists() || ! QFileInfo(fileName).isFile())
    {
        logError("Missing parametersSettings file: " + fileName);
        return false;
    }
    if (parametersSettings == nullptr)
    {
        logError("parametersSettings is null");
        return false;
    }

    if (isSnow)
    {
        parametersSettings->setValue("snow/tempMaxWithSnow", snowModel.snowParameters.tempMaxWithSnow);
        parametersSettings->setValue("snow/tempMinWithRain", snowModel.snowParameters.tempMinWithRain);
        parametersSettings->setValue("snow/snowWaterHoldingCapacity", snowModel.snowParameters.snowWaterHoldingCapacity);
        parametersSettings->setValue("snow/skinThickness", snowModel.snowParameters.skinThickness);
        parametersSettings->setValue("snow/snowVegetationHeight", snowModel.snowParameters.snowVegetationHeight);
        parametersSettings->setValue("snow/soilAlbedo", snowModel.snowParameters.soilAlbedo);
        parametersSettings->setValue("snow/snowSurfaceDampingDepth", snowModel.snowParameters.snowSurfaceDampingDepth);
    }

    if (isWater)
    {
        parametersSettings->setValue("soilWaterFluxes/isInitialWaterPotential", waterFluxesParameters.isInitialWaterPotential);
        parametersSettings->setValue("soilWaterFluxes/initialWaterPotential", waterFluxesParameters.initialWaterPotential);
        parametersSettings->setValue("soilWaterFluxes/initialDegreeOfSaturation", waterFluxesParameters.initialDegreeOfSaturation);

        parametersSettings->setValue("soilWaterFluxes/computeOnlySurface", waterFluxesParameters.computeOnlySurface);
        parametersSettings->setValue("soilWaterFluxes/computeAllSoilDepth", waterFluxesParameters.computeAllSoilDepth);
        parametersSettings->setValue("soilWaterFluxes/imposedComputationDepth", waterFluxesParameters.imposedComputationDepth);

        parametersSettings->setValue("soilWaterFluxes/conductivityHorizVertRatio", waterFluxesParameters.conductivityHorizVertRatio);

        parametersSettings->setValue("soilWaterFluxes/freeCatchmentRunoff", waterFluxesParameters.freeCatchmentRunoff);
        parametersSettings->setValue("soilWaterFluxes/freeBottomDrainage", waterFluxesParameters.freeBottomDrainage);
        parametersSettings->setValue("soilWaterFluxes/freeLateralDrainage", waterFluxesParameters.freeLateralDrainage);

        parametersSettings->setValue("soilWaterFluxes/modelAccuracy", waterFluxesParameters.modelAccuracy);
        parametersSettings->setValue("soilWaterFluxes/numberOfThreads", waterFluxesParameters.numberOfThreads);
    }

    if (isSoilCrack)
    {
        // TODO parametri soil crack
        // parametersSettings->setValue("soilCracking/ ", );

    }

    parametersSettings->sync();

    return true;
}


bool Crit3DProject::check3DProject()
{
    if (!DEM.isLoaded || !meteoPointsLoaded)
    {
        if (! DEM.isLoaded)
            errorString = ERROR_STR_MISSING_DEM;
        else if (! meteoPointsLoaded)
            errorString =  ERROR_STR_MISSING_DB;
        return false;
    }

    return true;
}


double Crit3DProject::getSoilVar(int soilIndex, int layerIndex, soil::soilVariable myVar)
{
    unsigned int hIndex = unsigned(soil::getHorizonIndex(soilList[unsigned(soilIndex)], layerDepth[unsigned(layerIndex)]));

    if (myVar == soil::soilWaterPotentialWP)
        return soilList[unsigned(soilIndex)].horizon[hIndex].wiltingPoint;
    else if (myVar == soil::soilWaterPotentialFC)
        return soilList[unsigned(soilIndex)].horizon[hIndex].fieldCapacity;
    else if (myVar == soil::soilWaterContentSat)
        return soilList[unsigned(soilIndex)].horizon[hIndex].waterContentSAT;
    else if (myVar == soil::soilWaterContentFC)
        return soilList[unsigned(soilIndex)].horizon[hIndex].waterContentFC;
    else if (myVar == soil::soilWaterContentWP)
        return soilList[unsigned(soilIndex)].horizon[hIndex].waterContentWP;
    else
        return NODATA;
}


void Crit3DProject::clear3DProject()
{
    snowMaps.clear();
    snowModel.initialize();

    dailyTminMap.clear();
    dailyTmaxMap.clear();

    degreeDaysMap.clear();

    clearGeometry();

    clearProject3D();
}


bool Crit3DProject::computeAllMeteoMaps(const QDateTime& myTime, bool showInfo)
{
    if (! this->DEM.isLoaded)
    {
        errorString = "Load a Digital Elevation Model (DEM) before.";
        return false;
    }
    if (this->hourlyMeteoMaps == nullptr)
    {
        errorString = "Meteo maps not initialized.";
        return false;
    }

    this->hourlyMeteoMaps->setComputed(false);

    if (showInfo)
    {
        setProgressBar("Computing air temperature...", 6);
    }

    if (! interpolateHourlyMeteoVar(airTemperature, myTime))
        return false;

    if (showInfo)
    {
        updateProgressBarText("Computing air relative humidity...");
        updateProgressBar(1);
    }

    if (! interpolateHourlyMeteoVar(airRelHumidity, myTime))
        return false;

    if (showInfo)
    {
        updateProgressBarText("Computing precipitation...");
        updateProgressBar(2);
    }

    if (! interpolateHourlyMeteoVar(precipitation, myTime))
        return false;

    if (showInfo)
    {
        updateProgressBarText("Computing wind intensity...");
        updateProgressBar(3);
    }

    if (! interpolateHourlyMeteoVar(windScalarIntensity, myTime))
        return false;

    if (showInfo)
    {
        updateProgressBarText("Computing global irradiance...");
        updateProgressBar(4);
    }

    if (! interpolateHourlyMeteoVar(globalIrradiance, myTime))
        return false;

    if (showInfo)
    {
        updateProgressBarText("Computing ET0...");
        updateProgressBar(5);
    }

    if (! this->hourlyMeteoMaps->computeET0PMMap(this->DEM, this->radiationMaps))
        return false;

    if (showInfo) closeProgressBar();

    this->hourlyMeteoMaps->setComputed(true);

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

        aggregateAndSaveDailyMap(airTemperature, aggrMin, crit3DDate, dailyPath, hourlyPath);
        aggregateAndSaveDailyMap(airTemperature, aggrMax, crit3DDate, dailyPath, hourlyPath);
        aggregateAndSaveDailyMap(airTemperature, aggrAverage, crit3DDate, dailyPath,hourlyPath);
        aggregateAndSaveDailyMap(precipitation, aggrSum, crit3DDate, dailyPath, hourlyPath);
        aggregateAndSaveDailyMap(referenceEvapotranspiration, aggrSum, crit3DDate, dailyPath, hourlyPath);
        aggregateAndSaveDailyMap(airRelHumidity, aggrMin, crit3DDate, dailyPath, hourlyPath);
        aggregateAndSaveDailyMap(airRelHumidity, aggrMax, crit3DDate, dailyPath, hourlyPath);
        aggregateAndSaveDailyMap(airRelHumidity, aggrAverage, crit3DDate, dailyPath, hourlyPath);
        aggregateAndSaveDailyMap(globalIrradiance, aggrSum, crit3DDate, dailyPath, hourlyPath);

        removeDirectory(hourlyPath);

        // save crop output

        // save water balance output
    }

    return true;
}


bool Crit3DProject::initializeSnowModel()
{
    if (! DEM.isLoaded)
    {
        errorString = ERROR_STR_MISSING_DEM;
        return false;
    }

    snowMaps.initializeSnowMaps(DEM, snowModel.snowParameters.skinThickness);
    isSnowInitialized = true;

    return true;
}


void Crit3DProject::computeSnowPoint(Crit3DSnow &snowPoint, int row, int col)
{
    snowMaps.setPoint(snowPoint, row, col);

    double airT = hourlyMeteoMaps->mapHourlyTair->value[row][col];
    double prec = hourlyMeteoMaps->mapHourlyPrec->value[row][col];
    double relHum = hourlyMeteoMaps->mapHourlyRelHum->value[row][col];
    double windInt = hourlyMeteoMaps->mapHourlyWindScalarInt->value[row][col];
    double globalRad = radiationMaps->globalRadiationMap->value[row][col];
    double beamRad = radiationMaps->beamRadiationMap->value[row][col];
    double transmissivity = radiationMaps->transmissivityMap->value[row][col];
    double clearSkyTrans = radSettings.getClearSky();
    double myWaterContent = 0;                              // [mm]

    snowPoint.setSnowInputData(airT, prec, relHum, windInt, globalRad, beamRad, transmissivity, clearSkyTrans, myWaterContent);

    snowPoint.computeSnowBrooksModel();

    snowMaps.updateMapRowCol(snowPoint, row, col);
}


// it assumes that header of meteo and snow maps = header of DEM
bool Crit3DProject::computeSnowModel()
{
    // check
    if (! snowMaps.isInitialized)
    {
        logError("Initialize snow model before.");
        return false;
    }

    if (! hourlyMeteoMaps->getComputed())
    {
        logError("Missing meteo maps.");
        return false;
    }

    if (! radiationMaps->getComputed())
    {
        logError("Missing radiation map.");
        return false;
    }

    if (getComputeOnlyPoints())
    {
        for (unsigned int i = 0; i < outputPoints.size(); i++)
        {
            if (outputPoints[i].active)
            {
                double x = outputPoints[i].utm.x;
                double y = outputPoints[i].utm.y;

                int row, col;
                DEM.getRowCol(x, y, row, col);
                if (! gis::isOutOfGridRowCol(row, col, DEM))
                {
                    computeSnowPoint(snowModel, row, col);
                }
            }
        }
    }
    else
    {
        Crit3DSnow snowPoint = snowModel;

        #pragma omp parallel for if (isParallelComputing()) firstprivate(snowPoint)
        for (int row = 0; row < DEM.header->nrRows; row++)
        {
            for (int col = 0; col < DEM.header->nrCols; col++)
            {
                if (! isEqual(DEM.value[row][col], DEM.header->flag))
                {
                    computeSnowPoint(snowPoint, row, col);
                }
                else
                {
                    snowMaps.flagMapRowCol(row, col);
                }
            }
        }

        snowMaps.updateRangeMaps();
    }

    return true;
}


bool Crit3DProject::computeHydrallModel(Crit3DHydrall &myHydrallModel, int row, int col, int forestIndex)
{
    // check
    if (! hourlyMeteoMaps->getComputed())
    {
        logError("Missing meteo maps.");
        return false;
    }
    if (! radiationMaps->getComputed())
    {
        logError("Missing radiation map.");
        return false;
    }
    if (! isCriteria3DInitialized)
    {
        logError("initialize soil fluxes before.");
        return false;
    }

    // set all necessary input variables
    if (! setHydrallVariables(myHydrallModel, row, col, forestIndex))
    {
        logError();
        return false;
    }

    // read state variables from corresponding state maps
    myHydrallModel.setStateVariables(hydrallMaps, row, col);

    // compute
    myHydrallModel.computeHydrallPoint();

    // check and save data
    // TODO CHECK NODATA
    myHydrallModel.saveStateVariables(hydrallMaps, row, col);

    // TODO clean
    //myHydrallModel.getPlantAndSoilVariables(hydrallMaps, row, col);

    return true;
}


bool Crit3DProject::setHydrallVariables(Crit3DHydrall &myHydrallModel, int row, int col, int forestIndex)
{
    //hourly variables
    myHydrallModel.setHourlyVariables(double(hourlyMeteoMaps->mapHourlyTair->value[row][col]), double(radiationMaps->globalRadiationMap->value[row][col]),
                                    double(hourlyMeteoMaps->mapHourlyPrec->value[row][col]), double(hourlyMeteoMaps->mapHourlyRelHum->value[row][col]),
                                    double(hourlyMeteoMaps->mapHourlyWindScalarInt->value[row][col]), double(radiationMaps->beamRadiationMap->value[row][col]),
                                    double(radiationMaps->diffuseRadiationMap->value[row][col]),
                                    double(radiationMaps->transmissivityMap->value[row][col] / CLEAR_SKY_TRANSMISSIVITY_DEFAULT),
                                    pressureFromAltitude(double(hourlyMeteoMaps->mapHourlyTair->value[row][col]), DEM.value[row][col]),
                                    getCrit3DDate(getCurrentDate()),double(radiationMaps->sunElevationMap->value[row][col]),
                                    mapLast30DaysTAvg.value[row][col],double(hourlyMeteoMaps->mapHourlyET0->value[row][col]));

    //TODO: plant height map
    hydrallMaps.plantHeight.value[row][col] = 10;   // [m]
    double chlorophyllContent = 500;                // da tabella

    if (! myHydrallModel.setPlantVariables(forestIndex, chlorophyllContent, hydrallMaps.plantHeight.value[row][col],
                                          hydrallMaps.minLeafWaterPotential->value[row][col]))
    {
        errorString = "Wrong forest index in hydrall model.";
        return false;
    }

    // check soil
    int soilIndex = int(soilIndexMap.value[row][col]);
    if (soilIndex != NODATA)
    {
        //TODO

		// the condition on this for cycle includes the check of existance of the layers
		for (unsigned int i = 0; ((i < nrLayers) && (soilList[soilIndex].getHorizonIndex(layerDepth[i]))!= NODATA); i++)
		{
            hydrallModel.setSoilVariables(i, indexMap.at(i).value[row][col], indexMap.at(i).header->flag,
                                          soilFluxes3D::getNodeWaterContent(indexMap.at(i).value[row][col]),
										  soilList[soilIndex].horizon[soilList[soilIndex].getHorizonIndex(layerDepth[i])].waterContentFC,
										  soilList[soilIndex].horizon[soilList[soilIndex].getHorizonIndex(layerDepth[i])].waterContentWP,
										  soilList[soilIndex].horizon[soilList[soilIndex].getHorizonIndex(layerDepth[i])].texture.clay,
										  soilList[soilIndex].horizon[soilList[soilIndex].getHorizonIndex(layerDepth[i])].texture.sand,
										  fabs(soilList[soilIndex].horizon[soilList[soilIndex].getHorizonIndex(layerDepth[i])].lowerDepth-soilList[soilIndex].horizon[soilList[soilIndex].getHorizonIndex(layerDepth[i])].upperDepth),
										  soilList[soilIndex].horizon[soilList[soilIndex].getHorizonIndex(layerDepth[i])].bulkDensity,
										  soilList[soilIndex].horizon[soilList[soilIndex].getHorizonIndex(layerDepth[i])].waterContentSAT,
										  soilList[soilIndex].horizon[soilList[soilIndex].getHorizonIndex(layerDepth[i])].waterConductivity.kSat,
                                          soilFluxes3D::getNodeMatricPotential(indexMap.at(i).value[row][col]));
		}
	}
    return true;
}


bool Crit3DProject::computeRothCModel()
{
    rothCModel.computeRothCPoint();
    return true;
}


bool Crit3DProject::updateLast30DaysTavg()
{
    if (! dailyTminMap.isLoaded || ! dailyTmaxMap.isLoaded || ! hourlyMeteoMaps->mapHourlyTair->isLoaded)
        return false;

    for (long row = 0; row < dailyTminMap.header->nrRows; row++)
    {
        for (long col = 0; col < dailyTminMap.header->nrCols; col++)
        {
            mapLast30DaysTAvg.value[row][col] = (29./30.)*mapLast30DaysTAvg.value[row][col] + (dailyTmaxMap.value[row][col] + dailyTminMap.value[row][col])/30;
        }
    }
    return true;
}

bool Crit3DProject::updateDailyTemperatures()
{
    if (! dailyTminMap.isLoaded || ! dailyTmaxMap.isLoaded || ! hourlyMeteoMaps->mapHourlyTair->isLoaded)
        return false;

    #pragma omp parallel for if (isParallelComputing())
    for (long row = 0; row < dailyTminMap.header->nrRows; row++)
    {
        for (long col = 0; col < dailyTminMap.header->nrCols; col++)
        {
            float airT = hourlyMeteoMaps->mapHourlyTair->value[row][col];
            if (! isEqual(airT, hourlyMeteoMaps->mapHourlyTair->header->flag))
            {
                float currentTmin = dailyTminMap.value[row][col];
                if (isEqual (currentTmin, dailyTminMap.header->flag))
                {
                    dailyTminMap.value[row][col] = airT;
                }
                else
                {
                    dailyTminMap.value[row][col] = std::min(currentTmin, airT);
                }

                float currentTmax = dailyTmaxMap.value[row][col];
                if (isEqual (currentTmax, dailyTmaxMap.header->flag))
                {
                    dailyTmaxMap.value[row][col] = airT;
                }
                else
                {
                    dailyTmaxMap.value[row][col] = std::max(currentTmax, airT);
                }
            }
        }
    }

    return true;
}


bool Crit3DProject::checkProcesses()
{
    if (! isProjectLoaded())
    {
        errorString = ERROR_STR_MISSING_PROJECT;
        return false;
    }

    if (! DEM.isLoaded)
    {
        errorString = ERROR_STR_MISSING_DEM;
        return false;
    }

    if (! (processes.computeRadiation || processes.computeMeteo ||
          processes.computeCrop || processes.computeWater || processes.computeSnow
          || processes.computeHydrall || processes.computeRothC))
    {
        errorString = ERROR_STR_MISSING_PROCESSES;
        return false;
    }

    if (processes.computeCrop)
    {
        if (! isCropInitialized)
        {
            errorString = ERROR_STR_INITIALIZE_CROP;
            return false;
        }
    }

    if (processes.computeWater)
    {
        if (! isCriteria3DInitialized)
        {
            errorString = ERROR_STR_INITIALIZE_3D;
            return false;
        }
    }

    if (processes.computeSnow)
    {
        if (! snowMaps.isInitialized)
        {
            if (! initializeSnowModel())
                return false;
        }
    }

    if (processes.computeHydrall)
    {
        if (! isHydrallInitialized)
        {
            hydrallModel.initialize();
        }
    }

    if (processes.computeRothC)
    {
        if (! isRothCInitialized)
        {
            initializeRothC();
        }
    }

    return true;
}


bool Crit3DProject::runModelHour(const QString& hourlyOutputPath, bool isRestart)
{
    if (! isRestart)
    {
        QDateTime myDateTime = getCurrentTime();
        currentSeconds = 0;

        hourlyMeteoMaps->setComputed(false);
        radiationMaps->setComputed(false);

        if (processes.computeMeteo)
        {
            if (! interpolateAndSaveHourlyMeteo(airTemperature, myDateTime, hourlyOutputPath, isSaveOutputRaster()))
                return false;

            if (! interpolateAndSaveHourlyMeteo(precipitation, myDateTime, hourlyOutputPath, isSaveOutputRaster()))
                return false;

            if (! interpolateAndSaveHourlyMeteo(airRelHumidity, myDateTime, hourlyOutputPath, isSaveOutputRaster()))
                return false;

            if (! interpolateAndSaveHourlyMeteo(windScalarIntensity, myDateTime, hourlyOutputPath, isSaveOutputRaster()))
                return false;

            hourlyMeteoMaps->setComputed(true);
            qApp->processEvents();
        }

        if (processes.computeRadiation)
        {
            if (! interpolateAndSaveHourlyMeteo(globalIrradiance, myDateTime, hourlyOutputPath, isSaveOutputRaster()))
                return false;

            qApp->processEvents();
        }

        if (processes.computeSnow)
        {
            // TODO: link evaporation to water flow
            // TODO: link snowmelt to surface water content
            if (! computeSnowModel())
            {
                return false;
            }
            qApp->processEvents();
        }

        if (processes.computeWater)
        {
            // initalize sink / source
            for (unsigned long i = 0; i < nrNodes; i++)
            {
                waterSinkSource[i] = 0.;                    // [m3 s-1]
            }
        }

        if (processes.computeCrop || processes.computeWater || processes.computeRothC)
        {
            if (! hourlyMeteoMaps->computeET0PMMap(DEM, radiationMaps))
            {
                errorString = "Missing ET0 values.";
                return false;
            }

            if (isSaveOutputRaster())
            {
                saveHourlyMeteoOutput(referenceEvapotranspiration, hourlyOutputPath, myDateTime);
            }

            if (processes.computeCrop || processes.computeRothC)
            {
                updateDailyTemperatures();
            }

            if (processes.computeWater)
            {
                assignETreal();
            }
            qApp->processEvents();
        }

        if (processes.computeWater)
        {
            assignPrecipitation();

            if (! setSinkSource())
                return false;
        }

        emit updateOutputSignal();
    }

    // soil fluxes
    if (processes.computeWater)
    {
        if (! isRestart)
        {
            logInfo("\nCompute soil fluxes: " + getCurrentTime().toString());
        }
        runWaterFluxes3DModel(3600, isRestart);

        qApp->processEvents();
    }
    else
    {
        currentSeconds = 3600;
    }

    // soil heat
    if (processes.computeHeat)
    {
        //to do;
    }

    return true;
}


bool Crit3DProject::saveModelsState(QString &dirName)
{
    if (! checkProcesses())
        return false;

    QString statePath = getProjectPath() + PATH_STATES;
    if (! QDir(statePath).exists())
    {
        QDir().mkdir(statePath);
    }

    char hourStr[3];
    sprintf(hourStr, "%02d", getCurrentHour());
    //sprintf_s(hourStr, sizeof(hourStr), "%02d", getCurrentHour());
    dirName = getCurrentDate().toString("yyyyMMdd") + "_H" + hourStr;
    QString currentStatePath = statePath + "/" + dirName;
    if (! QDir(currentStatePath).exists())
    {
        QDir().mkdir(currentStatePath);
    }

    if (processes.computeSnow)
    {
        if (! saveSnowModelState(currentStatePath))
            return false;
    }

    if (processes.computeCrop)
    {
        // create crop path
        QString cropPath = currentStatePath + "/crop";
        if (QDir(cropPath).exists())
        {
            QDir(cropPath).removeRecursively();
        }
        QDir().mkdir(cropPath);

        // save degree days (state variable)
        std::string errorStr;
        if (! gis::writeEsriGrid((cropPath + "/degreeDays").toStdString(), &degreeDaysMap, errorStr))
        {
            logError("Error saving degree days map: " + QString::fromStdString(errorStr));
            return false;
        }
    }

    if (processes.computeWater)
    {
        if (! saveSoilWaterState(currentStatePath))
            return false;
    }

    if (processes.computeRothC)
    {
        if (! saveRothCState(currentStatePath))
            return false;
    }

    if (processes.computeHydrall)
    {
        if (! saveHydrallState(currentStatePath))
            return false;
    }

    return true;
}


bool Crit3DProject::saveSoilWaterState(const QString &currentStatePath)
{
    if (! isCriteria3DInitialized)
    {
        logError("Initialize water fluxes model before.");
        return false;
    }

    // check soil layers
    if (layerDepth.size() != nrLayers)
    {
        logError("Wrong number of layers:" + QString::number(nrLayers));
        return false;
    }

    // create water path
    QString waterPath = currentStatePath + "/water";
    if (QDir(waterPath).exists())
    {
        if (! QDir(waterPath).removeRecursively())
        {
            logError("Error deleting water directory.");
        }
    }
    QDir().mkdir(waterPath);

    // save water potential
    gis::Crit3DRasterGrid rasterGrid;
    for (unsigned int i = 0; i < nrLayers; i++)
    {
        if (! computeCriteria3DMap(rasterGrid, waterMatricPotential, i))
        {
            logError();
            return false;
        }

        int depthCm = int(round(layerDepth[i] * 100));
        QString fileName = "WP_" + QString::number(depthCm);
        std::string errorStr;
        if (! gis::writeEsriGrid((waterPath + "/" + fileName).toStdString(), &rasterGrid, errorStr))
        {
            logError("Error saving water potential: " + QString::fromStdString(errorStr));
            return false;
        }
    }

    return true;
}


bool Crit3DProject::saveSnowModelState(const QString &currentStatePath)
{
    if (! snowMaps.isInitialized)
    {
        logError("Initialize snow model before.");
        return false;
    }

    // create snow path
    QString snowPath = currentStatePath + "/snow";
    QDir().mkdir(snowPath);
    QString imgPath = snowPath + "/img";
    QDir().mkdir(imgPath);

    logInfo("Saving snow state: " + currentStatePath);
    std::string errorStr;
    if (!gis::writeEsriGrid((snowPath+"/SWE").toStdString(), snowMaps.getSnowWaterEquivalentMap(), errorStr))
    {
        logError("Error saving water equivalent map: " + QString::fromStdString(errorStr));
        return false;
    }
    // ENVI file
    if (!gis::writeEnviGrid((imgPath+"/SWE").toStdString(), gisSettings.utmZone, snowMaps.getSnowWaterEquivalentMap(), errorStr))
    {
        logError("Error saving water equivalent map (ENVI file): " + QString::fromStdString(errorStr));
        return false;
    }

    if (!gis::writeEsriGrid((snowPath+"/AgeOfSnow").toStdString(), snowMaps.getAgeOfSnowMap(), errorStr))
    {
        logError("Error saving age of snow map: " + QString::fromStdString(errorStr));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/SnowSurfaceTemp").toStdString(), snowMaps.getSnowSurfaceTempMap(), errorStr))
    {
        logError("Error saving snow surface temp map: " + QString::fromStdString(errorStr));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/IceContent").toStdString(), snowMaps.getIceContentMap(), errorStr))
    {
        logError("Error saving ice content map: " + QString::fromStdString(errorStr));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/LWContent").toStdString(), snowMaps.getLWContentMap(), errorStr))
    {
        logError("Error saving LW content map: " + QString::fromStdString(errorStr));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/InternalEnergy").toStdString(), snowMaps.getInternalEnergyMap(), errorStr))
    {
        logError("Error saving internal energy map: " + QString::fromStdString(errorStr));
        return false;
    }
    if (!gis::writeEsriGrid((snowPath+"/SurfaceInternalEnergy").toStdString(), snowMaps.getSurfaceEnergyMap(), errorStr))
    {
        logError("Error saving surface energy map: " + QString::fromStdString(errorStr));
        return false;
    }

    return true;
}

bool Crit3DProject::saveRothCState(const QString &currentStatePath)
{
    if (! isRothCInitialized)
    {
        logError("Initialize rothC model before.");
        return false;
    }

    // create rothC path
    QString rothCPath = currentStatePath + "/rothC";
    QDir().mkdir(rothCPath);

    logInfo("Saving rothC state: " + currentStatePath);
    std::string errorStr;
    if (!gis::writeEsriGrid((rothCPath+"/DPM").toStdString(), rothCModel.map.getDPM(), errorStr))
    {
        logError("Error saving decomposable plant material map: " + QString::fromStdString(errorStr));
        return false;
    }

    if (!gis::writeEsriGrid((rothCPath+"/RPM").toStdString(), rothCModel.map.getRPM(), errorStr))
    {
        logError("Error saving resistant plant material map: " + QString::fromStdString(errorStr));
        return false;
    }

    if (!gis::writeEsriGrid((rothCPath+"/BIO").toStdString(), rothCModel.map.getBIO(), errorStr))
    {
        logError("Error saving microbial biomass map: " + QString::fromStdString(errorStr));
        return false;
    }

    if (!gis::writeEsriGrid((rothCPath+"/HUM").toStdString(), rothCModel.map.getHUM(), errorStr))
    {
        logError("Error saving humified organic matter map: " + QString::fromStdString(errorStr));
        return false;
    }

    if (!gis::writeEsriGrid((rothCPath+"/SOC").toStdString(), rothCModel.map.getSOC(), errorStr))
    {
        logError("Error saving soil organic carbon map: " + QString::fromStdString(errorStr));
        return false;
    }

    return true;
}

bool Crit3DProject::saveHydrallState(const QString &currentStatePath)
{
    if (! isHydrallInitialized)
    {
        logError("Initialize hydrall model before.");
        return false;
    }

    // create hydrall path
    QString hydrallPath = currentStatePath + "/hydrall";
    QDir().mkdir(hydrallPath);

    logInfo("Saving hydrall state: " + currentStatePath);
    std::string errorStr;
    if (!gis::writeEsriGrid((hydrallPath+"/treeNPP").toStdString(), hydrallMaps.treeNetPrimaryProduction, errorStr))
    {
        logError("Error saving tree net primary production map: " + QString::fromStdString(errorStr));
        return false;
    }

    if (!gis::writeEsriGrid((hydrallPath+"/understoreyNPP").toStdString(), hydrallMaps.understoreyNetPrimaryProduction, errorStr))
    {
        logError("Error saving understorey net primary production map: " + QString::fromStdString(errorStr));
        return false;
    }

    if (!gis::writeEsriGrid((hydrallPath+"/treeFoliage").toStdString(), hydrallMaps.treeBiomassFoliage, errorStr))
    {
        logError("Error saving tree foliage biomass map: " + QString::fromStdString(errorStr));
        return false;
    }

    if (!gis::writeEsriGrid((hydrallPath+"/treeRoot").toStdString(), hydrallMaps.treeBiomassRoot, errorStr))
    {
        logError("Error saving tree root biomass map: " + QString::fromStdString(errorStr));
        return false;
    }

    if (!gis::writeEsriGrid((hydrallPath+"/treeStand").toStdString(), hydrallMaps.treeBiomassSapwood, errorStr))
    {
        logError("Error saving tree sapwood biomass map: " + QString::fromStdString(errorStr));
        return false;
    }

    if (!gis::writeEsriGrid((hydrallPath+"/understoreyFoliage").toStdString(), hydrallMaps.understoreyBiomassFoliage, errorStr))
    {
        logError("Error saving understorey foliage biomass map: " + QString::fromStdString(errorStr));
        return false;
    }

    if (!gis::writeEsriGrid((hydrallPath+"/understoreyRoot").toStdString(), hydrallMaps.understoreyBiomassRoot, errorStr))
    {
        logError("Error saving understorey root biomass map: " + QString::fromStdString(errorStr));
        return false;
    }

    if (!gis::writeEsriGrid((hydrallPath+"/yearlyET0").toStdString(), hydrallMaps.yearlyET0, errorStr))
    {
        logError("Error saving yearly ET0 map: " + QString::fromStdString(errorStr));
        return false;
    }

    if (!gis::writeEsriGrid((hydrallPath+"/yearlyPrec").toStdString(), hydrallMaps.yearlyPrec, errorStr))
    {
        logError("Error saving yearly prec map: " + QString::fromStdString(errorStr));
        return false;
    }

    if (!gis::writeEsriGrid((hydrallPath+"/last30daysT").toStdString(), &mapLast30DaysTAvg, errorStr))
    {
        logError("Error saving average last 30 days temperature map: " + QString::fromStdString(errorStr));
        return false;
    }

    //other maps tbd

    return true;
}


bool Crit3DProject::getAllSavedState(QList<QString> &stateList)
{
    QString statesPath = getProjectPath() + PATH_STATES;
    QDir dir(statesPath);
    if (! dir.exists())
    {
        errorString = "STATES directory is missing: " + statesPath;
        return false;
    }
    // only directories, filter by name
    QFileInfoList list = dir.entryInfoList(QDir::AllDirs | QDir::NoDot | QDir::NoDotDot | QDir::NoSymLinks, QDir::Name);

    if (list.size() == 0)
    {
        errorString = "STATES directory is empty: " + statesPath;
        return false;
    }

    for (int i=0; i < list.size(); i++)
    {
        if (list[i].baseName().size() == 12)
        {
            stateList << list[i].baseName();
        }
    }

    return true;
}


bool Crit3DProject::loadModelState(QString statePath)
{
    QDir stateDir(statePath);
    if (! stateDir.exists())
    {
        errorString = "This state does not exist: " + statePath;
        return false;
    }

    // set current date/hour
    QString stateStr = getFileName(statePath);
    int year = stateStr.mid(0,4).toInt();
    int month = stateStr.mid(4,2).toInt();
    int day = stateStr.mid(6,2).toInt();
    int hour = stateStr.mid(10,2).toInt();
    if (hour == 24)
    {
        setCurrentDate(QDate(year, month, day).addDays(1));
        setCurrentHour(0);
    }
    else
    {
        setCurrentDate(QDate(year, month, day));
        setCurrentHour(hour);
    }

    std::string errorStr, fileName;

    bool isProcessesDefined = (isSnowInitialized || isCropInitialized || isCriteria3DInitialized
                               || isHydrallInitialized || isRothCInitialized);

    // snow model
    QString snowPath = statePath + "/snow";
    QDir snowDir(snowPath);
    if (snowDir.exists() && (!isProcessesDefined || isSnowInitialized))
    {
        if (! initializeSnowModel())
            return false;

        gis::Crit3DRasterGrid *tmpRaster = new gis::Crit3DRasterGrid();

        fileName = snowPath.toStdString() + "/SWE";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong Snow SWE map:\n" + QString::fromStdString(errorStr);
            snowMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, snowMaps.getSnowWaterEquivalentMap(), DEM.header, aggrAverage, 0.1f);

        fileName = snowPath.toStdString() + "/AgeOfSnow";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong Snow AgeOfSnow map:\n" + QString::fromStdString(errorStr);
            snowMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, snowMaps.getAgeOfSnowMap(), DEM.header, aggrAverage, 0.1f);

        fileName = snowPath.toStdString() + "/IceContent";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong Snow IceContent map:\n" + QString::fromStdString(errorStr);
            snowMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, snowMaps.getIceContentMap(), DEM.header, aggrAverage, 0.1f);

        fileName = snowPath.toStdString() + "/InternalEnergy";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong Snow InternalEnergy map:\n" + QString::fromStdString(errorStr);
            snowMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, snowMaps.getInternalEnergyMap(), DEM.header, aggrAverage, 0.1f);

        fileName = snowPath.toStdString() + "/LWContent";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong Snow LWContent map:\n" + QString::fromStdString(errorStr);
            snowMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, snowMaps.getLWContentMap(), DEM.header, aggrAverage, 0.1f);

        fileName = snowPath.toStdString() + "/SnowSurfaceTemp";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong Snow SurfaceTemp map:\n" + QString::fromStdString(errorStr);
            snowMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, snowMaps.getSnowSurfaceTempMap(), DEM.header, aggrAverage, 0.1f);

        fileName = snowPath.toStdString() + "/SurfaceInternalEnergy";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong Snow SurfaceInternalEnergy map:\n" + QString::fromStdString(errorStr);
            snowMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, snowMaps.getSurfaceEnergyMap(), DEM.header, aggrAverage, 0.1f);

        processes.setComputeSnow(true);
    }

    // crop model
    QString cropPath = statePath + "/crop";
    QDir cropDir(cropPath);
    if (cropDir.exists() && (!isProcessesDefined || isCropInitialized))
    {
        gis::Crit3DRasterGrid myDegreeDaysMap;
        fileName = cropPath.toStdString() + "/degreeDays";
        if (! gis::readEsriGrid(fileName, &myDegreeDaysMap, errorStr))
        {
            errorString = "Wrong degree days map:\n" + QString::fromStdString(errorStr);
            return false;
        }

        if (! initializeCropFromDegreeDays(myDegreeDaysMap))
            return false;

        processes.setComputeCrop(true);
    }

    // water fluxes
    QString waterPath = statePath + "/water";
    QDir waterDir(waterPath);
    if (waterDir.exists() && (!isProcessesDefined || isCriteria3DInitialized))
    {
        if (! loadWaterPotentialState(waterPath))
        {
            isCriteria3DInitialized = false;
            processes.setComputeWater(false);
            return false;
        }

        processes.setComputeWater(true);
    }

    //rothC model
    QString rothCPath = statePath + "/rothC";
    QDir rothCDir(rothCPath);
    gis::Crit3DRasterGrid *tmpRaster = new gis::Crit3DRasterGrid();
    if (rothCDir.exists() && (! isProcessesDefined || isRothCInitialized))
    {
        rothCModel.initialize();
        rothCModel.map.initialize(DEM);

        fileName = rothCPath.toStdString() + "/DPM";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong RothC decomposable plant matter map:\n" + QString::fromStdString(errorStr);
            rothCModel.map.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, rothCModel.map.decomposablePlantMaterial, DEM.header, aggrAverage, 0.1f);

        fileName = rothCPath.toStdString() + "/RPM";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong RothC resistant plant matter map:\n" + QString::fromStdString(errorStr);
            rothCModel.map.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, rothCModel.map.resistantPlantMaterial, DEM.header, aggrAverage, 0.1f);

        fileName = rothCPath.toStdString() + "/BIO";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong RothC microbial biomass map:\n" + QString::fromStdString(errorStr);
            rothCModel.map.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, rothCModel.map.microbialBiomass, DEM.header, aggrAverage, 0.1f);

        fileName = rothCPath.toStdString() + "/HUM";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong RothC humified organic matter map:\n" + QString::fromStdString(errorStr);
            rothCModel.map.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, rothCModel.map.humifiedOrganicMatter, DEM.header, aggrAverage, 0.1f);

        fileName = rothCPath.toStdString() + "/SOC";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong RothC soil organic carbon map:\n" + QString::fromStdString(errorStr);
            rothCModel.map.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, rothCModel.map.soilOrganicMatter, DEM.header, aggrAverage, 0.1f);
    }

    //hydrall model
    QString hydrallPath = statePath + "/hydrall";
    QDir hydrallDir(hydrallPath);
    tmpRaster = new gis::Crit3DRasterGrid();
    if (hydrallDir.exists() && (! isProcessesDefined || isHydrallInitialized))
    {
        fileName = hydrallPath.toStdString() + "/treeNPP";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong hydrall tree net primary production map:\n" + QString::fromStdString(errorStr);
            hydrallMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, hydrallMaps.treeNetPrimaryProduction, DEM.header, aggrAverage, 0.1f);

        fileName = hydrallPath.toStdString() + "/understoreyNPP";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong hydrall understorey net primary production map:\n" + QString::fromStdString(errorStr);
            hydrallMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, hydrallMaps.understoreyNetPrimaryProduction, DEM.header, aggrAverage, 0.1f);

        fileName = hydrallPath.toStdString() + "/treeFoliage";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong hydrall tree foliage biomass map:\n" + QString::fromStdString(errorStr);
            hydrallMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, hydrallMaps.treeBiomassFoliage, DEM.header, aggrAverage, 0.1f);

        fileName = hydrallPath.toStdString() + "/treeRoot";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong hydrall tree root biomass map:\n" + QString::fromStdString(errorStr);
            hydrallMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, hydrallMaps.treeBiomassRoot, DEM.header, aggrAverage, 0.1f);

        fileName = hydrallPath.toStdString() + "/treeStand";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong hydrall tree sapwood biomass map:\n" + QString::fromStdString(errorStr);
            hydrallMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, hydrallMaps.treeBiomassSapwood, DEM.header, aggrAverage, 0.1f);

        fileName = hydrallPath.toStdString() + "/understoreyFoliage";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong hydrall understorey foliage biomass map:\n" + QString::fromStdString(errorStr);
            hydrallMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, hydrallMaps.understoreyBiomassFoliage, DEM.header, aggrAverage, 0.1f);

        fileName = hydrallPath.toStdString() + "/understoreyRoot";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong hydrall understorey root biomass map:\n" + QString::fromStdString(errorStr);
            hydrallMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, hydrallMaps.understoreyBiomassRoot, DEM.header, aggrAverage, 0.1f);

        fileName = hydrallPath.toStdString() + "/yearlyET0";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong hydrall yearly ET0 map:\n" + QString::fromStdString(errorStr);
            hydrallMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, hydrallMaps.yearlyET0, DEM.header, aggrAverage, 0.1f);

        fileName = hydrallPath.toStdString() + "/yearlyPrec";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong hydrall yearly prec map:\n" + QString::fromStdString(errorStr);
            hydrallMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, hydrallMaps.yearlyPrec, DEM.header, aggrAverage, 0.1f);

        fileName = hydrallPath.toStdString() + "/last30daysT";
        if (! gis::readEsriGrid(fileName, tmpRaster, errorStr))
        {
            errorString = "Wrong hydrall last 30 days average temperature map:\n" + QString::fromStdString(errorStr);
            hydrallMaps.isInitialized = false;
            return false;
        }
        gis::resampleGrid(*tmpRaster, &mapLast30DaysTAvg, DEM.header, aggrAverage, 0.1f);


        //other maps tbd
    }

    logInfo("Current date is: " + getCurrentDate().toString("yyyy-MM-dd") + " hour: " + QString::number(getCurrentHour()));

    return true;
}


bool Crit3DProject::loadWaterPotentialState(QString waterPath)
{
    QDir waterDir(waterPath);

    QStringList filters ("*.flt");
    QFileInfoList fileList = waterDir.entryInfoList (filters);
    if (fileList.isEmpty())
    {
        errorString = "Water directory is empty.";
        return false;
    }

    if (! isCriteria3DInitialized)
    {
        logWarning("The water flow model will be initialized with the current settings.");
        if (! initializeCriteria3DModel())
        {
            logError();
            return false;
        }
    }

    std::vector<int> depthList;
    for (unsigned i = 0; i < (unsigned)fileList.size(); i++)
    {
        QString fileName = fileList.at(i).fileName();
        QString leftFileName = fileName.left(fileName.size() - 4);
        QString depthStr = leftFileName.right(leftFileName.size() - 3);
        bool isOk;
        int currentDepth = depthStr.toInt(&isOk);
        if (isOk)
        {
            depthList.push_back(currentDepth);
        }
    }

    if (depthList.empty())
    {
        errorString = "Missing depth in water potential fileName.";
        return false;
    }

    std::sort(depthList.begin(), depthList.end());
    double maxReadingDepth = *std::max_element(depthList.begin(), depthList.end()) / 100.;      // [m]
    double maxDepth = layerDepth[nrLayers-1];                                                   // [m]

    // check on data presence
    if (computationSoilDepth > 0)
    {
        double deltaDepth = std::max(0., maxDepth - maxReadingDepth);
        if ( (1. - deltaDepth/maxDepth) * 100 < meteoSettings->getMinimumPercentage() )
        {
            errorString = "Water potential data is not enough to cover the computation depth: "
                          + QString::number(computationSoilDepth) + " m";
            return false;
        }
    }

    std::vector<gis::Crit3DRasterGrid*> waterPotentialMapList;
    for (unsigned i = 0; i < depthList.size(); i++)
    {
        std::string fileName = waterPath.toStdString() + "/WP_" + std::to_string(depthList[i]);
        std::string errorStr;
        gis::Crit3DRasterGrid *currentWaterPotentialMap = new gis::Crit3DRasterGrid();
        if (! gis::readEsriGrid(fileName, currentWaterPotentialMap, errorStr))
        {
            errorString = "Wrong water potential map:\n" + QString::fromStdString(errorStr);
            return false;
        }
        waterPotentialMapList.push_back(currentWaterPotentialMap);
    }

    for (unsigned int layer = 0; layer < nrLayers; layer ++)
    {
        int currentDepthCm = int(round(layerDepth[layer] * 100));
        int lastDepthIndex = int(depthList.size()) - 1;
        int layer0, layer1;
        double w0, w1;
        int i = 0;
        while (currentDepthCm > depthList[i] && i < lastDepthIndex)
        {
            i++;
        }
        if (currentDepthCm == depthList[i])
        {
            layer0 = i;
            layer1 = i;
        }
        else
        {
            if (currentDepthCm > depthList[i])
            {
                layer0 = i;
                layer1 = std::min(i+1, lastDepthIndex);
            }
            else
            {
                layer0 = std::max(0, i-1);
                layer1 = i;
            }
        }
        int delta = depthList[layer1] - depthList[layer0];
        if (delta == 0)
        {
            w0 = 1;
            w1 = 0;
        }
        else
        {
            w0 = (currentDepthCm - depthList[layer0]) / delta;
            w1 = 1 - w0;
        }

        float flag = waterPotentialMapList.at(layer0)->header->flag;
        for (int row = 0; row < indexMap.at(layer).header->nrRows; row++)
        {
            for (int col = 0; col < indexMap.at(layer).header->nrCols; col++)
            {
                long index = long(indexMap.at(layer).value[row][col]);
                if (index != long(indexMap.at(layer).header->flag))
                {
                    double x, y;
                    float waterPotential = NODATA;

                    gis::getUtmXYFromRowCol(*(indexMap.at(layer).header), row, col, &x, &y);
                    float wp0 = gis::getValueFromXY(*(waterPotentialMapList.at(layer0)), x, y);

                    if (! isEqual(wp0, flag))
                    {
                        // valid value
                        waterPotential = wp0;
                        if (w1 > 0)
                        {
                            float wp1 = gis::getValueFromXY(*(waterPotentialMapList.at(layer1)), x, y);
                            if (! isEqual(wp1, flag))
                            {
                                waterPotential = (w0 * wp0) + (w1 * wp1);
                            }
                        }
                    }
                    else
                    {
                        // search first valid value
                        int lastDataLayer = layer0 - 1;
                        while (isEqual(wp0, flag) && lastDataLayer > 0)
                        {
                            wp0 = gis::getValueFromXY(*(waterPotentialMapList.at(lastDataLayer)), x, y);
                            if (isEqual(wp0, flag))
                            {
                                lastDataLayer--;
                            }
                        }

                        if (lastDataLayer == 0)
                        {
                            errorString = "Missing water potential data in row, col: "
                                          + QString::number(row) + ", " +  QString::number(col);
                            //return false;
                        }

                        double deltaDepth = (currentDepthCm - depthList[lastDataLayer]) / 100.;
                        if ( (1. - deltaDepth/maxDepth) * 100 < meteoSettings->getMinimumPercentage())
                        {
                            errorString = "The water potential data is not enough to cover the data in row, col: "
                                            + QString::number(row) + ", " +  QString::number(col);
                            //return false;
                        }

                        if (! isEqual(wp0, flag))
                        {
                            waterPotential = wp0;
                        }
                    }

                    if (! isEqual(waterPotential, NODATA))
                    {
                        auto myResult = soilFluxes3D::setNodeMatricPotential(index, waterPotential);
                        std::string errorName = "";
                        if(soilFluxes3D::getSF3DerrorName(myResult, errorName))
                        {
                            errorString = "Error in setMatricPotential: " + QString::fromStdString(errorName) + " in row:"
                                          + QString::number(row) + " col:" + QString::number(col);
                            return false;
                        }
                    }
                }
            }
        }
    }

    return true;
}


bool Crit3DProject::writeOutputPointsTables()
{
    if (outputPointsDbHandler == nullptr)
    {
        errorString = "Open output DB before.";
        return false;
    }

    for (unsigned int i = 0; i < outputPoints.size(); i++)
    {
        if (outputPoints[i].active)
        {
            QString tableName = QString::fromStdString(outputPoints[i].id);
            if (! outputPointsDbHandler->createTable(tableName, errorString))
                return false;

            if (processes.computeMeteo)
            {
                if (! outputPointsDbHandler->addColumn(tableName, airTemperature, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, precipitation, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, airRelHumidity, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, windScalarIntensity, errorString)) return false;
            }
            if (processes.computeRadiation)
            {
                if (! outputPointsDbHandler->addColumn(tableName, atmTransmissivity, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, globalIrradiance, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, directIrradiance, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, diffuseIrradiance, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, reflectedIrradiance, errorString)) return false;
            }
            if (processes.computeSnow)
            {
                if (! outputPointsDbHandler->addColumn(tableName, snowWaterEquivalent, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, snowFall, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, snowMelt, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, snowSurfaceTemperature, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, snowSurfaceEnergy, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, snowInternalEnergy, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, sensibleHeat, errorString)) return false;
                if (! outputPointsDbHandler->addColumn(tableName, latentHeat, errorString)) return false;
            }
            if (processes.computeWater)
            {
                for (int l = 0; l < (int)waterContentDepth.size(); l++)
                {
                    int depth_cm = waterContentDepth[l];
                    if (! outputPointsDbHandler->addCriteria3DColumn(tableName, volumetricWaterContent, depth_cm, errorString))
                        return false;
                }

                for (int l = 0; l < (int)waterPotentialDepth.size(); l++)
                {
                    int depth_cm = waterPotentialDepth[l];
                    if (! outputPointsDbHandler->addCriteria3DColumn(tableName, waterMatricPotential, depth_cm, errorString))
                        return false;
                }

                for (int l = 0; l < (int)degreeOfSaturationDepth.size(); l++)
                {
                    int depth_cm = degreeOfSaturationDepth[l];
                    if (! outputPointsDbHandler->addCriteria3DColumn(tableName, degreeOfSaturation, depth_cm, errorString))
                        return false;
                }

                for (int l = 0; l < (int)factorOfSafetyDepth.size(); l++)
                {
                    int depth_cm = factorOfSafetyDepth[l];
                    if (! outputPointsDbHandler->addCriteria3DColumn(tableName, factorOfSafety, depth_cm, errorString))
                        return false;
                }
            }
        }
    }

    return true;
}


bool Crit3DProject::writeOutputPointsData()
{
    QString tableName;
    std::vector<meteoVariable> meteoVarList;
    std::vector<float> meteoValuesList;
    std::vector<float> criteria3dValuesList;

    if (processes.computeMeteo)
    {
        meteoVarList.push_back(airTemperature);
        meteoVarList.push_back(precipitation);
        meteoVarList.push_back(airRelHumidity);
        meteoVarList.push_back(windScalarIntensity);
    }
    if (processes.computeRadiation)
    {
        meteoVarList.push_back(atmTransmissivity);
        meteoVarList.push_back(globalIrradiance);
        meteoVarList.push_back(directIrradiance);
        meteoVarList.push_back(diffuseIrradiance);
        meteoVarList.push_back(reflectedIrradiance);
    }
    if (processes.computeSnow)
    {
        meteoVarList.push_back(snowWaterEquivalent);
        meteoVarList.push_back(snowFall);
        meteoVarList.push_back(snowMelt);
        meteoVarList.push_back(snowSurfaceTemperature);
        meteoVarList.push_back(snowSurfaceEnergy);
        meteoVarList.push_back(snowInternalEnergy);
        meteoVarList.push_back(sensibleHeat);
        meteoVarList.push_back(latentHeat);
    }

    for (unsigned int i = 0; i < outputPoints.size(); i++)
    {
        if (outputPoints[i].active)
        {
            double x = outputPoints[i].utm.x;
            double y = outputPoints[i].utm.y;
            tableName = QString::fromStdString(outputPoints[i].id);

            if (processes.computeMeteo)
            {
                meteoValuesList.push_back(hourlyMeteoMaps->mapHourlyTair->getValueFromXY(x, y));
                meteoValuesList.push_back(hourlyMeteoMaps->mapHourlyPrec->getValueFromXY(x, y));
                meteoValuesList.push_back(hourlyMeteoMaps->mapHourlyRelHum->getValueFromXY(x, y));
                meteoValuesList.push_back(hourlyMeteoMaps->mapHourlyWindScalarInt->getValueFromXY(x, y));
            }
            if (processes.computeRadiation)
            {
                meteoValuesList.push_back(radiationMaps->transmissivityMap->getValueFromXY(x, y));
                meteoValuesList.push_back(radiationMaps->globalRadiationMap->getValueFromXY(x, y));
                meteoValuesList.push_back(radiationMaps->beamRadiationMap->getValueFromXY(x, y));
                meteoValuesList.push_back(radiationMaps->diffuseRadiationMap->getValueFromXY(x, y));
                meteoValuesList.push_back(radiationMaps->reflectedRadiationMap->getValueFromXY(x, y));
            }
            if (processes.computeSnow)
            {
                meteoValuesList.push_back(snowMaps.getSnowWaterEquivalentMap()->getValueFromXY(x, y));
                meteoValuesList.push_back(snowMaps.getSnowFallMap()->getValueFromXY(x, y));
                meteoValuesList.push_back(snowMaps.getSnowMeltMap()->getValueFromXY(x, y));
                meteoValuesList.push_back(snowMaps.getSnowSurfaceTempMap()->getValueFromXY(x, y));
                meteoValuesList.push_back(snowMaps.getSurfaceEnergyMap()->getValueFromXY(x, y));
                meteoValuesList.push_back(snowMaps.getInternalEnergyMap()->getValueFromXY(x, y));
                meteoValuesList.push_back(snowMaps.getSensibleHeatMap()->getValueFromXY(x, y));
                meteoValuesList.push_back(snowMaps.getLatentHeatMap()->getValueFromXY(x, y));
            }
            if (processes.computeWater)
            {
                int row, col;
                gis::getRowColFromXY((*DEM.header), x, y, &row, &col);

                appendCriteria3DOutputValue(volumetricWaterContent, row, col, waterContentDepth, criteria3dValuesList);
                appendCriteria3DOutputValue(waterMatricPotential, row, col, waterPotentialDepth, criteria3dValuesList);
                appendCriteria3DOutputValue(degreeOfSaturation, row, col, degreeOfSaturationDepth, criteria3dValuesList);
                appendCriteria3DOutputValue(factorOfSafety, row, col, factorOfSafetyDepth, criteria3dValuesList);
            }

            if (! outputPointsDbHandler->saveHourlyMeteoData(tableName, getCurrentTime(), meteoVarList, meteoValuesList, errorString))
            {
                return false;
            }
            if (! outputPointsDbHandler->saveHourlyCriteria3D_Data(tableName, getCurrentTime(), criteria3dValuesList,
                                                                  waterContentDepth, waterPotentialDepth,
                                                                  degreeOfSaturationDepth, factorOfSafetyDepth, errorString))
            {
                return false;
            }

            meteoValuesList.clear();
            criteria3dValuesList.clear();
        }
    }

    meteoVarList.clear();

    return true;
}

void Crit3DProject::appendCriteria3DOutputValue(criteria3DVariable myVar, int row, int col,
                                                const std::vector<int> &depthList, std::vector<float> &outputList)
{
    for (unsigned int l = 0; l < depthList.size(); l++)
    {
        float depth = depthList[l] * 0.01;                          // [cm] -> [m]
        int layerIndex = getSoilLayerIndex(depth);
        long nodeIndex = indexMap.at(layerIndex).value[row][col];
        float value = NODATA;

        if (nodeIndex != indexMap.at(layerIndex).header->flag)
        {
            if (myVar == factorOfSafety)
            {
                value = computeFactorOfSafety(row, col, layerIndex);
            }
            else
            {
                value = getCriteria3DVar(myVar, nodeIndex);
            }
        }

        outputList.push_back(value);
    }
}


bool Crit3DProject::writeMeteoPointsProperties(const QList<QString> &joinedPropertiesList, const QList<QString> &csvFields,
                                              const QList<QList<QString>> &csvData)
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

        if (! meteoPointsDbHandler->updatePointProperties(propertiesList, csvDataList))
        {
            errorString = meteoPointsDbHandler->getErrorString();
            return false;
        }
    }

    return true;
}


//------------------------------------- 3D geometry and color --------------------------------------

void Crit3DProject::clearGeometry()
{
    if (openGlGeometry != nullptr)
    {
        openGlGeometry->clear();
        delete openGlGeometry;
        openGlGeometry = nullptr;
    }
}


bool Crit3DProject::initializeGeometry()
{
    if (! DEM.isLoaded)
    {
        errorString = ERROR_STR_MISSING_DEM;
        return false;
    }

    this->clearGeometry();
    openGlGeometry = new Crit3DGeometry();

    // set center
    gis::Crit3DPoint center = DEM.getCenter();
    gis::updateMinMaxRasterGrid(&DEM);
    float zCenter = (DEM.maximum + DEM.minimum) * 0.5f;
    openGlGeometry->setCenter(float(center.utm.x), float(center.utm.y), zCenter);

    // set dimension
    float dx = float(DEM.header->nrCols * DEM.header->cellSize);
    float dy = float(DEM.header->nrRows * DEM.header->cellSize);
    float dz = DEM.maximum + DEM.minimum;
    openGlGeometry->setDimension(dx, dy);
    float magnify = ((dx + dy) * 0.5f) / (dz * 10.f);
    openGlGeometry->setMagnify(std::min(5.f, std::max(1.f, magnify)));

    // set triangles
    double x, y;
    float z1, z2, z3;
    gis::Crit3DPoint p1, p2, p3;
    Crit3DColor *c1, *c2, *c3;
    Crit3DColor sc1, sc2, sc3;
    for (long row = 0; row < DEM.header->nrRows; row++)
    {
        for (long col = 0; col < DEM.header->nrCols; col++)
        {
            z1 = DEM.getValueFromRowCol(row, col);
            if (! isEqual(z1, DEM.header->flag))
            {
                DEM.getXY(row, col, x, y);
                p1 = gis::Crit3DPoint(x, y, z1);
                c1 = DEM.colorScale->getColor(z1);
                shadowColor(*c1, sc1, row, col);

                z3 = DEM.getValueFromRowCol(row+1, col+1);
                if (! isEqual(z3, DEM.header->flag))
                {
                    DEM.getXY(row+1, col+1, x, y);
                    p3 = gis::Crit3DPoint(x, y, z3);
                    c3 = DEM.colorScale->getColor(z3);
                    shadowColor(*c3, sc3, row+1, col+1);

                    z2 = DEM.getValueFromRowCol(row+1, col);
                    if (! isEqual(z2, DEM.header->flag))
                    {
                        DEM.getXY(row+1, col, x, y);
                        p2 = gis::Crit3DPoint(x, y, z2);
                        c2 = DEM.colorScale->getColor(z2);
                        shadowColor(*c2, sc2, row+1, col);
                        openGlGeometry->addTriangle(p1, p2, p3, sc1, sc2, sc3);
                    }

                    z2 = DEM.getValueFromRowCol(row, col+1);
                    if (! isEqual(z2, DEM.header->flag))
                    {
                        DEM.getXY(row, col+1, x, y);
                        p2 = gis::Crit3DPoint(x, y, z2);
                        c2 = DEM.colorScale->getColor(z2);
                        shadowColor(*c2, sc2, row, col+1);
                        openGlGeometry->addTriangle(p3, p2, p1, sc3, sc2, sc1);
                    }
                }
            }
        }
    }

    return true;
}


void Crit3DProject::shadowColor(const Crit3DColor &colorIn, Crit3DColor &colorOut, int row, int col)
{
    colorOut.red = colorIn.red;
    colorOut.green = colorIn.green;
    colorOut.blue = colorIn.blue;

    float aspect = radiationMaps->aspectMap->getValueFromRowCol(row, col);
    if (! isEqual(aspect, radiationMaps->aspectMap->header->flag))
    {
        float slopeDegree = radiationMaps->slopeMap->getValueFromRowCol(row, col);
        if (! isEqual(slopeDegree, radiationMaps->slopeMap->header->flag))
        {
            float slopeAmplification = 120.f / std::max(radiationMaps->slopeMap->maximum, 1.f);
            float shadow = -cos(aspect * DEG_TO_RAD) * std::max(5.f, slopeDegree * slopeAmplification);

            colorOut.red = std::min(255, std::max(0, int(colorOut.red + shadow)));
            colorOut.green = std::min(255, std::max(0, int(colorOut.green + shadow)));
            colorOut.blue = std::min(255, std::max(0, int(colorOut.blue + shadow)));
            if (slopeDegree > openGlGeometry->artifactSlope())
            {
                colorOut.red = std::min(255, std::max(0, int((colorOut.red + 256) / 2)));
                colorOut.green = std::min(255, std::max(0, int((colorOut.green + 256) / 2)));
                colorOut.blue = std::min(255, std::max(0, int((colorOut.blue + 256) / 2)));
            }
        }
    }
}


bool Crit3DProject::update3DColors(gis::Crit3DRasterGrid *rasterPointer)
{
    const double DEFAULT_ALPHA = 0.7;
    const double ALPHA_POW = 0.25;

    if (openGlGeometry == nullptr)
    {
        errorString = "Initialize 3D openGlGeometry before.";
        return false;
    }

    bool isShowVariable = false;
    if (rasterPointer != nullptr)
    {
        if (rasterPointer->header->isEqualTo(*(DEM.header)))
        {
            isShowVariable = true;
        }
    }

    float z1, z2, z3, value;
    Crit3DColor dtmColor1, dtmColor2, dtmColor3;
    Crit3DColor color1, color2, color3;             // final colors

    double variableRange = 0;
    if (isShowVariable)
    {
        variableRange = std::max(EPSILON, rasterPointer->colorScale->maximum() - rasterPointer->colorScale->minimum());
    }

    long i = 0;
    for (long row = 0; row < DEM.header->nrRows; row++)
    {
        for (long col = 0; col < DEM.header->nrCols; col++)
        {
            z1 = DEM.getValueFromRowCol(row, col);
            if (! isEqual(z1, DEM.header->flag))  
            {
                z3 = DEM.getValueFromRowCol(row+1, col+1);
                if (! isEqual(z3, DEM.header->flag))
                {
                    Crit3DColor* c1 = DEM.colorScale->getColor(z1);
                    shadowColor(*c1, dtmColor1, row, col);
                    color1 = dtmColor1;

                    Crit3DColor* c3 = DEM.colorScale->getColor(z3);
                    shadowColor(*c3, dtmColor3, row+1, col+1);
                    color3 = dtmColor3;

                    if (isShowVariable)
                    {
                        value = rasterPointer->getValueFromRowCol(row, col);
                        if (! isEqual(value, rasterPointer->header->flag))
                        {
                            Crit3DColor* variableColor = rasterPointer->colorScale->getColor(value);
                            double alpha = DEFAULT_ALPHA;

                            // check outliers
                            if (rasterPointer->colorScale->isHideMinimum())
                            {
                                if (value == 0 || value < rasterPointer->colorScale->minimum())
                                    alpha = 0;
                            }
                            if (rasterPointer->colorScale->isTransparent())
                            {
                                double step = std::max(0., value - rasterPointer->colorScale->minimum());
                                alpha = std::min(1., step/variableRange);
                                alpha = pow(alpha, ALPHA_POW);
                            }
                            mixColor(dtmColor1, *variableColor, color1, alpha);
                        }

                        value = rasterPointer->getValueFromRowCol(row+1, col+1);
                        if (! isEqual(value, rasterPointer->header->flag))
                        {
                            Crit3DColor* variableColor = rasterPointer->colorScale->getColor(value);
                            double alpha = DEFAULT_ALPHA;

                            // check minimum (transparent)
                            if (rasterPointer->colorScale->isHideMinimum())
                            {
                                if (isEqual(value, 0) || value <= rasterPointer->colorScale->minimum())
                                    alpha = 0;
                            }
                            if (rasterPointer->colorScale->isTransparent())
                            {
                                double step = std::max(0., value - rasterPointer->colorScale->minimum());
                                alpha = std::min(1., step/variableRange);
                                alpha = pow(alpha, ALPHA_POW);
                            }
                            mixColor(dtmColor3, *variableColor, color3, alpha);
                        }
                    }

                    z2 = DEM.getValueFromRowCol(row+1, col);
                    if (! isEqual(z2, DEM.header->flag))
                    {
                        Crit3DColor* c2 = DEM.colorScale->getColor(z2);
                        shadowColor(*c2, dtmColor2, row+1, col);
                        color2 = dtmColor2;

                        if (isShowVariable)
                        {
                            value = rasterPointer->getValueFromRowCol(row+1, col);
                            if (! isEqual(value, rasterPointer->header->flag))
                            {
                                Crit3DColor* variableColor = rasterPointer->colorScale->getColor(value);
                                double alpha = DEFAULT_ALPHA;

                                // check minimum
                                if (rasterPointer->colorScale->isHideMinimum())
                                {
                                    if (isEqual(value, 0) || value <= rasterPointer->colorScale->minimum())
                                        alpha = 0;
                                }
                                if (rasterPointer->colorScale->isTransparent())
                                {
                                    double step = std::max(0., value - rasterPointer->colorScale->minimum());
                                    alpha = std::min(1., step/variableRange);
                                    alpha = pow(alpha, ALPHA_POW);
                                }
                                mixColor(dtmColor2, *variableColor, color2, alpha);
                            }
                        }

                        openGlGeometry->setVertexColor(i++, color1);
                        openGlGeometry->setVertexColor(i++, color2);
                        openGlGeometry->setVertexColor(i++, color3);
                    }

                    z2 = DEM.getValueFromRowCol(row, col+1);
                    if (! isEqual(z2, DEM.header->flag))
                    {
                        Crit3DColor* c2 = DEM.colorScale->getColor(z2);
                        shadowColor(*c2, dtmColor2, row, col+1);
                        color2 = dtmColor2;

                        if (isShowVariable)
                        {
                            value = rasterPointer->getValueFromRowCol(row, col+1);
                            if (! isEqual(value, rasterPointer->header->flag))
                            {
                                Crit3DColor* variableColor = rasterPointer->colorScale->getColor(value);
                                double alpha = DEFAULT_ALPHA;

                                if (rasterPointer->colorScale->isHideMinimum())
                                {
                                    if (isEqual(value, 0) || value <= rasterPointer->colorScale->minimum())
                                        alpha = 0;
                                }

                                if (rasterPointer->colorScale->isTransparent())
                                {
                                    double step = std::max(0., value - rasterPointer->colorScale->minimum());
                                    alpha = std::min(1., step/variableRange);
                                    alpha = pow(alpha, ALPHA_POW);
                                }
                                mixColor(dtmColor2, *variableColor, color2, alpha);
                            }
                        }

                        openGlGeometry->setVertexColor(i++, color3);
                        openGlGeometry->setVertexColor(i++, color2);
                        openGlGeometry->setVertexColor(i++, color1);
                    }
                }
            }
        }
    }

    return true;
}


// -------------------------  SHELL -------------------------

int Crit3DProject::criteria3DShell()
{
    #ifdef _WIN32
        openNewConsole();
    #endif

    printCriteria3DVersion();

        while (! isRequestedExit())
    {
        QString commandLine = getCommandLine("CRITERIA3D");
        if (commandLine != "")
        {
            QList<QString> argumentList = getArgumentList(commandLine);
            if (argumentList.size() > 0)
            {
                int result = executeCommand(argumentList);
                if (result != CRIT3D_OK)
                {
                    logError();
                }
            }
        }
    }

    #ifdef _WIN32
        closeConsole();
    #endif

    return CRIT3D_OK;
}


int Crit3DProject::criteria3DBatch(const QString &scriptFileName)
{
    #ifdef _WIN32
        attachOutputToConsole();
    #endif

    printCriteria3DVersion();

    int result = executeScript(scriptFileName);
    if (result != CRIT3D_OK)
    {
        logError();
    }

    logInfo("Batch finished at: " + QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss"));

    #ifdef _WIN32
        // Send "enter" to release application from the console
        // This is a hack, but if not used the console doesn't know the application has
        // returned. The "enter" key only sent if the console window is in focus.
        if (isConsoleForeground())
            sendEnterKey();
    #endif

    return result;
}


int Crit3DProject::executeScript(const QString &scriptFileName)
{
    if (scriptFileName.isEmpty())
    {
        errorString = "No script file provided.";
        return CRIT3D_ERROR;
    }
    logInfo("Execute script: " + scriptFileName + "\n");

    QFile scriptFile(scriptFileName);
    if(!scriptFile.open(QIODevice::ReadOnly))
    {
        errorString = "Error in opening: " + scriptFileName + " " + scriptFile.errorString();
        return CRIT3D_ERROR;
    }

    while (!scriptFile.atEnd())
    {
        QString cmdLine = scriptFile.readLine();
        QList<QString> argumentList = getArgumentList(cmdLine);

        int result = executeCommand(argumentList);
        if (result != CRIT3D_OK)
            return result;

        logInfo("");
    }
    scriptFile.close();

    return CRIT3D_OK;
}


int Crit3DProject::executeCommand(const QList<QString> &argumentList)
{
    if (argumentList.size() == 0)
        return CRIT3D_OK;

    logInfo(getTimeStamp(argumentList));

    // comment
    if (argumentList[0].at(0) == '#')
        return CRIT3D_OK;

    bool isCommandFound;

    int result = executeSharedCommand(this, argumentList, &isCommandFound);
    if (isCommandFound)
        return result;

    result = executeCriteria3DCommand(argumentList, isCommandFound);
    if (isCommandFound)
        return result;

    errorString = "This is not a valid CRITERIA3D command: " + argumentList[0];
    return CRIT3D_INVALID_COMMAND;
}


int Crit3DProject::printCriteria3DCommandList()
{
    //QList<QString> list = getSharedCommandList();
    QList<QString> list;

    // criteria3D commands
    list.append("?               | ListCommands");
    list.append("Version         | Criteria3DVersion");
    list.append("Ls              | List");
    list.append("Proj            | OpenProject");
    list.append("State           | LoadState");
    list.append("Thread          | SetThreadNr");
    list.append("Run             | RunModels");
    list.append("Save            | SaveState");
    list.append("Quit            | Exit");

    std::cout << "Available Console commands:" << std::endl;
    std::cout << "(short          | long version)" << std::endl << std::endl;
    for (int i = 0; i < list.size(); i++)
    {
        std::cout << list[i].toStdString() << std::endl;
    }
    std::cout << std::endl;

    return CRIT3D_OK;
}


int Crit3DProject::executeCriteria3DCommand(const QList<QString> &argumentList, bool &isCommandFound)
{
    isCommandFound = false;
    if (argumentList.empty())
        return CRIT3D_INVALID_COMMAND;

    QString command = argumentList[0].toUpper();

    if (command == "?" || command == "LISTCOMMANDS")
    {
        isCommandFound = true;
        return printCriteria3DCommandList();
    }
    if (command == "VERSION" || command == "CRITERIA3DVERSION")
    {
        isCommandFound = true;
        return printCriteria3DVersion();
    }
    else if (command == "PROJ" || command == "OPENPROJECT")
    {
        isCommandFound = true;
        return cmdOpenCriteria3DProject(argumentList);
    }
    else if (command == "LS" || command == "LIST")
    {
        isCommandFound = true;
        return cmdList(argumentList);
    }
    else if (command == "STATE" || command == "LOADSTATE")
    {
        isCommandFound = true;
        return cmdLoadState(argumentList);
    }
    else if (command == "SAVE" || command == "SAVESTATE")
    {
        isCommandFound = true;
        return cmdSaveCurrentState();
    }
    else if (command == "RUN" || command == "RUNMODELS")
    {
        isCommandFound = true;
        return cmdRunModels(argumentList);
    }
    else if (command == "THREAD" || command == "SETTHREADNR")
    {
        isCommandFound = true;
        return cmdSetThreadsNr(argumentList);
    }

    return CRIT3D_INVALID_COMMAND;
}


int Crit3DProject::printCriteria3DVersion()
{
    std::cout << "CRITERIA3D " << CRITERIA3D_VERSION << std::endl;
    std::cout << std::endl;

    return CRIT3D_OK;
}


int Crit3DProject::cmdOpenCriteria3DProject(const QList<QString> &argumentList)
{
    if (argumentList.size() < 2)
    {
        errorString = "Missing project file name";
        return CRIT3D_INVALID_COMMAND;
    }

    // set fileName and projectFolder
    QString filename = argumentList.at(1);
    QString filePath = getFilePath(filename);

    QString projectFolder = "";
    if (filePath.isEmpty())
    {
        if (filename.left(filename.length()-4) == ".ini")
            projectFolder = filename.left(filename.length()-4) + "/";
        else
        {
            projectFolder = filename + "/";
            filename += ".ini";
        }
    }

    projectFolder = PATH_PROJECT + projectFolder;
    QString projectName = getCompleteFileName(filename, projectFolder);

    if (! loadCriteria3DProject(projectName))
        return CRIT3D_ERROR;

    return CRIT3D_OK;
}


int Crit3DProject::cmdList(const QList<QString> &argumentList)
{
    QString typeStr;
    if (argumentList.size() >= 2)
    {
        typeStr = argumentList.at(1);
    }

    if (typeStr != "proj" && typeStr != "projects" && typeStr != "states")
    {
        std::cout << "Usage: list [type]" << std::endl;
        std::cout << "type:" << std::endl;
        std::cout << "projects" << std::endl;
        std::cout << "states" << std::endl;
        std::cout << std::endl;
        return CRIT3D_OK;
    }

    QList<QString> list;
    if (typeStr == "proj" || typeStr == "projects")
    {
        if (! getProjectList(list))
            return CRIT3D_ERROR;

        std::cout << "Available projects:" << std::endl;
    }
    else if (typeStr == "states")
    {
        if (getProjectName() == "default")
        {
            errorString = "Open a Project before.";
            return CRIT3D_ERROR;
        }

        if (! getAllSavedState(list))
            return CRIT3D_ERROR;

        std::cout << "Available states:" << std::endl;
    }

    for (int i = 0; i < list.size(); i++)
    {
        std::cout << list[i].toStdString() << std::endl;
    }
    std::cout << std::endl;

    return CRIT3D_OK;
}


int Crit3DProject::cmdLoadState(const QList<QString> &argumentList)
{
    if (argumentList.size() < 2)
    {
        std::cout << "Usage: LoadState <YYYYMMDD_Hhh> | [LAST]" << std::endl;
        std::cout << "Example: LoadState 20230516_H23" << std::endl;
        return CRIT3D_OK;
    }

    QString stateStr;
    if (argumentList.at(1).toUpper() == "LAST")
    {
        QList<QString> list;
        if (! getAllSavedState(list))
            return CRIT3D_ERROR;

        stateStr = list.last();
    }
    else
    {
        stateStr = argumentList.at(1);
    }

    QString statePath = getProjectPath() + PATH_STATES + stateStr;
    if (! loadModelState(statePath))
    {
        std::cout << "Usage: LoadState <YYYYMMDD-Hhh> | [LAST]" << std::endl;
        return CRIT3D_ERROR;
    }

    return CRIT3D_OK;
}


int Crit3DProject::cmdSaveCurrentState()
{
    QString dirName;
    if (! saveModelsState(dirName))
        return CRIT3D_ERROR;

    std::cout << "State successfully saved: " << dirName.toStdString() << std::endl;

    return CRIT3D_OK;
}



void usage_cmdRunModels()
{
    std::cout << "Usage: RunModels <YYYY-MM-DD> <HH> | [ONE_DAY] | [ONE_WEEK] | [ONE_MONTH]" << std::endl;
}

int Crit3DProject::cmdRunModels(const QList<QString> &argumentList)
{
    if (argumentList.size() < 2)
    {
        usage_cmdRunModels();
        return CRIT3D_OK;
    }

    // first time: next hour
    QDateTime firstTime = getCurrentTime();
    QDateTime lastTime = getCurrentTime();;
    firstTime = firstTime.addSecs(HOUR_SECONDS);

    if (argumentList.at(1).toUpper() == "ONE_DAY")
    {
        lastTime = lastTime.addDays(1);
    }
    else if (argumentList.at(1).toUpper() == "ONE_WEEK")
    {
        lastTime = lastTime.addDays(7);
    }
    else if (argumentList.at(1).toUpper() == "ONE_MONTH")
    {
        lastTime = lastTime.addMonths(1);
    }
    else
    {
        QString dateStr, hourStr;
        if (argumentList.size() >= 3)
        {
            dateStr = argumentList.at(1);
            hourStr = argumentList.at(2);
        }

        int lastHour = hourStr.toInt();
        if ((lastHour < 0) || (lastHour > 23))
        {
            std::cout << "Wrong hour! 00-23 are allowed." << std::endl;
            usage_cmdRunModels();
            return CRIT3D_OK;
        }

        lastTime.setDate(QDate::fromString(dateStr, "yyyy-MM-dd"));
        lastTime.setTime(QTime(lastHour, 0, 0, 0));
        if (! lastTime.isValid())
        {
            std::cout << "Wrong time parameters!" << std::endl;
            usage_cmdRunModels();
            return CRIT3D_OK;
        }
    }

    if (! startModels(firstTime, lastTime))
        return CRIT3D_ERROR;

    return CRIT3D_OK;
}


// zero or negative number to set maximum available
int Crit3DProject::cmdSetThreadsNr(const QList<QString> &argumentList)
{
    int nr = 0;
    if (argumentList.size() >= 2)
		nr = argumentList.at(1).toInt();

    auto threadNr = soilFluxes3D::setThreadsNumber(nr);
	
    waterFluxesParameters.numberOfThreads = threadNr;
    std::cout << "Number of threads: " << threadNr << std::endl;
    std::cout << std::endl;

    return CRIT3D_OK;
}

