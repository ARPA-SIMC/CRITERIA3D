#include <QDateTime>
#include <QFile>

#include "basicMath.h"
#include "interpolation.h"
#include "plant.h"
#include "vine3DProject.h"
#include "commonConstants.h"


void Crit3DStatePlantMaps::initialize()
{
    leafAreaIndexMap = new gis::Crit3DRasterGrid;

    pHBerryMap = new gis::Crit3DRasterGrid;
    daysAfterBloomMap = new gis::Crit3DRasterGrid;
    cumulatedBiomassMap = new gis::Crit3DRasterGrid;
    fruitBiomassMap = new gis::Crit3DRasterGrid;
    shootLeafNumberMap = new gis::Crit3DRasterGrid;
    meanTemperatureLastMonthMap = new gis::Crit3DRasterGrid;
    chillingStateMap = new gis::Crit3DRasterGrid;
    forceStateBudBurstMap = new gis::Crit3DRasterGrid;
    forceStateVegetativeSeasonMap = new gis::Crit3DRasterGrid;
    stageMap = new gis::Crit3DRasterGrid;
    cumulatedRadiationFromFruitsetToVeraisonMap = new gis::Crit3DRasterGrid;
    degreeDaysFromFirstMarchMap = new gis::Crit3DRasterGrid;
    degreeDay10FromBudBurstMap = new gis::Crit3DRasterGrid;
    degreeDaysAtFruitSetMap = new gis::Crit3DRasterGrid;
    powderyAICMap = new gis::Crit3DRasterGrid;
    powderyCurrentColoniesMap  = new gis::Crit3DRasterGrid;
    powderySporulatingColoniesMap = new gis::Crit3DRasterGrid;
    isHarvestedMap =  new gis::Crit3DRasterGrid;
    fruitBiomassIndexMap =  new gis::Crit3DRasterGrid;

    isLoaded = false;
}

Crit3DStatePlantMaps::Crit3DStatePlantMaps()
{
    this->initialize();
}

Crit3DStatePlantMaps::Crit3DStatePlantMaps(const gis::Crit3DRasterGrid &myDEM)
{
    this->initialize();

    leafAreaIndexMap->initializeGrid(myDEM);

    pHBerryMap->initializeGrid(myDEM);
    daysAfterBloomMap->initializeGrid(myDEM);
    cumulatedBiomassMap->initializeGrid(myDEM);
    fruitBiomassMap->initializeGrid(myDEM);
    shootLeafNumberMap->initializeGrid(myDEM);
    meanTemperatureLastMonthMap->initializeGrid(myDEM);
    chillingStateMap->initializeGrid(myDEM);
    forceStateBudBurstMap->initializeGrid(myDEM);
    forceStateVegetativeSeasonMap->initializeGrid(myDEM);
    stageMap->initializeGrid(myDEM);
    cumulatedRadiationFromFruitsetToVeraisonMap->initializeGrid(myDEM);
    degreeDaysFromFirstMarchMap->initializeGrid(myDEM);
    degreeDay10FromBudBurstMap->initializeGrid(myDEM);
    degreeDaysAtFruitSetMap->initializeGrid(myDEM);
    powderyAICMap->initializeGrid(myDEM);
    powderyCurrentColoniesMap->initializeGrid(myDEM);
    powderySporulatingColoniesMap->initializeGrid(myDEM);
    isHarvestedMap->initializeGrid(myDEM);
    fruitBiomassIndexMap->initializeGrid(myDEM);

    isLoaded = true;
}


gis::Crit3DRasterGrid* Crit3DStatePlantMaps::getMapFromVar(plantVariable myVar)
{
    if (myVar == leafAreaIndexVar)
        return leafAreaIndexMap;
    else if (myVar == fruitBiomassIndexVar)
        return fruitBiomassIndexMap;
    else if (myVar == isHarvestedVar)
        return isHarvestedMap;
    else if (myVar == pHBerryVar)
        return pHBerryMap;
    else if (myVar == daysAfterBloomVar)
        return daysAfterBloomMap;
    else if (myVar == cumulatedBiomassVar)
        return cumulatedBiomassMap;
    else if (myVar == fruitBiomassVar)
        return fruitBiomassMap;
    else if (myVar == shootLeafNumberVar)
        return shootLeafNumberMap;
    else if (myVar == meanTemperatureLastMonthVar)
        return meanTemperatureLastMonthMap;
    else if (myVar == chillingUnitsVar)
        return chillingStateMap;
    else if (myVar == forceStateBudBurstVar)
        return forceStateBudBurstMap;
    else if (myVar == forceStateVegetativeSeasonVar)
        return forceStateVegetativeSeasonMap;
    else if (myVar == stageVar)
        return stageMap;
    else if (myVar == cumRadFruitsetVerVar)
        return cumulatedRadiationFromFruitsetToVeraisonMap;
    else if (myVar == degreeDaysFromFirstMarchVar)
        return degreeDaysFromFirstMarchMap;
    else if (myVar == degreeDays10FromBudBurstVar)
        return degreeDay10FromBudBurstMap;
    else if (myVar == degreeDaysAtFruitSetVar)
        return degreeDaysAtFruitSetMap;
    else if (myVar == powderyAICVar)
        return powderyAICMap;
    else if (myVar == powderyCurrentColoniesVar)
        return powderyCurrentColoniesMap;
    else if (myVar == powderySporulatingColoniesVar)
        return powderySporulatingColoniesMap;
    else
        return nullptr;
}

Crit3DOutputPlantMaps::Crit3DOutputPlantMaps()
{
    this->initialize();
    transpirationLayerMaps = new gis::Crit3DRasterGrid*[0];
}

Crit3DOutputPlantMaps::Crit3DOutputPlantMaps(const gis::Crit3DRasterGrid &myDEM)
{
    this->initializeWithDEM(myDEM);
    transpirationLayerMaps = new gis::Crit3DRasterGrid*[0];
    transpirationLayerMaps[0]->initializeGrid(myDEM);
}

Crit3DOutputPlantMaps::Crit3DOutputPlantMaps(const gis::Crit3DRasterGrid &myDEM, int nrSoilLayers)
{
    this->initializeWithDEM(myDEM);

    transpirationLayerMaps = new gis::Crit3DRasterGrid*[unsigned(nrSoilLayers)];

    for (int layer=0; layer<nrSoilLayers; layer++)
    {
        transpirationLayerMaps[layer] = new gis::Crit3DRasterGrid();
        transpirationLayerMaps[layer]->initializeGrid(myDEM);
    }
}

void Crit3DOutputPlantMaps::initialize()
{
    wineYieldMap = new gis::Crit3DRasterGrid;
    daysFromFloweringMap = new gis::Crit3DRasterGrid;
    brixMaximumMap = new gis::Crit3DRasterGrid;
    brixBerryMap = new gis::Crit3DRasterGrid;
    vineStressMap = new gis::Crit3DRasterGrid;
    vineyardTranspirationMap = new gis::Crit3DRasterGrid;
    grassTranspirationMap = new gis::Crit3DRasterGrid;
    powderyCOLMap = new gis::Crit3DRasterGrid;
    powderyINFRMap = new gis::Crit3DRasterGrid;
    powderyPrimaryInfectionRiskMap = new gis::Crit3DRasterGrid;
    tartaricAcidMap = new gis::Crit3DRasterGrid;
    deltaBrixMap = new gis::Crit3DRasterGrid;
    downyDormantOosporeMap = new gis::Crit3DRasterGrid;
    downyInfectionRateMap = new gis::Crit3DRasterGrid;
    downyOilSpotMap = new gis::Crit3DRasterGrid;
}

void Crit3DOutputPlantMaps::initializeWithDEM(const gis::Crit3DRasterGrid &myDEM)
{
    this->initialize();

    wineYieldMap->initializeGrid(myDEM);
    daysFromFloweringMap->initializeGrid(myDEM);
    brixMaximumMap->initializeGrid(myDEM);
    brixBerryMap->initializeGrid(myDEM);
    deltaBrixMap->initializeGrid(myDEM);
    vineStressMap->initializeGrid(myDEM);
    vineyardTranspirationMap->initializeGrid(myDEM);
    grassTranspirationMap->initializeGrid(myDEM);
    tartaricAcidMap->initializeGrid(myDEM);
    powderyCOLMap->initializeGrid(myDEM);
    powderyINFRMap->initializeGrid(myDEM);
    powderyPrimaryInfectionRiskMap->initializeGrid(myDEM);
    downyDormantOosporeMap->initializeGrid(myDEM);
    downyInfectionRateMap->initializeGrid(myDEM);
    downyOilSpotMap->initializeGrid(myDEM);
}




gis::Crit3DRasterGrid* Crit3DOutputPlantMaps::getMapFromVar(plantVariable myVar)
{
    if (myVar == brixBerryVar)
        return brixBerryMap;
    else if (myVar == wineYieldVar)
        return wineYieldMap;
    else if (myVar == brixMaximumVar)
        return brixMaximumMap;
    else if (myVar == deltaBrixVar)
        return deltaBrixMap;
    else if (myVar == tartaricAcidVar)
        return tartaricAcidMap;
    else if (myVar == daysFromFloweringVar)
        return daysFromFloweringMap;
    else if (myVar == transpirationStressVar)
        return vineStressMap;
    else if (myVar == transpirationVineyardVar)
        return vineyardTranspirationMap;
    else if (myVar == transpirationGrassVar)
        return grassTranspirationMap;
    else if (myVar == powderyCOLVar)
        return powderyCOLMap;
    else if (myVar == powderyINFRVar)
        return powderyINFRMap;
    else if (myVar == powderyPrimaryInfectionRiskVar)
        return powderyPrimaryInfectionRiskMap;
    else if (myVar == downyDormantOosporeVar)
        return downyDormantOosporeMap;
    else if (myVar == downyInfectionRateVar)
        return downyInfectionRateMap;
    else if (myVar == downyOilSpotVar)
        return downyOilSpotMap;

    else
        return nullptr;
}

bool setStatePlantfromMap(long row, long col , Vine3DProject* myProject)
{
    //growth
    myProject->statePlant.stateGrowth.fruitBiomassIndex = double(myProject->statePlantMaps->fruitBiomassIndexMap->value[row][col]);
    myProject->statePlant.stateGrowth.isHarvested = int(myProject->statePlantMaps->isHarvestedMap->value[row][col]);
    myProject->statePlant.stateGrowth.fruitBiomass = double(myProject->statePlantMaps->fruitBiomassMap->value[row][col]);
    myProject->statePlant.stateGrowth.cumulatedBiomass = double(myProject->statePlantMaps->cumulatedBiomassMap->value[row][col]);
    myProject->statePlant.stateGrowth.meanTemperatureLastMonth = double(myProject->statePlantMaps->meanTemperatureLastMonthMap->value[row][col]);
    myProject->statePlant.stateGrowth.shootLeafNumber = double(myProject->statePlantMaps->shootLeafNumberMap->value[row][col]);
    //pheno
    myProject->statePlant.statePheno.chillingState = double(myProject->statePlantMaps->chillingStateMap->value[row][col]);
    myProject->statePlant.statePheno.forceStateBudBurst = double(myProject->statePlantMaps->forceStateBudBurstMap->value[row][col]);
    myProject->statePlant.statePheno.forceStateVegetativeSeason = double(myProject->statePlantMaps->forceStateVegetativeSeasonMap->value[row][col]);
    myProject->statePlant.statePheno.degreeDaysFromFirstMarch = double(myProject->statePlantMaps->degreeDaysFromFirstMarchMap->value[row][col]);

    if (isEqual(myProject->statePlantMaps->degreeDaysAtFruitSetMap->value[row][col], myProject->statePlantMaps->degreeDaysAtFruitSetMap->header->flag))
        myProject->statePlant.statePheno.degreeDaysAtFruitSet = NODATA;
    else
        myProject->statePlant.statePheno.degreeDaysAtFruitSet = double(myProject->statePlantMaps->degreeDaysAtFruitSetMap->value[row][col]);

    myProject->statePlant.statePheno.daysAfterBloom = double(myProject->statePlantMaps->daysAfterBloomMap->value[row][col]);
    myProject->statePlant.statePheno.stage = double(myProject->statePlantMaps->stageMap->value[row][col]);
    myProject->statePlant.statePheno.cumulatedRadiationFromFruitsetToVeraison = double(myProject->statePlantMaps->cumulatedRadiationFromFruitsetToVeraisonMap->value[row][col]);

    myProject->statePlant.stateGrowth.leafAreaIndex = double(myProject->statePlantMaps->leafAreaIndexMap->value[row][col]);
    return true;
}

bool getStatePlantToMap(long row,long col, Vine3DProject* myProject, TstatePlant* statePlant)
{
    myProject->statePlantMaps->fruitBiomassIndexMap->value[row][col] = float(statePlant->stateGrowth.fruitBiomassIndex);
    myProject->statePlantMaps->isHarvestedMap->value[row][col] = float(statePlant->stateGrowth.isHarvested);
    myProject->statePlantMaps->fruitBiomassMap->value[row][col] = float(statePlant->stateGrowth.fruitBiomass);
    myProject->statePlantMaps->cumulatedBiomassMap->value[row][col] = float(statePlant->stateGrowth.cumulatedBiomass);

    myProject->statePlantMaps->meanTemperatureLastMonthMap->value[row][col] = float(statePlant->stateGrowth.meanTemperatureLastMonth);
    myProject->statePlantMaps->shootLeafNumberMap->value[row][col] = float(statePlant->stateGrowth.shootLeafNumber);
    //pheno
    myProject->statePlantMaps->chillingStateMap->value[row][col] = float(statePlant->statePheno.chillingState);
    myProject->statePlantMaps->forceStateBudBurstMap->value[row][col] = float(statePlant->statePheno.forceStateBudBurst);
    myProject->statePlantMaps->forceStateVegetativeSeasonMap->value[row][col] = float(statePlant->statePheno.forceStateVegetativeSeason);
    myProject->statePlantMaps->daysAfterBloomMap->value[row][col] = float(statePlant->statePheno.daysAfterBloom);
    myProject->statePlantMaps->stageMap->value[row][col] = float(statePlant->statePheno.stage);

    myProject->statePlantMaps->leafAreaIndexMap->value[row][col] = float(myProject->statePlant.stateGrowth.leafAreaIndex);
    myProject->statePlantMaps->cumulatedRadiationFromFruitsetToVeraisonMap->value[row][col] = float(myProject->statePlant.statePheno.cumulatedRadiationFromFruitsetToVeraison);

    myProject->outputPlantMaps->tartaricAcidMap->value[row][col] = float(statePlant->stateGrowth.tartaricAcid);
    myProject->outputPlantMaps->brixBerryMap->value[row][col] = float(myProject->statePlant.outputPlant.brixBerry);
    myProject->outputPlantMaps->brixMaximumMap->value[row][col] = float(myProject->statePlant.outputPlant.brixMaximum);
    myProject->outputPlantMaps->daysFromFloweringMap->value[row][col] = float(myProject->statePlant.statePheno.daysAfterBloom);
    if (myProject->statePlantMaps->stageMap->value[row][col] >= veraison)
        myProject->outputPlantMaps->wineYieldMap->value[row][col] = float(myProject->statePlantMaps->fruitBiomassMap->value[row][col] * 240.0);
    else
        myProject->outputPlantMaps->wineYieldMap->value[row][col] = float(myProject->outputPlantMaps->wineYieldMap->header->flag);

    //delta brix
    myProject->outputPlantMaps->deltaBrixMap->value[row][col] = float(myProject->outputPlantMaps->deltaBrixMap->header->flag);
    if (! isEqual(myProject->outputPlantMaps->brixMaximumMap->value[row][col], myProject->outputPlantMaps->brixMaximumMap->header->flag) &&
        ! isEqual(myProject->outputPlantMaps->brixBerryMap->value[row][col], myProject->outputPlantMaps->brixBerryMap->header->flag))
        myProject->outputPlantMaps->deltaBrixMap->value[row][col] = MAXVALUE(myProject->outputPlantMaps->brixMaximumMap->value[row][col] - myProject->outputPlantMaps->brixBerryMap->value[row][col],0);

    return(true);
}

bool savePlantState(Vine3DProject* myProject, plantVariable myVar, QDate myDate, QString myPath)
{
    QString varName = getVarNameFromPlantVariable(myVar);
    QString fileName = myPath + myDate.toString("yyyyMMdd_") + "_" + varName;
    std::string errorStr;

    gis::Crit3DRasterGrid* myMap;
    myMap = myProject->statePlantMaps->getMapFromVar(myVar);

    if (! gis::writeEsriGrid(fileName.toStdString(), myMap, errorStr))
    {
        myProject->logError(QString::fromStdString(errorStr));
        return false;
    }

    return true;
}


bool savePlantOutput(Vine3DProject* myProject, plantVariable myVar,
                   QDate myDate, QString myPath, QString notes, bool isStateMap, bool isMasked)
{
    QString varName = getVarNameFromPlantVariable(myVar);
    QString fileName = getOutputNameDaily(varName, notes, myDate);
    QString outputFileName = myPath + fileName;

    gis::Crit3DRasterGrid *myMap;
    gis::Crit3DRasterGrid outputMap;

    if (isStateMap)
        myMap = myProject->statePlantMaps->getMapFromVar(myVar);
    else
        myMap = myProject->outputPlantMaps->getMapFromVar(myVar);

    outputMap.initializeGrid(*myMap);

    for (long row = 0; row < myMap->header->nrRows ; row++)
        for (long col = 0; col < myMap->header->nrCols; col++)
        {
            outputMap.value[row][col] = myMap->value[row][col];
            if (isMasked)
            {
                if ((! myProject->isVineyard(row,col))
                    || (myProject->statePlantMaps->stageMap->value[row][col] == NOT_INITIALIZED_VINE))
                    outputMap.value[row][col] = outputMap.header->flag;
            }

        }

    std::string errorStr;
    if (! gis::writeEsriGrid(outputFileName.toStdString(), &outputMap, errorStr))
    {
        myProject->logError(QString::fromStdString(errorStr));
        return false;
    }

    outputMap.clear();

    return true;
}


bool loadPlantState(Vine3DProject* myProject, plantVariable myVar, QDate myDate, QString myPath)
{
    QString varName = getVarNameFromPlantVariable(myVar);
    QString fileName = myPath + myDate.toString("yyyyMMdd_") + "_" + varName;

    std::string errorString;
    gis::Crit3DRasterGrid* myMap = myProject->statePlantMaps->getMapFromVar(myVar);

    QFile myFile;
    myFile.setFileName(fileName + ".hdr");
    if (! myFile.exists()) return false;

    if (! gis::readEsriGrid(fileName.toStdString(), myMap, errorString))
    {
        myProject->logError(QString::fromStdString(errorString));
        return false;
    }

    return true;
}


bool updateThermalSum(Vine3DProject* myProject, QDate myDate)
{
    float phenoPhase, airTemp, degreeDays, flag;
    bool  isFruitSet;
    QDate firstMarch;
    QDate fifteenNovember;

    myProject->logInfo("\nUpdate thermal sum...");
    firstMarch.setDate(myDate.year(),3,1);
    fifteenNovember.setDate(myDate.year(),11,15);
    flag = myProject->statePlantMaps->degreeDaysFromFirstMarchMap->header->flag;

    for (long row = 0; row < myProject->DEM.header->nrRows ; row++)
    {
        for (long col = 0; col < myProject->DEM.header->nrCols; col++)
        {
            if (myProject->DEM.value[row][col] != myProject->DEM.header->flag)
            {
                phenoPhase = myProject->statePlantMaps->stageMap->value[row][col];
                isFruitSet = (phenoPhase >= fruitSet);

                airTemp = myProject->vine3DMapsD->mapDailyTAvg->value[row][col];
                degreeDays = myProject->statePlantMaps->degreeDaysFromFirstMarchMap->value[row][col];

                // thermal sum at fruit set
                if ((isFruitSet) && (myProject->statePlantMaps->degreeDaysAtFruitSetMap->value[row][col] == flag))
                    myProject->statePlantMaps->degreeDaysAtFruitSetMap->value[row][col] = degreeDays;

                if (myDate >= firstMarch)
                {
                    if (degreeDays == flag)
                        degreeDays = MAXVALUE(0.0, airTemp);
                    else
                        degreeDays += MAXVALUE(0.0, airTemp);
                }
                else
                    degreeDays = flag;


                myProject->statePlantMaps->degreeDaysFromFirstMarchMap->value[row][col] = degreeDays;
                if (!isFruitSet)
                {
                    myProject->statePlantMaps->degreeDaysAtFruitSetMap->value[row][col] = NODATA;
                    if ((myDate < firstMarch)||(myDate >= fifteenNovember))
                    {
                        myProject->statePlantMaps->degreeDaysFromFirstMarchMap->value[row][col] = NODATA;
                    }
                }
            }
        }
    }

    return true;
}


QString getVarNameFromPlantVariable(plantVariable myVar)
{
    if (myVar == tartaricAcidVar)
        return "tartaricAcid";
    else if (myVar == wineYieldVar)
        return "wineYield";
    else if (myVar == pHBerryVar)
        return "pHBerry";
    else if (myVar == brixBerryVar)
        return "brixBerry";
    else if (myVar == brixMaximumVar)
        return "brixMaximum";
    else if (myVar == deltaBrixVar)
        return "deltabrix";
    else if (myVar == daysAfterBloomVar)
        return "daysAfterBloom";
    else if (myVar == cumulatedBiomassVar)
        return "totalBiomass";
    else if (myVar == daysFromFloweringVar)
        return "daysFromFlowering";
    else if (myVar == isHarvestedVar)
        return "isHarvested";
    else if (myVar == fruitBiomassVar)
        return "fruitBiomass";
    else if (myVar == shootLeafNumberVar)
        return "shootLeafNumber";
    else if (myVar == meanTemperatureLastMonthVar)
        return "meanTLastMonth";
    else if (myVar == chillingUnitsVar)
        return "chillingUnits";
    else if (myVar == forceStateBudBurstVar)
        return "forceStBudBurst";
    else if (myVar == forceStateVegetativeSeasonVar)
        return "forceStVegSeason";
    else if (myVar == stageVar)
        return "phenoPhase";
    else if (myVar == cumRadFruitsetVerVar)
        return "cumRadFSVeraison";
    else if (myVar == leafAreaIndexVar)
        return "leafAreaIndex";
    else if (myVar == transpirationStressVar)
        return "vineStress";
    else if (myVar == transpirationVineyardVar)
        return "transpirationVine";
    else if (myVar == transpirationGrassVar)
        return "transpirationGrass";
    else if (myVar == degreeDaysFromFirstMarchVar)
        return "degreeDaysFromFirstMarch";
    else if (myVar == degreeDays10FromBudBurstVar)
        return "degreeDaysFromBudBurst";
    else if (myVar == degreeDaysAtFruitSetVar)
        return "degreeDaysAtFruitSet";
    else if (myVar == powderyCurrentColoniesVar)
        return "powderyCurrentColonies";
    else if (myVar == powderyAICVar)
        return "powderyAIC";
    else if (myVar == powderyCOLVar)
        return "powderyCOL";
    else if (myVar == powderyINFRVar)
        return "powderyINFR";
    else if (myVar == powderySporulatingColoniesVar)
        return "powderyTSCOL";
    else if (myVar == powderyPrimaryInfectionRiskVar)
        return "powderyPIR";
    else if (myVar == fruitBiomassIndexVar)
        return "fruitBiomassIndex";
    else
        return "";
}

