#include <string>
#include "meteo.h"
#include "utilities.h"
#include "commonConstants.h"
#include "dataHandler.h"


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



meteoVariable getMeteoVariable(int myVar)
{
    if (myVar == 14) return(airTemperature);
    else if (myVar == 15) return(precipitation);
    else if (myVar == 16) return(airRelHumidity);
    else if (myVar == 17) return(globalIrradiance);
    else if (myVar == 18) return(windScalarIntensity);
    else if (myVar == 20) return(leafWetness);
    else if (myVar == 21) return(atmPressure);
    else if (myVar == 43) return(windVectorDirection);
    else return(noMeteoVar);
}

int getMeteoVarIndex(meteoVariable myVar)
{
    if (myVar == airTemperature) return 14;
    else if (myVar == precipitation)return 15;
    else if (myVar == airRelHumidity) return 16;
    else if (myVar == globalIrradiance) return 17;
    else if (myVar == windScalarIntensity) return 18;
    else if (myVar == leafWetness) return 20;
    else if (myVar == atmPressure) return 21;
    else if (myVar == windVectorDirection) return 43;
    else
        return NODATA;
}

float getTimeStepFromHourlyInterval(int myHourlyIntervals)
{    return 3600. / ((float)myHourlyIntervals);}


