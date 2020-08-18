#include <stdio.h>
#include <math.h>
#include <qstring.h>
#include <QDate>

#include "vine3DProject.h"
#include "waterBalance.h"
#include "crit3dDate.h"
#include "meteo.h"
#include "dataHandler.h"
#include "solarRadiation.h"
#include "grapevine.h"
#include "atmosphere.h"
#include "utilities.h"
#include <QDebug>

extern Vine3DProject myProject;

// [Pa] default atmospheric pressure at sea level
#define PRESS 101325

bool setSoilProfileCrop(Vine3DProject* myProject, int row, int col, Crit3DModelCase* modelCase)
{
    double* soilWPProfile = getSoilVarProfile(myProject, row, col, soil::soilWaterPotentialWP);
    double* soilFCProfile = getSoilVarProfile(myProject, row, col, soil::soilWaterPotentialFC) ;
    double* matricPotentialProfile = getCriteria3DVarProfile(myProject, row, col, waterMatricPotential);
    double* waterContentProfile = getCriteria3DVarProfile(myProject, row, col, waterContent);
    double* waterContentProfileWP = getSoilVarProfile(myProject, row, col, soil::soilWaterContentWP);
    double* waterContentProfileFC = getSoilVarProfile(myProject, row, col, soil::soilWaterContentFC);

    if (! myProject->grapevine.setSoilProfile(modelCase, soilWPProfile, soilFCProfile,
            matricPotentialProfile, waterContentProfile, waterContentProfileFC,
            waterContentProfileWP)) return false;

    free(soilWPProfile);
    free(soilFCProfile);
    free(matricPotentialProfile);
    free(waterContentProfile);
    free(waterContentProfileWP);
    free(waterContentProfileFC);

    return true;
}


bool assignIrrigation(Vine3DProject* myProject, Crit3DTime myTime)
{
    int fieldIndex;
    float nrHours;
    float irrigationRate, rate;
    int hour = myTime.getHour();
    int idBook;
    QDate myDate = getQDate(myTime.date);
    int row, col;

    for (row = 0; row < myProject->DEM.header->nrRows ; row++)
        for (col = 0; col < myProject->DEM.header->nrCols; col++)
            if (int(myProject->DEM.value[row][col]) != int(myProject->DEM.header->flag))
            {
                //initialize
                myProject->vine3DMapsH->mapHourlyIrrigation->value[row][col] = 0.0;

                fieldIndex = myProject->getModelCaseIndex(unsigned(row), unsigned(col));
                if (fieldIndex > 0)
                {
                    idBook = 0;
                    while (myProject->getFieldBookIndex(idBook, myDate, fieldIndex, &idBook))
                    {
                        if (myProject->fieldBook[idBook].operation == irrigationOperation)
                        {
                            nrHours = myProject->fieldBook[idBook].quantity;
                            if (hour >= (24-nrHours))
                            {
                                irrigationRate = myProject->modelCases[fieldIndex].maxIrrigationRate;
                                rate = irrigationRate / myProject->meteoSettings->getHourlyIntervals();
                                myProject->vine3DMapsH->mapHourlyIrrigation->value[row][col] = rate;
                            }
                        }
                        idBook++;
                    }
                }
            }
    return true;
}


QString grapevineError(Crit3DTime myTime, long row, long col, QString errorIni)
{
    QString myString = "Error computing grapevine for DEM cell (" + QString::number(row) + "," + QString::number(col) + ")\n";
    myString += errorIni + "\n";
    myString += QString::fromStdString(myTime.date.toStdString()) + " " + QString("%1").arg(myTime.getHour(), 2, 10, QChar('0')) + ":00\n";
    return myString;
}


bool modelDailyCycle(bool isInitialState, Crit3DDate myDate, int nrHours,
                     Vine3DProject* myProject, const QString& myOutputPath,
                     bool saveOutput, const QString& myArea)
{

    TfieldOperation operation;
    float quantity;
    QDate myQDate = getQDate(myDate);
    Crit3DTime myCurrentTime, myFirstTime, myLastTime;
    int myTimeStep = int(myProject->getTimeStep());
    myFirstTime = Crit3DTime(myDate, myTimeStep);
    myLastTime = Crit3DTime(myDate, nrHours * 3600);
    int nrStep = (nrHours * 3600) / myTimeStep;
    bool isNewModelCase;
    int modelCaseIndex;
    double* myProfile;

    for (myCurrentTime = myFirstTime; myCurrentTime <= myLastTime; myCurrentTime = myCurrentTime.addSeconds(myTimeStep))
    {
        myProject->logInfo("\n" + getQDateTime(myCurrentTime).toString("yyyy-MM-dd hh:mm"));
        myProject->grapevine.setDate(myCurrentTime);

        // meteo interpolation
        myProject->logInfo("Interpolate meteo data");
        interpolateAndSaveHourlyMeteo(myProject, airTemperature, myCurrentTime, myOutputPath, saveOutput, myArea);
        interpolateAndSaveHourlyMeteo(myProject, precipitation, myCurrentTime, myOutputPath, saveOutput, myArea);
        interpolateAndSaveHourlyMeteo(myProject, airRelHumidity, myCurrentTime, myOutputPath, saveOutput, myArea);
        interpolateAndSaveHourlyMeteo(myProject, windScalarIntensity, myCurrentTime, myOutputPath, saveOutput, myArea);
        interpolateAndSaveHourlyMeteo(myProject, globalIrradiance, myCurrentTime, myOutputPath, saveOutput, myArea);

        // ET0
        myProject->hourlyMeteoMaps->computeET0PMMap(myProject->DEM, myProject->radiationMaps);
        if (saveOutput)
        {
            myProject->saveHourlyMeteoOutput(referenceEvapotranspiration, myOutputPath, getQDateTime(myCurrentTime), myArea);
        }

        // Leaf Wetness
        myProject->hourlyMeteoMaps->computeLeafWetnessMap();
        if (saveOutput)
        {
            myProject->saveHourlyMeteoOutput(leafWetness, myOutputPath, getQDateTime(myCurrentTime), myArea);
        }

        if (isInitialState)
        {
            myProject->initializeSoilMoisture(myCurrentTime.date.month);
        }

        //Grapevine
        double vineTranspiration, grassTranspiration;
        myProject->logInfo("Compute grapevine");
        for (long row = 0; row < myProject->DEM.header->nrRows ; row++)
        {
            for (long col = 0; col < myProject->DEM.header->nrCols; col++)
            {
                if (int(myProject->DEM.value[row][col]) != int(myProject->DEM.header->flag))
                {
                    modelCaseIndex = myProject->getModelCaseIndex(unsigned(row), unsigned(col));
                    isNewModelCase = (int(myProject->statePlantMaps->fruitBiomassMap->value[row][col])
                                  == int(myProject->statePlantMaps->fruitBiomassMap->header->flag));

                    if (! myProject->grapevine.setWeather(
                                double(myProject->vine3DMapsD->mapDailyTAvg->value[row][col]),
                                double(myProject->hourlyMeteoMaps->mapHourlyTair->value[row][col]),
                                double(myProject->radiationMaps->globalRadiationMap->value[row][col]),
                                double(myProject->hourlyMeteoMaps->mapHourlyPrec->value[row][col]),
                                double(myProject->hourlyMeteoMaps->mapHourlyRelHum->value[row][col]),
                                double(myProject->hourlyMeteoMaps->mapHourlyWindScalarInt->value[row][col]),
                                PRESS)) {
                        myProject->errorString = grapevineError(myCurrentTime, row, col, "Weather data missing");
                        return(false);
                    }

                    if (!myProject->grapevine.setDerivedVariables(
                                double(myProject->radiationMaps->diffuseRadiationMap->value[row][col]),
                                double(myProject->radiationMaps->beamRadiationMap->value[row][col]),
                                double(myProject->radiationMaps->transmissivityMap->value[row][col] / CLEAR_SKY_TRANSMISSIVITY_DEFAULT),
                                double(myProject->radiationMaps->sunElevationMap->value[row][col])))
                    {
                        myProject->errorString = grapevineError(myCurrentTime, row, col, "Radiation data missing");
                        return (false);
                    }

                    myProject->grapevine.resetLayers();

                    if (! setSoilProfileCrop(myProject, row, col, &(myProject->modelCases[modelCaseIndex]))) {
                        myProject->errorString = grapevineError(myCurrentTime, row, col, "Error setting soil profile");
                        return false;
                    }

                    if ((isInitialState) || (isNewModelCase))
                    {
                        if(!myProject->grapevine.initializeStatePlant(getDoyFromDate(myDate), &(myProject->modelCases[modelCaseIndex])))
                        {
                            myProject->logInfo("Could not initialize grapevine in the present growing season.\nIt will be replaced by a complete grass cover.");
                        }
                    }
                    else
                    {
                        if (!setStatePlantfromMap(row, col, myProject)) return(false);
                        myProject->grapevine.setStatePlant(myProject->statePlant, true);
                    }
                    double chlorophyll = NODATA;

                    if (! myProject->grapevine.compute((myCurrentTime == myFirstTime), myTimeStep, &(myProject->modelCases[modelCaseIndex]), chlorophyll))
                    {
                        myProject->errorString = grapevineError(myCurrentTime, row, col, "Error in grapevine computation");
                        return(false);
                    }

                    // check field book (first hour)
                    if (myCurrentTime.getHour() == 1)
                    {
                        int idBook = 0;
                        while (myProject->getFieldBookIndex(idBook, myQDate, modelCaseIndex, &idBook))
                        {
                            operation = myProject->fieldBook[idBook].operation;
                            quantity = myProject->fieldBook[idBook].quantity;
                            myProject->grapevine.fieldBookAction(&(myProject->modelCases[modelCaseIndex]), operation, quantity);
                            idBook++;
                        }
                    }

                    myProject->statePlant = myProject->grapevine.getStatePlant();
                    getStatePlantToMap(row, col, myProject, &(myProject->statePlant));

                    myProfile = myProject->grapevine.getExtractedWater(&(myProject->modelCases[modelCaseIndex]));
                    for (int layer=0; layer < myProject->nrLayers; layer++)
                        myProject->outputPlantMaps->transpirationLayerMaps[layer]->value[row][col] = float(myProfile[layer]);

                    vineTranspiration = myProject->grapevine.getRealTranspirationGrapevine(&(myProject->modelCases[modelCaseIndex]));
                    grassTranspiration = myProject->grapevine.getRealTranspirationGrass(&(myProject->modelCases[modelCaseIndex]));

                    if (myCurrentTime == myFirstTime)
                    {
                        myProject->outputPlantMaps->vineyardTranspirationMap->value[row][col] = float(vineTranspiration);
                        myProject->outputPlantMaps->grassTranspirationMap->value[row][col] = float(grassTranspiration);
                        myProject->outputPlantMaps->vineStressMap->value[row][col] = float(myProject->grapevine.getStressCoefficient());
                    }
                    else
                    {
                        // summed values
                        myProject->outputPlantMaps->vineyardTranspirationMap->value[row][col] += float(vineTranspiration);
                        myProject->outputPlantMaps->grassTranspirationMap->value[row][col] += float(grassTranspiration);
                        myProject->outputPlantMaps->vineStressMap->value[row][col] += float(myProject->grapevine.getStressCoefficient());
                    }

                    if (myCurrentTime == myLastTime)
                    {
                        // average values
                        myProject->outputPlantMaps->vineStressMap->value[row][col] /= nrStep;
                    }
                }
            }
        }

        // Irrigation
        assignIrrigation(myProject, myCurrentTime);


        if (! myProject->computeVine3DWaterSinkSource())
            return false;

        // 3D soil water balance
        myProject->computeWaterBalance3D(3600);

        if (myCurrentTime == myFirstTime)
            resetWaterBalanceMap(myProject);

        updateWaterBalanceMaps(myProject);

        if (isInitialState) isInitialState = false;
    }

    return true;
}
