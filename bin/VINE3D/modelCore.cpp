#include <stdio.h>
#include <math.h>
#include <qstring.h>
#include <QDate>

#include "commonConstants.h"
#include "vine3DProject.h"
#include "waterBalance.h"
#include "crit3dDate.h"
#include "meteo.h"
#include "solarRadiation.h"
#include "grapevine.h"
#include "utilities.h"


extern Vine3DProject myProject;


QString grapevineError(Crit3DTime myTime, long row, long col, QString errorIni)
{
    QString myString = "Error computing grapevine for DEM cell (" + QString::number(row) + "," + QString::number(col) + ")\n";
    myString += errorIni + "\n";
    myString += QString::fromStdString(myTime.date.toStdString()) + " " + QString("%1").arg(myTime.getHour(), 2, 10, QChar('0')) + ":00\n";
    return myString;
}


bool Vine3DProject::setSoilProfileCrop(int row, int col, Crit3DModelCase* modelCase)
{
    std::vector<double> soilWPProfile = getSoilVarProfile(this, row, col, soil::soilWaterPotentialWP);
    std::vector<double> soilFCProfile = getSoilVarProfile(this, row, col, soil::soilWaterPotentialFC) ;
    std::vector<double> waterContentProfileWP = getSoilVarProfile(this, row, col, soil::soilWaterContentWP);
    std::vector<double> waterContentProfileFC = getSoilVarProfile(this, row, col, soil::soilWaterContentFC);

    std::vector<double> matricPotentialProfile = getCriteria3DVarProfile(this, row, col, waterMatricPotential);
    std::vector<double> waterContentProfile = getCriteria3DVarProfile(this, row, col, volumetricWaterContent);

    return grapevine.setSoilProfile(modelCase, soilWPProfile, soilFCProfile,
                                    matricPotentialProfile, waterContentProfile, waterContentProfileFC, waterContentProfileWP);
}


bool Vine3DProject::assignIrrigation(Crit3DTime myTime)
{
    float nrHours;
    float irrigationRate, rate;
    int hour = myTime.getHour();
    int idBook;
    QDate myDate = getQDate(myTime.date);
    int row, col;

    for (row = 0; row < DEM.header->nrRows ; row++)
    {
        for (col = 0; col < DEM.header->nrCols; col++)
        {
            if (int(DEM.value[row][col]) != int(DEM.header->flag))
            {
                //initialize
                vine3DMapsH->mapHourlyIrrigation->value[row][col] = 0.0;

                int caseIndex = getModelCaseIndex(row, col);
                if (caseIndex != NODATA)
                {
                    idBook = 0;
                    int fieldIndex = modelCases[caseIndex].id;
                    while (getFieldBookIndex(idBook, myDate, fieldIndex, &idBook))
                    {
                        if (fieldBook[idBook].operation == irrigationOperation)
                        {
                            nrHours = fieldBook[idBook].quantity;
                            if (hour >= (24-nrHours))
                            {
                                irrigationRate = modelCases[fieldIndex].maxIrrigationRate;
                                rate = irrigationRate / meteoSettings->getHourlyIntervals();
                                vine3DMapsH->mapHourlyIrrigation->value[row][col] = rate;
                            }
                        }

                        idBook++;
                    }
                }
            }
        }
    }

    return true;
}


bool Vine3DProject::modelDailyCycle(bool isInitialState, Crit3DDate myDate, int nrHours,
                                    const QString& myOutputPath, bool saveOutput)
{

    TfieldOperation operation;
    float quantity;
    QDate myQDate = getQDate(myDate);
    Crit3DTime myCurrentTime, myFirstTime, myLastTime;
    int myTimeStep = int(getTimeStep());
    myFirstTime = Crit3DTime(myDate, myTimeStep);
    myLastTime = Crit3DTime(myDate, nrHours * 3600);
    bool isNewModelCase;
    int modelCaseIndex;
    double* myProfile;

    for (myCurrentTime = myFirstTime; myCurrentTime <= myLastTime; myCurrentTime = myCurrentTime.addSeconds(myTimeStep))
    {
        QDateTime myQTime = getQDateTime(myCurrentTime);

        logInfo("\n" + myQTime.toString("yyyy-MM-dd hh:mm"));
        grapevine.setDate(myCurrentTime);

        // meteo interpolation
        logInfo("Interpolate meteo data");
        interpolateAndSaveHourlyMeteo(airTemperature, myQTime, myOutputPath, saveOutput);
        interpolateAndSaveHourlyMeteo(precipitation, myQTime, myOutputPath, saveOutput);
        interpolateAndSaveHourlyMeteo(airRelHumidity, myQTime, myOutputPath, saveOutput);
        interpolateAndSaveHourlyMeteo(windScalarIntensity, myQTime, myOutputPath, saveOutput);
        interpolateAndSaveHourlyMeteo(globalIrradiance, myQTime, myOutputPath, saveOutput);

        // ET0
        hourlyMeteoMaps->computeET0PMMap(DEM, radiationMaps);
        if (saveOutput)
        {
            saveHourlyMeteoOutput(referenceEvapotranspiration, myOutputPath, myQTime);
        }

        // Leaf Wetness
        hourlyMeteoMaps->computeLeafWetnessMap();
        if (saveOutput)
        {
            saveHourlyMeteoOutput(leafWetness, myOutputPath, myQTime);
        }

        if (isInitialState)
        {
            initializeSoilMoisture(myCurrentTime.date.month);
        }

        //Grapevine
        double vineTranspiration, grassTranspiration;
        logInfo("Compute grapevine");
        for (long row = 0; row < DEM.header->nrRows ; row++)
        {
            for (long col = 0; col < DEM.header->nrCols; col++)
            {
                modelCaseIndex = getModelCaseIndex(row, col);
                if (modelCaseIndex != NODATA)
                {
                    isNewModelCase = (int(statePlantMaps->fruitBiomassMap->value[row][col])
                                        == int(statePlantMaps->fruitBiomassMap->header->flag));

                    if (! grapevine.setWeather(
                                double(vine3DMapsD->mapDailyTAvg->value[row][col]),
                                double(hourlyMeteoMaps->mapHourlyTair->value[row][col]),
                                double(radiationMaps->globalRadiationMap->value[row][col]),
                                double(hourlyMeteoMaps->mapHourlyPrec->value[row][col]),
                                double(hourlyMeteoMaps->mapHourlyRelHum->value[row][col]),
                                double(hourlyMeteoMaps->mapHourlyWindScalarInt->value[row][col]),
                                P0))
                    {
                        errorString = grapevineError(myCurrentTime, row, col, "Weather data missing");
                        return(false);
                    }

                    if (! grapevine.setDerivedVariables(
                                double(radiationMaps->diffuseRadiationMap->value[row][col]),
                                double(radiationMaps->beamRadiationMap->value[row][col]),
                                double(radiationMaps->transmissivityMap->value[row][col] / CLEAR_SKY_TRANSMISSIVITY_DEFAULT),
                                double(radiationMaps->sunElevationMap->value[row][col])))
                    {
                        errorString = grapevineError(myCurrentTime, row, col, "Radiation data missing");
                        return (false);
                    }

                    grapevine.resetLayers();

                    if (! setSoilProfileCrop(row, col, &(modelCases[modelCaseIndex])))
                    {
                        errorString = grapevineError(myCurrentTime, row, col, "Error in soil profile setting");
                        return false;
                    }

                    if ((isInitialState) || (isNewModelCase))
                    {
                        if(! grapevine.initializeStatePlant(getDoyFromDate(myDate), &(modelCases[modelCaseIndex])))
                        {
                            logInfo("Could not initialize grapevine in the growing season.\nIt will be replaced by a complete grass cover.");
                        }
                    }
                    else
                    {
                        if (! setStatePlantfromMap(row, col, this))
                            return false;
                        grapevine.setStatePlant(statePlant, true);
                    }
                    double chlorophyll = NODATA;

                    if (! grapevine.compute((myCurrentTime == myFirstTime), myTimeStep, &(modelCases[modelCaseIndex]), chlorophyll))
                    {
                        errorString = grapevineError(myCurrentTime, row, col, "Error in grapevine computation");
                        return false;
                    }

                    // check field book (first hour)
                    if (myCurrentTime.getHour() == 1)
                    {
                        int idBook = 0;
                        while (getFieldBookIndex(idBook, myQDate, modelCaseIndex, &idBook))
                        {
                            operation = fieldBook[idBook].operation;
                            quantity = fieldBook[idBook].quantity;
                            grapevine.fieldBookAction(&(modelCases[modelCaseIndex]), operation, quantity);
                            idBook++;
                        }
                    }

                    statePlant = grapevine.getStatePlant();
                    getStatePlantToMap(row, col, this, &(statePlant));

                    myProfile = grapevine.getExtractedWater(&(modelCases[modelCaseIndex]));

                    for (unsigned int layer=0; layer < nrLayers; layer++)
                        outputPlantMaps->transpirationLayerMaps[layer]->value[row][col] = float(myProfile[layer]);

                    vineTranspiration = grapevine.getRealTranspirationGrapevine(&(modelCases[modelCaseIndex]));
                    grassTranspiration = grapevine.getRealTranspirationGrass(&(modelCases[modelCaseIndex]));

                    if (myCurrentTime == myFirstTime)
                    {
                        outputPlantMaps->vineyardTranspirationMap->value[row][col] = float(vineTranspiration);
                        outputPlantMaps->grassTranspirationMap->value[row][col] = float(grassTranspiration);
                    }
                    else
                    {
                        // summed values
                        outputPlantMaps->vineyardTranspirationMap->value[row][col] += float(vineTranspiration);
                        outputPlantMaps->grassTranspirationMap->value[row][col] += float(grassTranspiration);
                    }

                    // vine stress (midday)
                    if (myCurrentTime.getHour() == 12)
                    {
                        outputPlantMaps->vineStressMap->value[row][col] = float(grapevine.getStressCoefficient());
                    }
                }
            }
        }

        // Irrigation
        assignIrrigation(myCurrentTime);

        if (! computeVine3DWaterSinkSource())
            return false;

        // 3D soil water fluxes
        bool isRestart = false;
        runWaterFluxes3DModel(3600, isRestart);

        if (myCurrentTime == myFirstTime)
        {
            resetWaterBalanceMap();
        }

        updateWaterBalanceMaps();

        if (isInitialState)
            isInitialState = false;
    }

    return true;
}
