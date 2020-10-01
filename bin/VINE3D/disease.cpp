#include "disease.h"
#include "powderyMildew.h"
#include "downyMildew.h"
#include "meteo.h"
#include "atmosphere.h"
#include "dataHandler.h"
#include "commonConstants.h"
#include "utilities.h"
#include <iostream>

#define MAXPOINTS 4096
#define VEGETATIVESTART 90
#define VEGETATIVEEND 270

// OIDIO
bool computePowderyMildew(Vine3DProject* myProject)
{
    Tmildew powdery;
    float phenoPhase;
    bool isBudBurst;
    myProject->logInfo("Compute powdery mildew...");

    for (long row = 0; row < myProject->DEM.header->nrRows ; row++)
        for (long col = 0; col < myProject->DEM.header->nrCols; col++)
        {
            if (myProject->DEM.value[row][col] != myProject->DEM.header->flag)
            {
                phenoPhase = myProject->statePlantMaps->stageMap->value[row][col];
                if (phenoPhase != myProject->statePlantMaps->stageMap->header->flag)
                {
                    // initialize output
                    myProject->outputPlantMaps->powderyCOLMap->value[row][col] = 0.0;
                    myProject->outputPlantMaps->powderyINFRMap->value[row][col] = 0.0;
                    myProject->outputPlantMaps->powderyPrimaryInfectionRiskMap->value[row][col] = 0.0;

                    // bud burst
                    if (phenoPhase >= 2.0)
                    {
                        // read state
                        powdery.state.degreeDays = myProject->statePlantMaps->degreeDay10FromBudBurstMap->value[row][col];
                        if (powdery.state.degreeDays == myProject->statePlantMaps->degreeDay10FromBudBurstMap->header->flag)
                            isBudBurst = true;
                        else
                            isBudBurst = false;

                        // Ascospores in Chasmothecia
                        powdery.state.aic = myProject->statePlantMaps->powderyAICMap->value[row][col];
                        powdery.state.currentColonies = myProject->statePlantMaps->powderyCurrentColoniesMap->value[row][col];
                        powdery.state.totalSporulatingColonies = myProject->statePlantMaps->powderySporulatingColoniesMap->value[row][col];

                        // read meteo
                        powdery.input.tavg = myProject->vine3DMapsD->mapDailyTAvg->value[row][col];
                        powdery.input.relativeHumidity = myProject->vine3DMapsD->mapDailyRHAvg->value[row][col];
                        powdery.input.rain = myProject->vine3DMapsD->mapDailyPrec->value[row][col];
                        powdery.input.leafWetness = int(myProject->vine3DMapsD->mapDailyLeafW->value[row][col]);

                        powderyMildew(&powdery, isBudBurst);

                        // save state
                        myProject->statePlantMaps->degreeDay10FromBudBurstMap->value[row][col] = powdery.state.degreeDays;
                        myProject->statePlantMaps->powderyAICMap->value[row][col] = powdery.state.aic;
                        myProject->statePlantMaps->powderyCurrentColoniesMap->value[row][col] = powdery.state.currentColonies;
                        myProject->statePlantMaps->powderySporulatingColoniesMap->value[row][col] = powdery.state.totalSporulatingColonies;

                        // save output
                        myProject->outputPlantMaps->powderyCOLMap->value[row][col] = powdery.output.col;
                        myProject->outputPlantMaps->powderyINFRMap->value[row][col] = powdery.output.infectionRate;
                        myProject->outputPlantMaps->powderyPrimaryInfectionRiskMap->value[row][col] = MINVALUE(5.0 * powdery.output.infectionRisk, 1.0);
                    }
                    else
                    {
                        myProject->statePlantMaps->degreeDay10FromBudBurstMap->value[row][col] = myProject->statePlantMaps->degreeDay10FromBudBurstMap->header->flag;
                        myProject->statePlantMaps->powderyAICMap->value[row][col] = 0.0;
                        myProject->statePlantMaps->powderyCurrentColoniesMap->value[row][col]  = 0.0;
                        myProject->statePlantMaps->powderySporulatingColoniesMap->value[row][col] = 0.0;
                    }
                }
            }
        }
    return true;
}


bool computeDownyMildew(Vine3DProject* myProject, QDate firstDate, QDate lastDate, unsigned lastHour, QString myArea)
{
    using namespace std;

    myProject->logInfo("\nCompute downy mildew...");

    QDate firstJanuary;
    firstJanuary.setDate(lastDate.year(), 1, 1);

    int lastDoy = firstJanuary.daysTo(lastDate) + 1;
    int firstDoy = firstJanuary.daysTo(firstDate) + 1;

    //check vegetative season
    if ((lastDoy < VEGETATIVESTART) || (firstDoy >= VEGETATIVEEND))
    {
        myProject->logInfo("Out of vegetative season");
        return true;
    }

    //check date
    firstDoy = MAXVALUE(firstDoy, VEGETATIVESTART);
    firstDate = firstJanuary.addDays(firstDoy - 1);
    if (lastDoy > VEGETATIVEEND)
    {
        lastDoy = VEGETATIVEEND;
        lastHour = 23;
    }
    lastDate = firstJanuary.addDays(lastDoy - 1);
    unsigned nrHours = (lastDoy -1)* 24 + lastHour;
    unsigned nrSavingDays = lastDoy - firstDoy + 1;

    QDateTime firstTime, lastTime;
    firstTime.setDate(firstJanuary);
    firstTime.setTime(QTime(1, 0, 0, 0));
    lastTime.setDate(lastDate);
    lastTime.setTime(QTime(int(lastHour), 0, 0, 0));

    if (!myProject->loadObsDataFilled(firstTime, lastTime))
    {
        myProject->logError();
        return false;
    }

    unsigned rowPoint[MAXPOINTS], colPoint[MAXPOINTS];

    vector<TdownyMildewInput> input;
    vector<gis::Crit3DRasterGrid*> infectionMap;
    vector<gis::Crit3DRasterGrid*> oilSpotMap;

    TdownyMildew downyMildewCore;

    unsigned n, nrPoints, row, col;
    int doy;
    QString dailyPath, variableMissing;
    float sumOilSpots;

    infectionMap.resize(nrSavingDays);
    oilSpotMap.resize(nrSavingDays);
    for (n = 0; n < nrSavingDays; n++)
    {
        infectionMap[n] = new gis::Crit3DRasterGrid;
        infectionMap[n]->initializeGrid(myProject->DEM);
        oilSpotMap[n] = new gis::Crit3DRasterGrid;
        oilSpotMap[n]->initializeGrid(myProject->DEM);
    }

    Crit3DTime myTime;
    bool missingData = false;
    bool isLastCell = false;
    nrPoints = 0;
    row = 0;
    col = 0;
    int groupId = 0;

    while ((! missingData) && (! isLastCell))
    {
        if (myProject->isVineyard(row, col))
        {
            rowPoint[nrPoints] = row;
            colPoint[nrPoints] = col;
            nrPoints++;
        }

        col++;
        if (col == unsigned(myProject->DEM.header->nrCols))
        {
            row++;
            if (row == unsigned(myProject->DEM.header->nrRows)) isLastCell = true;
            col = 0;
        }

        if ((nrPoints == MAXPOINTS) || (isLastCell && (nrPoints > 0)))
        {
            groupId++;

            input.clear();
            input.resize(nrPoints*nrHours);

            myProject->logInfo("Interpolating hourly data. Group " + QString::number(groupId) + ". Nr points: " + QString::number(nrPoints));
            myTime = getCrit3DTime(firstTime);
            for (unsigned h = 0; h < nrHours; h++)
            {
                if ((myTime.date.day == 1) && (myTime.getHour() == 1))
                    myProject->logInfo("Compute hourly data - month: " + QString::number(myTime.date.month));

                if (! interpolationProjectDemMain(myProject, airTemperature, myTime, true))
                {
                    missingData = true;
                    variableMissing = "Air temperature";
                    break;
                }
                if (! interpolationProjectDemMain(myProject, airRelHumidity, myTime, true))
                {
                    missingData = true;
                    variableMissing = "Air humidity";
                    break;
                }
                if (! interpolationProjectDemMain(myProject, precipitation, myTime, true))
                {
                    missingData = true;
                    variableMissing = "Rainfall";
                    break;
                }

                myProject->hourlyMeteoMaps->computeLeafWetnessMap();

                for (n = 0; n < nrPoints; n++)
                {
                    row = rowPoint[n];
                    col = colPoint[n];
                    input[n*nrHours+h].tair = myProject->hourlyMeteoMaps->mapHourlyTair->value[row][col];
                    input[n*nrHours+h].relativeHumidity = myProject->hourlyMeteoMaps->mapHourlyRelHum->value[row][col];
                    input[n*nrHours+h].rain = myProject->hourlyMeteoMaps->mapHourlyPrec->value[row][col];
                    input[n*nrHours+h].leafWetness = int(myProject->hourlyMeteoMaps->mapHourlyLeafW->value[row][col]);
                }
                myTime = myTime.addSeconds(3600);
            }

            // Downy cycle computation
            if (! missingData)
            {
                myProject->logInfo("Downy mildew model. Group " + QString::number(groupId) + ". Nr points: " + QString::number(nrPoints));
                for (n = 0; n < nrPoints; n++)
                {
                    row = rowPoint[n];
                    col = colPoint[n];
                    sumOilSpots = 0;
                    // Downy Mildew initialization
                    downyMildewCore.input = input[n*nrHours];
                    downyMildew(&downyMildewCore, true);

                    for (unsigned h = 1; h < nrHours; h++)
                    {
                        downyMildewCore.input = input[n*nrHours + h];
                        downyMildew(&downyMildewCore, false);

                        if (downyMildewCore.output.oilSpots > 0.0) sumOilSpots += (downyMildewCore.output.oilSpots * 100.0);

                        if (((h % 24)==0) || (h == (nrHours-1)))
                        {
                            if (h == (nrHours-1))
                                doy = lastDoy;
                            else
                                doy = int(h / 24);
                            if (doy >= firstDoy)
                            {
                                infectionMap[doy-firstDoy]->value[row][col] = downyMildewCore.output.infectionRate * 100.0;
                                oilSpotMap[doy-firstDoy]->value[row][col] = sumOilSpots;
                            }
                        }
                    }
                }
            } 
            nrPoints = 0;
        }
    }

    // Save output
    if (! missingData)
    {
        QString fileName, outputFileName;
        QDate myDate;
        std::string myErrorString;
        for (n = 0; n < nrSavingDays; n++)
        {
            myDate = firstDate.addDays(n);
            dailyPath = myProject->getProjectPath() + myProject->dailyOutputPath + myDate.toString("yyyy/MM/dd/");

            fileName = getOutputNameDaily("downyINFR", myArea, "", myDate);
            outputFileName = dailyPath + fileName;
            gis::writeEsriGrid(outputFileName.toStdString(), infectionMap[n], &myErrorString);

            fileName = getOutputNameDaily("downySymptoms", myArea, "", myDate);
            outputFileName = dailyPath + fileName;
            gis::writeEsriGrid(outputFileName.toStdString(), oilSpotMap[n], &myErrorString);
        }
        myProject->logInfo("Downy mildew computed.");
    }

    // Clean memory
    for (n = 0; n < nrSavingDays; n++)
    {
        infectionMap[n]->clear();
        oilSpotMap[n]->clear();
    }
    infectionMap.clear();
    oilSpotMap.clear();
    input.clear();

    if (missingData)
    {
        myProject->logInfo("\nMissing hourly data to compute DownyMildew:"
                + getQDate(myTime.date).toString("yyyy/MM/dd")
                + "\n" + variableMissing);
        return false;
    }
    else
    {
        return true;
    }
}
