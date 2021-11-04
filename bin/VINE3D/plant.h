#ifndef PLANT_H
#define PLANT_H

    #ifndef QDATETIME_H
        #include <QDateTime>
    #endif
    #ifndef QSTRING_H
        #include <QString>
    #endif
    #ifndef GIS_H
        #include "gis.h"
    #endif


    class Vine3DProject;

    struct TstatePlant;

    enum plantVariable {fruitBiomassIndexVar,wineYieldVar,isHarvestedVar,tartaricAcidVar, pHBerryVar, brixBerryVar, brixMaximumVar,deltaBrixVar,daysFromFloweringVar, daysAfterBloomVar,
                    cumulatedBiomassVar, fruitBiomassVar, shootLeafNumberVar, meanTemperatureLastMonthVar,
                    chillingUnitsVar, forceStateBudBurstVar, forceStateVegetativeSeasonVar, stageVar,
                    cumRadFruitsetVerVar, leafAreaIndexVar, transpirationStressVar,
                    transpirationVineyardVar, transpirationGrassVar,
                    degreeDaysFromFirstMarchVar, degreeDays10FromBudBurstVar, degreeDaysAtFruitSetVar,
                    powderyAICVar, powderyCurrentColoniesVar, powderySporulatingColoniesVar,
                    powderyCOLVar, powderyINFRVar, powderyPrimaryInfectionRiskVar,
                    downyDormantOosporeVar, downyInfectionRateVar, downyOilSpotVar,
                    noPlantVar};

    class Crit3DStatePlantMaps
    {
    public:
        gis::Crit3DRasterGrid* leafAreaIndexMap;

        gis::Crit3DRasterGrid* pHBerryMap;
        gis::Crit3DRasterGrid* daysAfterBloomMap;

        gis::Crit3DRasterGrid* fruitBiomassIndexMap;
        gis::Crit3DRasterGrid* isHarvestedMap;
        gis::Crit3DRasterGrid* cumulatedBiomassMap;
        gis::Crit3DRasterGrid* fruitBiomassMap;
        gis::Crit3DRasterGrid* shootLeafNumberMap;
        gis::Crit3DRasterGrid* meanTemperatureLastMonthMap;
        gis::Crit3DRasterGrid* chillingStateMap;
        gis::Crit3DRasterGrid* forceStateBudBurstMap;
        gis::Crit3DRasterGrid* forceStateVegetativeSeasonMap;
        gis::Crit3DRasterGrid* stageMap;
        gis::Crit3DRasterGrid* cumulatedRadiationFromFruitsetToVeraisonMap;
        gis::Crit3DRasterGrid* degreeDaysFromFirstMarchMap;
        gis::Crit3DRasterGrid* degreeDay10FromBudBurstMap;
        gis::Crit3DRasterGrid* degreeDaysAtFruitSetMap;
        gis::Crit3DRasterGrid* powderyAICMap;
        gis::Crit3DRasterGrid* powderyCurrentColoniesMap;
        gis::Crit3DRasterGrid* powderySporulatingColoniesMap;

        Crit3DStatePlantMaps();
        Crit3DStatePlantMaps(const gis::Crit3DRasterGrid& myDEM);
        void initialize();

        gis::Crit3DRasterGrid* getMapFromVar(plantVariable myVar);

        bool isLoaded;
    };

    class Crit3DOutputPlantMaps
    {
    public:
        gis::Crit3DRasterGrid* daysFromFloweringMap;
        gis::Crit3DRasterGrid* brixMaximumMap;
        gis::Crit3DRasterGrid* brixBerryMap;
        gis::Crit3DRasterGrid* vineStressMap;
        gis::Crit3DRasterGrid* vineyardTranspirationMap;
        gis::Crit3DRasterGrid* grassTranspirationMap;
        gis::Crit3DRasterGrid* powderyCOLMap;
        gis::Crit3DRasterGrid* powderyINFRMap;
        gis::Crit3DRasterGrid* powderyPrimaryInfectionRiskMap;
        gis::Crit3DRasterGrid** transpirationLayerMaps;
        gis::Crit3DRasterGrid* tartaricAcidMap;
        gis::Crit3DRasterGrid* deltaBrixMap;
        gis::Crit3DRasterGrid* wineYieldMap;
        gis::Crit3DRasterGrid* downyDormantOosporeMap;
        gis::Crit3DRasterGrid* downyInfectionRateMap;
        gis::Crit3DRasterGrid* downyOilSpotMap;

        Crit3DOutputPlantMaps();
        Crit3DOutputPlantMaps(const gis::Crit3DRasterGrid &myDEM);
        Crit3DOutputPlantMaps(const gis::Crit3DRasterGrid &myDEM, int nrSoilLayers);

        void initialize();
        void initializeWithDEM(const gis::Crit3DRasterGrid &myDEM);
        gis::Crit3DRasterGrid* getMapFromVar(plantVariable myVar);
    };

    bool initializeGrapevine(Vine3DProject* myProject);
    bool setStatePlantfromMap(long row, long col , Vine3DProject* myProject);
    bool getStatePlantToMap(long row,long col, Vine3DProject* myProject, TstatePlant* statePlant);
    bool getOutputPlantToMap(long row, long col, Vine3DProject* myProject);

    bool savePlantState(Vine3DProject* myProject, plantVariable myVar,
                        QDate myDate, QString myPath, QString myArea);
    bool loadPlantState(Vine3DProject* myProject, plantVariable myVar,
                        QDate myDate, QString myPath, QString myArea);
    bool savePlantOutput(Vine3DProject* myProject, plantVariable myVar,
                        QDate myDate, QString myPath, QString myArea, QString notes,
                         bool isStateMap, bool isMasked);

    bool updateThermalSum(Vine3DProject* myProject, QDate myDate);
#endif
