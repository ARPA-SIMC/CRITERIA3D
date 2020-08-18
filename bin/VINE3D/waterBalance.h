#ifndef WATERBALANCE_H
#define WATERBALANCE_H

    #ifndef GIS_H
        #include "gis.h"
    #endif
    #ifndef CRIT3DDATE_H
        #include "crit3dDate.h"
    #endif
    #ifndef PROJECT3D_H
        #include "project3D.h"
    #endif

    class QString;
    class QDateTime;
    class Vine3DProject;


    class Crit3DWaterBalanceMaps
    {
        public:
            gis::Crit3DRasterGrid* bottomDrainageMap;
            gis::Crit3DRasterGrid* waterInflowMap;

            Crit3DWaterBalanceMaps();
            Crit3DWaterBalanceMaps(const gis::Crit3DRasterGrid &myDEM);

            void initialize();
            void initializeWithDEM(const gis::Crit3DRasterGrid &myDEM);
            gis::Crit3DRasterGrid* getMapFromVar(criteria3DVariable myVar);
    };

    bool saveWaterBalanceState(Vine3DProject* myProject, QDate myDate, QString myArea,
                               QString myStatePath, criteria3DVariable myVar);

    bool loadWaterBalanceState(Vine3DProject* myProject, QDate myDate, QString myArea,
                               QString myStatePath, criteria3DVariable myVar);

    bool getCriteria3DVarMap(Vine3DProject* myProject, criteria3DVariable myVar, int layerIndex,
                             gis::Crit3DRasterGrid* criteria3DMap);

    bool getRootZoneAWCmap(Vine3DProject* myProject, gis::Crit3DRasterGrid* outputMap);

    bool getCriteria3DIntegrationMap(Vine3DProject* myProject, criteria3DVariable myVar,
                           double upperDepth, double lowerDepth, gis::Crit3DRasterGrid* criteria3DMap);

    bool saveWaterBalanceOutput(Vine3DProject* myProject, QDate myDate, criteria3DVariable myVar,
                                QString varName, QString notes, QString outputPath, QString myArea,
                                double upperDepth, double lowerDepth);

    bool saveWaterBalanceCumulatedOutput(Vine3DProject* myProject, QDate myDate, criteria3DVariable myVar,
                                QString varName, QString notes, QString outputPath, QString myArea);

    double* getCriteria3DVarProfile(Vine3DProject* myProject, int myRow, int myCol, criteria3DVariable myVar);
    double* getSoilVarProfile(Vine3DProject* myProject, int myRow, int myCol, soil::soilVariable myVar);

    void resetWaterBalanceMap(Vine3DProject* myProject);
    void updateWaterBalanceMaps(Vine3DProject* myProject);


#endif // WATERBALANCE_H
