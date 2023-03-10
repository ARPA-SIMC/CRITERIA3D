#ifndef SPATIALCONTROL_H
#define SPATIALCONTROL_H

    #ifndef QUALITY_H
        #include "quality.h"
    #endif
    #ifndef METEOPOINT_H
        #include "meteoPoint.h"
    #endif
    #ifndef INTERPOLATIONSETTINGS_H
        #include "interpolationSettings.h"
    #endif
    #ifndef INTERPOLATIONPOINT_H
        #include "interpolationPoint.h"
    #endif

    bool checkData(Crit3DQuality* myQuality, meteoVariable myVar, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints, Crit3DTime myTime,
                   Crit3DInterpolationSettings* spatialQualityInterpolationSettings, Crit3DMeteoSettings *meteoSettings, Crit3DClimateParameters *myClimate, bool checkSpatial);

    bool checkAndPassDataToInterpolation(Crit3DQuality* myQuality, meteoVariable myVar, Crit3DMeteoPoint* meteoPoints,
                                         int nrMeteoPoints, Crit3DTime myTime, Crit3DInterpolationSettings *SQinterpolationSettings,
                                         Crit3DInterpolationSettings* interpolationSettings, Crit3DMeteoSettings *meteoSettings, Crit3DClimateParameters *myClimate,
                                         std::vector<Crit3DInterpolationDataPoint> &myInterpolationPoints,
                                         bool checkSpatial);

    bool passDataToInterpolation(Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints,
                             std::vector<Crit3DInterpolationDataPoint> &myInterpolationPoints, Crit3DInterpolationSettings* mySettings);

    bool computeResiduals(meteoVariable myVar, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints,
                          std::vector <Crit3DInterpolationDataPoint> &interpolationPoints, Crit3DInterpolationSettings* settings, Crit3DMeteoSettings* meteoSettings, bool excludeOutsideDem, bool excludeSupplemental);

    float computeErrorCrossValidation(meteoVariable myVar, Crit3DMeteoPoint *myPoints, int nrMeteoPoints, const Crit3DTime& myTime, Crit3DMeteoSettings *meteoSettings);


#endif // SPATIALCONTROL_H
