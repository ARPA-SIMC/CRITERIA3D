#ifndef SPATIALCONTROL_H
#define SPATIALCONTROL_H

    #ifndef QUALITY_H
        #include "quality.h"
    #endif
    #ifndef METEO_H
        #include "meteo.h"
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

    bool checkData(Crit3DQuality* myQuality, meteoVariable myVar, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints, const Crit3DTime &myTime,
                    Crit3DInterpolationSettings* spatialQualityInterpolationSettings, Crit3DMeteoSettings *meteoSettings,
                    Crit3DClimateParameters *myClimate, bool checkSpatial, std::string &errorStr);

    bool checkAndPassDataToInterpolation(Crit3DQuality* myQuality, meteoVariable myVar, Crit3DMeteoPoint* meteoPoints,
                                         int nrMeteoPoints, const Crit3DTime &myTime, Crit3DInterpolationSettings *SQinterpolationSettings,
                                         Crit3DInterpolationSettings* interpolationSettings, Crit3DMeteoSettings *meteoSettings,
                                         Crit3DClimateParameters *myClimate, std::vector<Crit3DInterpolationDataPoint> &interpolationPoints,
                                         bool checkSpatial, std::string &errorStr);

    bool passDataToInterpolation(Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints,
                                 std::vector<Crit3DInterpolationDataPoint> &myInterpolationPoints,
                                 Crit3DInterpolationSettings* mySettings);

    bool computeResiduals(meteoVariable myVar, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints,
                          const std::vector <Crit3DInterpolationDataPoint> &interpolationPoints,
                          Crit3DInterpolationSettings* settings, Crit3DMeteoSettings* meteoSettings,
                          bool excludeOutsideDem, bool excludeSupplemental);

    bool computeResidualsLocalDetrending(meteoVariable myVar, const Crit3DTime &myTime, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints,
                                         std::vector <Crit3DInterpolationDataPoint> &interpolationPoints,
                                         Crit3DInterpolationSettings* settings, Crit3DMeteoSettings* meteoSettings,
                                         Crit3DClimateParameters *climateParameters, bool excludeOutsideDem, bool excludeSupplemental);

    bool computeResidualsGlocalDetrending(meteoVariable myVar, const Crit3DMacroArea &myArea, int elevationPos,
                                          Crit3DMeteoPoint* meteoPoints, std::vector <Crit3DInterpolationDataPoint> &interpolationPoints,
                                          Crit3DInterpolationSettings* settings, Crit3DMeteoSettings* meteoSettings,
                                          bool excludeOutsideDem, bool excludeSupplemental);

    float computeErrorCrossValidation(Crit3DMeteoPoint *myPoints, int nrMeteoPoints);

    bool spatialQualityControl(meteoVariable myVar, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints,
                               Crit3DInterpolationSettings *settings, Crit3DMeteoSettings* meteoSettings,
                               Crit3DClimateParameters* myClimate, const Crit3DTime &myTime, std::string &errorStr);

#endif // SPATIALCONTROL_H
