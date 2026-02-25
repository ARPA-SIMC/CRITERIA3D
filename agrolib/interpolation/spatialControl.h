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

bool checkData(Crit3DQuality* myQuality, meteoVariable myVar, std::vector<Crit3DMeteoPoint> &meteoPoints,
               const Crit3DTime &myTime, Crit3DInterpolationSettings &spatialQualityInterpolationSettings,
               Crit3DMeteoSettings* meteoSettings, Crit3DClimateParameters* climateParameters, bool checkSpatial, std::string &errorStr);

bool checkAndPassDataToInterpolation(Crit3DQuality* myQuality, meteoVariable myVar, std::vector<Crit3DMeteoPoint> &meteoPoints,
                                     const Crit3DTime &myTime, Crit3DInterpolationSettings &SQinterpolationSettings,
                                     Crit3DInterpolationSettings &interpolationSettings, Crit3DMeteoSettings *meteoSettings,
                                     Crit3DClimateParameters *climateParameters, std::vector<Crit3DInterpolationDataPoint> &interpolationPoints,
                                     bool checkSpatial, std::string &errorStr);

    bool passDataToInterpolation(const std::vector<Crit3DMeteoPoint> &meteoPoints,
                                 std::vector<Crit3DInterpolationDataPoint> &myInterpolationPoints,
                                 Crit3DInterpolationSettings &interpolationSettings);

    bool computeResiduals(meteoVariable myVar, std::vector<Crit3DMeteoPoint> &meteoPoints,
                          const std::vector <Crit3DInterpolationDataPoint> &interpolationPoints,
                          Crit3DInterpolationSettings &interpolationSettings, Crit3DMeteoSettings* meteoSettings,
                          bool excludeOutsideDem, bool excludeSupplemental);

    bool computeResidualsLocalDetrending(meteoVariable myVar, const Crit3DTime &myTime, std::vector<Crit3DMeteoPoint> &meteoPoints,
                                         std::vector <Crit3DInterpolationDataPoint> &interpolationPoints,
                                         Crit3DInterpolationSettings &interpolationSettings,
                                         Crit3DMeteoSettings* meteoSettings, Crit3DClimateParameters* climateParameters,
                                         bool excludeOutsideDem, bool excludeSupplemental);

    bool computeResidualsGlocalDetrending(meteoVariable myVar, const Crit3DMacroArea &myArea, int elevationPos,
                                          std::vector<Crit3DMeteoPoint> &meteoPoints, std::vector <Crit3DInterpolationDataPoint> &interpolationPoints,
                                          Crit3DInterpolationSettings &interpolationSettings, Crit3DMeteoSettings* meteoSettings,
                                          bool excludeOutsideDem, bool excludeSupplemental);

    float computeErrorCrossValidation(const std::vector<Crit3DMeteoPoint> &meteoPoints);

    bool spatialQualityControl(meteoVariable myVar, std::vector<Crit3DMeteoPoint> &meteoPoints,
                               Crit3DInterpolationSettings &interpolationSettings, Crit3DMeteoSettings* meteoSettings,
                               Crit3DClimateParameters* climateParameters, const Crit3DTime &myTime, std::string &errorStr);

#endif // SPATIALCONTROL_H
