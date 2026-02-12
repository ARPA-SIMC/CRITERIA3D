#ifndef TRANSMISSIVITY_H
#define TRANSMISSIVITY_H

    #ifndef METEOPOINT_H
        #include "meteoPoint.h"
    #endif

    class Crit3DRadiationSettings;

    float computePointTransmissivitySamani(float tmin, float tmax, float samaniCoeff);

    bool computeTransmissivity(Crit3DRadiationSettings *mySettings, std::vector<Crit3DMeteoPoint> &meteoPoints,
                               int intervalWidth, Crit3DTime myTime, const gis::Crit3DRasterGrid& myDEM);

    bool computeTransmissivityFromTRange(std::vector<Crit3DMeteoPoint> &meteoPoints, Crit3DTime currentTime);


#endif // TRANSMISSIVITY_H
