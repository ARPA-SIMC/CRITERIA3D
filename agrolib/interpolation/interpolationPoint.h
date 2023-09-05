#ifndef INTERPOLATIONPOINT_H
#define INTERPOLATIONPOINT_H

    #ifndef GIS_H
        #include "gis.h"
    #endif
    #ifndef METEO_H
        #include "meteo.h"
    #endif
    #ifndef INTERPOLATIONSETTINGS_H
        #include "interpolationSettings.h"
    #endif

    class Crit3DInterpolationDataPoint {
    private:

    public:
        gis::Crit3DPoint* point;
        int index;
        bool isActive;
        bool isMarked;
        float distance;
        float deltaZ;
        float value;
        lapseRateCodeType lapseRateCode;
        gis::Crit3DRasterGrid* topographicDistance;
        std::vector<float> proxyValues;

        Crit3DInterpolationDataPoint();

        float getProxyValue(unsigned int pos);
        std::vector <float> getProxyValues();
        bool getActiveProxyValues(Crit3DProxyCombination &activeCombination, std::vector<double> &myValues);
    };

#endif // INTERPOLATIONPOINT_H
