#ifndef INTERPOLATIONCONSTS_H
#define INTERPOLATIONCONSTS_H

    #define MIN_REGRESSION_POINTS 5
    #define PEARSONSTANDARDTHRESHOLD 0.1
    #define SHEPARD_MIN_NRPOINTS 4
    #define SHEPARD_AVG_NRPOINTS 7
    #define SHEPARD_MAX_NRPOINTS 10

    #ifndef _STRING_
        #include <string>
    #endif

    #ifndef _MAP_
        #include <map>
    #endif

    enum TInterpolationMethod { idw, shepard, shepard_modified };

    const std::map<std::string, TInterpolationMethod> interpolationMethodNames = {
      { "idw", idw },
      { "shepard", shepard },
      { "shepard_modified", shepard_modified }
    };

    enum TProxyVar { height, heightInversion, urbanFraction, orogIndex, seaDistance, aspect, slope, noProxy };

    const std::map<std::string, TProxyVar> ProxyVarNames = {
      { "elevation", height },
      { "altitude", height },
      { "orography", height },
      { "orogIndex", orogIndex },
      { "urbanFraction", urbanFraction },
      { "seaDistance", seaDistance },
      { "aspect", aspect },
      { "slope", slope }
    };


    enum TkrigingMode {KRIGING_SPHERICAL = 1,
                       KRIGING_EXPONENTIAL=2,
                       KRIGING_GAUSSIAN=3,
                       KRIGING_LINEAR=4
                      };


#endif // INTERPOLATIONCONSTS_H
