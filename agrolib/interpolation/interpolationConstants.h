#ifndef INTERPOLATIONCONSTS_H
#define INTERPOLATIONCONSTS_H

    #define MIN_REGRESSION_POINTS 5
    #define PEARSONSTANDARDTHRESHOLD 0.1
    #define SHEPARD_MIN_NRPOINTS 5
    #define SHEPARD_AVG_NRPOINTS 8
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

    enum TProxyVar { proxyHeight, proxyUrbanFraction, proxyOrogIndex, proxySeaDistance, proxyAspect, proxySlope, proxyWaterIndex, noProxy };

    const std::map<std::string, TProxyVar> ProxyVarNames = {
      { "elevation", proxyHeight },
      { "altitude", proxyHeight },
      { "orography", proxyHeight },
      { "orogIndex", proxyOrogIndex },
      { "urbanFraction", proxyUrbanFraction },
      { "urban", proxyUrbanFraction},
      { "seaDistance", proxySeaDistance },
      { "aspect", proxyAspect },
      { "slope", proxySlope },
      { "water_index", proxyWaterIndex}
    };

    enum TFittingFunction { piecewiseTwo, piecewiseThreeFree, piecewiseThree, linear, noFunction };

    const std::map<std::string, TFittingFunction> fittingFunctionNames = {
        { "double_piecewise", piecewiseTwo },
        { "free_triple_piecewise", piecewiseThreeFree},
        { "triple_piecewise", piecewiseThree},
        { "linear", linear }
    };


    enum TkrigingMode {KRIGING_SPHERICAL = 1,
                       KRIGING_EXPONENTIAL=2,
                       KRIGING_GAUSSIAN=3,
                       KRIGING_LINEAR=4
                      };


#endif // INTERPOLATIONCONSTS_H
