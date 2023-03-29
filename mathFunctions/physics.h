#ifndef PHYSICS_H
#define PHYSICS_H

    #include <vector>

    double pressureFromAltitude(double myHeight);
    double airMolarDensity(double myPressure, double myT);
    double airVolumetricSpecificHeat(double myPressure, double myT);
    double saturationVaporPressure(double myTCelsius);
    double saturationSlope(double myTCelsius, double mySatVapPressure);
    double vapourPressureDeficit(double tAir, double relativeHumidity);
    double vaporConcentrationFromPressure(double myPressure, double myT);
    double vaporPressureFromConcentration(double myConcentration, double myT);
    double getAirVaporDeficit(double myT, double myVapor);
    double volumetricLatentHeatVaporization(double myPressure, double myT);
    double latentHeatVaporization(double myTCelsius);
    double psychro(double myPressure, double myTemp);
    double aerodynamicConductance(double heightTemperature, double heightWind, double soilSurfaceTemperature,
                                  double roughnessHeight, double airTemperature, double windSpeed);
    double aerodynamicConductanceOpenwater(double myHeight, double myWaterBodySurface, double myAirTemperature, double myWindSpeed10);
    float erosivityFactor(std::vector<float> values, int nValues);
    float rainIntensity(std::vector<float> values, int nValues, float rainfallThreshold);
    int windPrevailingDir(std::vector<float> intensity, std::vector<float> dir, int nValues, bool useIntensity);
    float timeIntegrationFunction(std::vector<float> values, float timeStep);

#endif // PHYSICS_H
