#ifndef PHYSICS_H
#define PHYSICS_H

    #ifndef _VECTOR_
        #include <vector>
    #endif

    double PressureFromAltitude(double myHeight);
    double AirMolarDensity(double myPressure, double myT);
    double AirVolumetricSpecificHeat(double myPressure, double myT);
    double SaturationVaporPressure(double myTCelsius);
    double SaturationSlope(double myTCelsius, double mySatVapPressure);
    double vapourPressureDeficit(double tAir, double relativeHumidity);
    double VaporConcentrationFromPressure(double myPressure, double myT);
    double VaporPressureFromConcentration(double myConcentration, double myT);
    double getAirVaporDeficit(double myT, double myVapor);
    double VolumetricLatentHeatVaporization(double myPressure, double myT);
    double LatentHeatVaporization(double myTCelsius);
    double Psychro(double myPressure, double myTemp);
    double AerodynamicConductance(double heightTemperature, double heightWind, double soilSurfaceTemperature,
                                  double roughnessHeight, double airTemperature, double windSpeed);
    double AerodynamicConductanceOpenwater(double myHeight, double myWaterBodySurface, double myAirTemperature, double myWindSpeed10);
    float erosivityFactor(std::vector<float> values, int nValues);
    float rainIntensity(std::vector<float> values, int nValues, float rainfallThreshold);
    int windPrevailingDir(std::vector<float> intensity, std::vector<float> dir, int nValues, bool useIntensity);
    float TimeIntegration(std::vector<float> values, float timeStep);

#endif // PHYSICS_H
