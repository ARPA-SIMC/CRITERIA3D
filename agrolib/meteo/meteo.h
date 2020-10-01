#ifndef METEO_H
#define METEO_H

    #ifndef CRIT3DDATE_H
        #include "crit3dDate.h"
    #endif

    #ifndef STATISTICS_H
        #include "statistics.h"
    #endif

    #ifndef VECTOR_H
        #include <vector>
    #endif

    #ifndef _MAP_
        #include <map>
    #endif


    class Crit3DColorScale;

    #define DEFAULT_MIN_PERCENTAGE 80
    #define DEFAULT_RAINFALL_THRESHOLD 0.2f
    #define DEFAULT_THOM_THRESHOLD 24
    #define DEFAULT_TRANSMISSIVITY_SAMANI 0.17f
    #define DEFAULT_HOURLY_INTERVALS 1
    #define DEFAULT_WIND_INTENSITY 2.0f

    #define TABLE_METEO_POINTS "point_properties"
    #define FIELD_METEO_POINT "id_point"
    #define FIELD_METEO_DATETIME "date"
    #define FIELD_METEO_VARIABLE "id_variable"
    #define FIELD_METEO_VARIABLE_NAME "variable"

    class Crit3DMeteoSettings
    {
    public:
        Crit3DMeteoSettings();

        void initialize();

        float getMinimumPercentage() const;
        void setMinimumPercentage(float value);

        float getRainfallThreshold() const;
        void setRainfallThreshold(float value);

        float getThomThreshold() const;
        void setThomThreshold(float value);

        float getTransSamaniCoefficient() const;
        void setTransSamaniCoefficient(float value);

        int getHourlyIntervals() const;
        void setHourlyIntervals(int value);

        float getWindIntensityDefault() const;
        void setWindIntensityDefault(float value);

    private:
        float minimumPercentage;
        float rainfallThreshold;
        float thomThreshold;
        float transSamaniCoefficient;
        int hourlyIntervals;
        float windIntensityDefault;
    };

    enum lapseRateCodeType {primary, secondary, supplemental};

    enum meteoVariable {airTemperature, dailyAirTemperatureMin, dailyAirTemperatureMax, dailyAirTemperatureAvg, dailyAirTemperatureRange,
                    precipitation, dailyPrecipitation,
                    airRelHumidity, dailyAirRelHumidityMin, dailyAirRelHumidityMax, dailyAirRelHumidityAvg,
                    airDewTemperature, dailyAirDewTemperatureMin, dailyAirDewTemperatureMax, dailyAirDewTemperatureAvg,
                    thom, dailyThomMax, dailyThomAvg, dailyThomHoursAbove, dailyThomDaytime, dailyThomNighttime,
                    globalIrradiance, directIrradiance, diffuseIrradiance, reflectedIrradiance, atmTransmissivity,
                    dailyGlobalRadiation, dailyDirectRadiation, dailyDiffuseRadiation, dailyReflectedRadiation,
                    windScalarIntensity, windVectorIntensity, windVectorDirection, windVectorX, windVectorY,
                    dailyWindVectorIntensityAvg, dailyWindVectorIntensityMax, dailyWindVectorDirectionPrevailing, dailyWindScalarIntensityAvg, dailyWindScalarIntensityMax,
                    leafWetness, dailyLeafWetness, atmPressure,
                    referenceEvapotranspiration, dailyReferenceEvapotranspirationHS, dailyReferenceEvapotranspirationPM, actualEvaporation,
                    dailyBIC,
                    dailyWaterTableDepth,
                    anomaly, noMeteoTerrain, noMeteoVar};


    const std::map<std::string, meteoVariable> MapDailyMeteoVar = {
      { "DAILY_TMIN", dailyAirTemperatureMin },
      { "DAILY_TMAX", dailyAirTemperatureMax },
      { "DAILY_TAVG", dailyAirTemperatureAvg },
      { "DAILY_PREC", dailyPrecipitation },
      { "DAILY_RHMIN", dailyAirRelHumidityMin },
      { "DAILY_RHMAX", dailyAirRelHumidityMax },
      { "DAILY_RHAVG", dailyAirRelHumidityAvg },
      { "DAILY_RAD", dailyGlobalRadiation },        
      { "DAILY_W_VEC_INT_AVG", dailyWindVectorIntensityAvg },
      { "DAILY_W_VEC_DIR_PREV", dailyWindVectorDirectionPrevailing },
      { "DAILY_W_VEC_INT_MAX", dailyWindVectorIntensityMax },
      { "DAILY_W_SCAL_INT_AVG", dailyWindScalarIntensityAvg },
      { "DAILY_W_SCAL_INT_MAX", dailyWindScalarIntensityMax },
      { "DAILY_ET0_HS", dailyReferenceEvapotranspirationHS },
      { "DAILY_ET0_PM", dailyReferenceEvapotranspirationPM },
      { "DAILY_LEAFW", dailyLeafWetness },
      { "DAILY_TEMPRANGE", dailyAirTemperatureRange },
      { "DAILY_AIRDEW_TMIN", dailyAirDewTemperatureMin },
      { "DAILY_AIRDEW_TMAX", dailyAirDewTemperatureMax },
      { "DAILY_AIRDEW_TAVG", dailyAirDewTemperatureAvg },
      { "DAILY_THOMMAX", dailyThomMax },
      { "DAILY_THOMAVG", dailyThomAvg },
      { "DAILY_THOM_HABOVE", dailyThomHoursAbove },
      { "DAILY_THOM_DAYTIME", dailyThomDaytime },
      { "DAILY_THOM_NIGHTTIME", dailyThomNighttime },
      { "DAILY_DIRECT_RAD", dailyDirectRadiation },
      { "DAILY_DIFFUSE_RAD", dailyDiffuseRadiation },
      { "DAILY_REFLEC_RAD", dailyReflectedRadiation },
      { "DAILY_BIC", dailyBIC },
      { "DAILY_WATER_TABLE_DEPTH", dailyWaterTableDepth }
    };

    const std::map<meteoVariable, std::string> MapDailyMeteoVarToString = {
      { dailyAirTemperatureMin , "DAILY_TMIN" },
      { dailyAirTemperatureMax, "DAILY_TMAX"  },
      { dailyAirTemperatureAvg, "DAILY_TAVG"  },
      { dailyPrecipitation, "DAILY_PREC" },
      { dailyAirRelHumidityMin, "DAILY_RHMIN" },
      { dailyAirRelHumidityMax, "DAILY_RHMAX" },
      { dailyAirRelHumidityAvg, "DAILY_RHAVG" },
      { dailyGlobalRadiation, "DAILY_RAD" },
      { dailyWindVectorIntensityAvg, "DAILY_W_VEC_INT_AVG", },
      { dailyWindVectorDirectionPrevailing, "DAILY_W_VEC_DIR_PREV" },
      { dailyWindVectorIntensityMax, "DAILY_W_VEC_INT_MAX" },
      { dailyWindScalarIntensityAvg, "DAILY_W_SCAL_INT_AVG" },
      { dailyWindScalarIntensityMax, "DAILY_W_SCAL_INT_MAX" },
      { dailyReferenceEvapotranspirationHS, "DAILY_ET0_HS" },
      { dailyReferenceEvapotranspirationPM, "DAILY_ET0_PM" },
      { dailyLeafWetness, "DAILY_LEAFW" },
      { dailyAirTemperatureRange, "DAILY_TEMPRANGE" },
      { dailyAirDewTemperatureMin, "DAILY_AIRDEW_TMIN" },
      { dailyAirDewTemperatureMax, "DAILY_AIRDEW_TMAX" },
      { dailyAirDewTemperatureAvg, "DAILY_AIRDEW_TAVG" },
      { dailyThomMax, "DAILY_THOMMAX" },
      { dailyThomAvg, "DAILY_THOMAVG" },
      { dailyThomHoursAbove, "DAILY_THOM_HABOVE" },
      { dailyThomDaytime, "DAILY_THOM_DAYTIME" },
      { dailyThomNighttime, "DAILY_THOM_NIGHTTIME" },
      { dailyDirectRadiation, "DAILY_DIRECT_RAD" },
      { dailyDiffuseRadiation, "DAILY_DIFFUSE_RAD" },
      { dailyReflectedRadiation, "DAILY_REFLEC_RAD" },
      { dailyBIC, "DAILY_BIC" },
      { dailyWaterTableDepth, "DAILY_WATER_TABLE_DEPTH" }
    };

    const std::map<std::string, meteoVariable> MapHourlyMeteoVar = {
      { "TAVG", airTemperature },
      { "PREC", precipitation },
      { "RHAVG", airRelHumidity },
      { "RAD", globalIrradiance },
      { "W_VEC_INT", windVectorIntensity },
      { "W_VEC_DIR", windVectorDirection },
      { "W_SCAL_INT", windScalarIntensity },
      { "W_VEC_X", windVectorX},
      { "W_VEC_Y", windVectorY},
      { "ET0", referenceEvapotranspiration },
      { "LEAFW", leafWetness },
      { "TDAVG", airDewTemperature },
      { "THOM", thom },
      { "DIRECT_RAD", directIrradiance },
      { "DIFFUSE_RAD", diffuseIrradiance },
      { "REFLEC_RAD", reflectedIrradiance },
      { "ATM_TRANSMIT", atmTransmissivity },
      { "ATM_PRESSURE", atmPressure },
      { "ACTUAL_EVAPO", actualEvaporation }
    };

    const std::map<meteoVariable, std::string> MapHourlyMeteoVarToString = {
        { airTemperature, "TAVG" },
        { precipitation, "PREC" },
        { airRelHumidity, "RHAVG" },
        { airDewTemperature, "TDAVG" },
        { globalIrradiance, "RAD" },
        { windVectorIntensity, "W_VEC_INT" },
        { windVectorDirection, "W_VEC_DIR" },
        { windVectorX, "W_VEC_X" },
        { windVectorY, "W_VEC_Y" },
        { windScalarIntensity, "W_SCAL_INT" },
        { referenceEvapotranspiration, "ET0" },
        { leafWetness, "LEAFW" }
    };


    enum frequencyType {hourly, daily, monthly, noFrequency};

    enum surfaceType   {SurfaceTypeWater, SurfaceTypeSoil, SurfaceTypeCrop};

    class Crit3DClimateParameters
    {
    public:
        Crit3DClimateParameters();

        std::vector <float> tmin;
        std::vector <float> tmax;
        std::vector <float> tdmin;
        std::vector <float> tdmax;
        std::vector <float> tminLapseRate;
        std::vector <float> tmaxLapseRate;
        std::vector <float> tdMinLapseRate;
        std::vector <float> tdMaxLapseRate;

        float getClimateLapseRate(meteoVariable myVar, Crit3DTime myTime);
        float getClimateVar(meteoVariable myVar, Crit3DDate myDate, int myHour);
    };

    bool computeWindCartesian(float intensity, float direction, float* u, float* v);
    bool computeWindPolar(float u, float v, float* intensity, float* direction);

    float relHumFromTdew(float dewT, float airT);

    float tDewFromRelHum(float rhAir, float airT);

    double ET0_Penman_hourly(double heigth, double clearSkyIndex, double globalSWRadiation,
                    double airTemp, double airHum, double windSpeed10);

    double ET0_Penman_daily(int myDOY, double myElevation, double myLatitude,
                            double myTmin, double myTmax, double myTminDayAfter,
                            double myUmed, double myVmed10, double mySWGlobRad);

    double ET0_Hargreaves(double KT, double myLat, int myDoy, double tmax, double tmin);

    float computeThomIndex(float temp, float relHum);

    bool setColorScale(meteoVariable variable, Crit3DColorScale *colorScale);

    frequencyType getVarFrequency(meteoVariable myVar);

    std::string getVariableString(meteoVariable myVar);
    std::string getKeyStringMeteoMap(std::map<std::string, meteoVariable> map, meteoVariable value);
    meteoVariable getKeyMeteoVarMeteoMap(std::map<meteoVariable,std::string> map, const std::string &value);
    meteoVariable getMeteoVar(std::string varString);
    meteoVariable getHourlyMeteoVar(std::string varString);
    std::string getMeteoVarName(meteoVariable var);

    bool checkLapseRateCode(lapseRateCodeType myType, bool useLapseRateCode, bool useSupplemental);
    meteoVariable getDailyMeteoVarFromHourly(meteoVariable myVar, aggregationMethod myAggregation);
    meteoVariable updateMeteoVariable(meteoVariable myVar, frequencyType myFreq);


#endif // METEO_H

