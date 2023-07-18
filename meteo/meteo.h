#ifndef METEO_H
#define METEO_H

    #ifndef CRIT3DDATE_H
        #include "crit3dDate.h"
    #endif

    #ifndef STATISTICS_H
        #include "statistics.h"
    #endif

    #include <map>

    #define DEFAULT_MIN_PERCENTAGE 80
    #define DEFAULT_RAINFALL_THRESHOLD 0.2f
    #define DEFAULT_LEAFWETNESS_RH_THRESHOLD 87
    #define DEFAULT_THOM_THRESHOLD 24
    #define DEFAULT_TEMPERATURE_THRESHOLD 30
    #define DEFAULT_TRANSMISSIVITY_SAMANI 0.17f
    #define DEFAULT_HOURLY_INTERVALS 1
    #define DEFAULT_WIND_INTENSITY 2.0f
    #define DEFAULT_AUTOMATIC_TMED true
    #define DEFAULT_AUTOMATIC_ET0HS true
    #define DDHEATING_THRESHOLD 20
    #define DDCOOLING_THRESHOLD 25
    #define DDCOOLING_SUBTRACTION 21

    #define TABLE_METEO_POINTS "point_properties"
    #define FIELD_METEO_POINT "id_point"
    #define FIELD_METEO_DATETIME "date"
    #define FIELD_METEO_VARIABLE "id_variable"
    #define FIELD_METEO_VARIABLE_NAME "variable"

    class Crit3DColorScale;

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

        float getTemperatureThreshold() const;
        void setTemperatureThreshold(float value);

        float getTransSamaniCoefficient() const;
        void setTransSamaniCoefficient(float value);

        int getHourlyIntervals() const;
        void setHourlyIntervals(int value);

        float getWindIntensityDefault() const;
        void setWindIntensityDefault(float value);

        bool getAutomaticTavg() const;
        void setAutomaticTavg(bool value);

        bool getAutomaticET0HS() const;
        void setAutomaticET0HS(bool value);

    private:
        float minimumPercentage;
        float rainfallThreshold;
        float thomThreshold;
        float temperatureThreshold;
        float transSamaniCoefficient;
        int hourlyIntervals;
        float windIntensityDefault;
        bool automaticTavg;
        bool automaticET0HS;
    };

    enum lapseRateCodeType {primary, secondary, supplemental};
    const std::map<lapseRateCodeType, std::string> MapLapseRateCodeToString = {
      { primary , "primary" },
      { secondary, "secondary"  },
      { supplemental, "supplemental"}
    };

    enum meteoVariable {airTemperature, dailyAirTemperatureMin, monthlyAirTemperatureMin, dailyAirTemperatureMax, monthlyAirTemperatureMax,
                    dailyAirTemperatureAvg, monthlyAirTemperatureAvg, dailyAirTemperatureRange,
                    precipitation, dailyPrecipitation, monthlyPrecipitation,
                    airRelHumidity, airDewTemperature, dailyAirRelHumidityMin, dailyAirRelHumidityMax, dailyAirRelHumidityAvg,
                    thom, dailyThomMax, dailyThomAvg, dailyThomHoursAbove, dailyThomDaytime, dailyThomNighttime,dailyTemperatureHoursAbove,
                    globalIrradiance, netIrradiance, directIrradiance, diffuseIrradiance, reflectedIrradiance, atmTransmissivity,
                    dailyGlobalRadiation, monthlyGlobalRadiation, dailyDirectRadiation, dailyDiffuseRadiation, dailyReflectedRadiation,
                    windScalarIntensity, windVectorIntensity, windVectorDirection, windVectorX, windVectorY,
                    dailyWindVectorIntensityAvg, dailyWindVectorIntensityMax, dailyWindVectorDirectionPrevailing, dailyWindScalarIntensityAvg, dailyWindScalarIntensityMax,
                    leafWetness, dailyLeafWetness, atmPressure,
                    referenceEvapotranspiration, dailyReferenceEvapotranspirationHS, monthlyReferenceEvapotranspirationHS, dailyReferenceEvapotranspirationPM, actualEvaporation,
                    dailyBIC, monthlyBIC, dailyHeatingDegreeDays, dailyCoolingDegreeDays,
                    snowWaterEquivalent, snowFall, snowSurfaceTemperature, snowInternalEnergy, snowSurfaceEnergy,
                    snowAge, snowLiquidWaterContent, snowMelt, sensibleHeat, latentHeat,
                    dailyWaterTableDepth,
                    anomaly, elaboration, noMeteoTerrain,
                    noMeteoVar};


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
      { "DAILY_THOMMAX", dailyThomMax },
      { "DAILY_THOMAVG", dailyThomAvg },
      { "DAILY_THOM_HABOVE", dailyThomHoursAbove },
      { "DAILY_THOM_DAYTIME", dailyThomDaytime },
      { "DAILY_THOM_NIGHTTIME", dailyThomNighttime },
      { "DAILY_TEMPERATURE_HABOVE", dailyTemperatureHoursAbove},
      { "DAILY_DIRECT_RAD", dailyDirectRadiation },
      { "DAILY_DIFFUSE_RAD", dailyDiffuseRadiation },
      { "DAILY_REFLEC_RAD", dailyReflectedRadiation },
      { "DAILY_BIC", dailyBIC },
      { "DAILY_DEGREEDAYS_HEATING", dailyHeatingDegreeDays },
      { "DAILY_DEGREEDAYS_COOLING", dailyCoolingDegreeDays },
      { "DAILY_WATER_TABLE_DEPTH", dailyWaterTableDepth },
      { "ELABORATION", elaboration },
      { "ANOMALY", anomaly }
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
      { dailyThomMax, "DAILY_THOMMAX" },
      { dailyThomAvg, "DAILY_THOMAVG" },
      { dailyThomHoursAbove, "DAILY_THOM_HABOVE" },
      { dailyThomDaytime, "DAILY_THOM_DAYTIME" },
      { dailyThomNighttime, "DAILY_THOM_NIGHTTIME" },
      { dailyTemperatureHoursAbove, "DAILY_TEMPERATURE_HABOVE" },
      { dailyDirectRadiation, "DAILY_DIRECT_RAD" },
      { dailyDiffuseRadiation, "DAILY_DIFFUSE_RAD" },
      { dailyReflectedRadiation, "DAILY_REFLEC_RAD" },
      { dailyBIC, "DAILY_BIC" },
      { dailyHeatingDegreeDays, "DAILY_DEGREEDAYS_HEATING" },
      { dailyCoolingDegreeDays, "DAILY_DEGREEDAYS_COOLING" },
      { dailyWaterTableDepth, "DAILY_WATER_TABLE_DEPTH" },
      { elaboration, "ELABORATION" },
      { anomaly, "ANOMALY" }
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
      { "NET_RAD", netIrradiance },
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
        { leafWetness, "LEAFW" },
        { thom, "THOM"},
        { netIrradiance, "NET_RAD"},
        { directIrradiance, "DIRECT_RAD"},
        { diffuseIrradiance, "DIFFUSE_RAD"},
        { reflectedIrradiance, "REFLEC_RAD"},
        { atmTransmissivity, "ATM_TRANSMIT"},
        { atmPressure, "ATM_PRESSURE"},
        { actualEvaporation, "ACTUAL_EVAPO"},
        { snowWaterEquivalent, "SWE"},
        { snowFall, "SNOWFALL"},
        { snowMelt, "SNOWMELT"},
        { snowSurfaceTemperature, "SNOW_SURF_TEMP"},
        { snowSurfaceEnergy, "SNOW_SURF_ENERGY"},
        { snowInternalEnergy, "SNOW_INT_ENERGY"},
        { sensibleHeat, "SENSIBLE_HEAT"},
        { latentHeat, "LATENT_HEAT"}
    };

    const std::map<std::string, meteoVariable> MapMonthlyMeteoVar = {
      { "MONTHLY_TAVG", monthlyAirTemperatureAvg },
      { "MONTHLY_TMIN", monthlyAirTemperatureMin },
      { "MONTHLY_TMAX", monthlyAirTemperatureMax },
      { "MONTHLY_PREC", monthlyPrecipitation },
      { "MONTHLY_ET0_HS", monthlyReferenceEvapotranspirationHS },
      { "MONTHLY_BIC", monthlyBIC },
      { "MONTHLY_RAD", monthlyGlobalRadiation }
    };

    const std::map<meteoVariable, std::string> MapMonthlyMeteoVarToString = {
        { monthlyAirTemperatureAvg, "MONTHLY_TAVG"} ,
        { monthlyAirTemperatureMin, "MONTHLY_TMIN" },
        { monthlyAirTemperatureMax, "MONTHLY_TMAX" },
        { monthlyPrecipitation, "MONTHLY_PREC" },
        { monthlyReferenceEvapotranspirationHS, "MONTHLY_ET0_HS" },
        { monthlyBIC, "MONTHLY_BIC" },
        { monthlyGlobalRadiation, "MONTHLY_RAD" }
    };

    const std::map<std::vector<meteoVariable>, std::string> MapVarUnit = {
        { {dailyAirTemperatureMin,airTemperature,monthlyAirTemperatureMin}, "°C"} ,
        { {dailyAirTemperatureMax,monthlyAirTemperatureMax}, "°C"} ,
        { {dailyAirTemperatureAvg,dailyAirTemperatureRange,monthlyAirTemperatureAvg}, "°C"} ,
        { {dailyPrecipitation,precipitation,monthlyPrecipitation}, "mm"} ,
        { {dailyReferenceEvapotranspirationHS,dailyReferenceEvapotranspirationPM,referenceEvapotranspiration,monthlyReferenceEvapotranspirationHS}, "mm"} ,
        { {dailyAirRelHumidityMin,dailyAirRelHumidityMax,dailyAirRelHumidityAvg,airRelHumidity}, "%"} ,
        { {dailyGlobalRadiation,monthlyGlobalRadiation}, "MJ m-2"} ,
        { {globalIrradiance,netIrradiance,directIrradiance,diffuseIrradiance,reflectedIrradiance}, "W m-2"} ,
        { {dailyBIC,monthlyBIC}, "mm"} ,
        { {dailyWindScalarIntensityAvg,dailyWindVectorIntensityAvg,dailyWindScalarIntensityMax, windScalarIntensity}, "m s-1"} ,
        { {dailyWindVectorDirectionPrevailing, dailyWindVectorIntensityMax, windVectorDirection}, "deg"} ,
        { {windVectorIntensity, windVectorX, windVectorY}, "m s-1"} ,
        { {dailyLeafWetness,leafWetness}, "h"} ,
        { {dailyHeatingDegreeDays,dailyCoolingDegreeDays}, "°D"} ,
        { {airRelHumidity,dailyAirRelHumidityMin,dailyAirRelHumidityMax,dailyAirRelHumidityAvg}, "%"} ,
        { {airDewTemperature}, "°C"} ,
        { {dailyThomAvg,dailyThomDaytime,dailyThomNighttime,thom}, "-"} ,
        { {dailyWaterTableDepth,snowWaterEquivalent,snowFall,snowMelt,snowLiquidWaterContent}, "mm"} ,
        { {snowSurfaceTemperature}, "°C"} ,
        { {snowInternalEnergy,snowSurfaceEnergy,sensibleHeat,latentHeat}, "kJ m-2"} ,
    };


    enum frequencyType {hourly, daily, monthly, noFrequency};

    enum surfaceType   {SurfaceTypeWater, SurfaceTypeSoil, SurfaceTypeCrop};

    enum droughtIndex {INDEX_SPI, INDEX_SPEI, INDEX_DECILES};

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
        float getClimateLapseRate(meteoVariable myVar, int month);
        float getClimateVar(meteoVariable myVar, int month, float height, float refHeight);
    };

    bool computeWindCartesian(float intensity, float direction, float* u, float* v);
    bool computeWindPolar(float u, float v, float* intensity, float* direction);

    float relHumFromTdew(float Td, float T);
    float tDewFromRelHum(float RH, float T);

    double tDewFromRelHum(double RH, double T);
    double tDewFromRelHum(double RH, double T);

    bool computeLeafWetness(double prec, double relHumidity, short* leafW);

    double ET0_Penman_hourly(double heigth, double clearSkyIndex, double globalIrradiance,
                    double airTemp, double airHum, double windSpeed10);

    double ET0_Penman_hourly_net_rad(double heigth, double netRadiation, double airTemp, double airHum, double windSpeed10);

    double ET0_Penman_daily(int myDOY, double myElevation, double myLatitude,
                            double myTmin, double myTmax, double myTminDayAfter,
                            double myUmed, double myVmed10, double mySWGlobRad);

    double ET0_Hargreaves(double KT, double myLat, int myDoy, double tmax, double tmin);

    float computeThomIndex(float temp, float relHum);

    bool setColorScale(meteoVariable variable, Crit3DColorScale *colorScale);

    frequencyType getVarFrequency(meteoVariable myVar);

    std::string getVariableString(meteoVariable myVar);
    std::string getUnitFromVariable(meteoVariable var);
    std::string getKeyStringMeteoMap(std::map<std::string, meteoVariable> map, meteoVariable value);
    meteoVariable getKeyMeteoVarMeteoMap(std::map<meteoVariable,std::string> map, const std::string &value);
    meteoVariable getKeyMeteoVarMeteoMapWithoutUnderscore(std::map<meteoVariable,std::string> map, const std::string& value);
    meteoVariable getMeteoVar(std::string varString);
    meteoVariable getHourlyMeteoVar(std::string varString);
    std::string getMeteoVarName(meteoVariable var);
    std::string getLapseRateCodeName(lapseRateCodeType code);

    bool checkLapseRateCode(lapseRateCodeType myType, bool useLapseRateCode, bool useSupplemental);
    meteoVariable getDailyMeteoVarFromHourly(meteoVariable myVar, aggregationMethod myAggregation);
    meteoVariable updateMeteoVariable(meteoVariable myVar, frequencyType myFreq);


#endif // METEO_H

