#ifndef DROUGHT_H
#define DROUGHT_H

#ifndef METEOPOINT_H
    #include "meteoPoint.h"
#endif

//SPI Gamma Distribution
struct gammaParam {
    double beta;
    double gamma;
    double pzero;
};

// SPEI Log-Logistic Distribution
struct logLogisticParam {
    double alpha;
    double beta;
    double gamma;
};

class Drought
{
public:
    Drought(droughtIndex index, int firstYear, int lastYear, Crit3DDate date, Crit3DMeteoPoint* meteoPoint, Crit3DMeteoSettings* meteoSettings);

    droughtIndex getIndex() const;
    void setIndex(const droughtIndex &value);

    int getTimeScale() const;
    void setTimeScale(int value);

    int getFirstYear() const;
    void setFirstYear(int value);

    int getLastYear() const;
    void setLastYear(int value);

    bool getComputeAll() const;
    void setComputeAll(bool value);

    float computeDroughtIndex();
    bool computeSpiParameters();
    bool computeSpeiParameters();
    bool computePercentileValuesCurrentDay();

    void setMeteoPoint(Crit3DMeteoPoint *value);
    Crit3DMeteoSettings *getMeteoSettings() const;
    Crit3DDate getDate() const;
    void setDate(const Crit3DDate &value);
    float getCurrentPercentileValue() const;
    void setMyVar(const meteoVariable &value);

private:
    Crit3DMeteoPoint* meteoPoint;
    Crit3DMeteoSettings* meteoSettings;
    Crit3DDate date;
    droughtIndex index;
    meteoVariable myVar;
    int timeScale;
    int firstYear;
    int lastYear;
    bool computeAll;
    gammaParam gammaStruct;
    logLogisticParam logLogisticStruct;
    std::vector<gammaParam> currentGamma;
    std::vector<logLogisticParam> currentLogLogistic;
    std::vector<float> droughtResults;
    float currentPercentileValue;

};

#endif // DROUGHT_H



