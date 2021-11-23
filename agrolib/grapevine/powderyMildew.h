#ifndef POWDERYMILDEW_H
#define POWDERYMILDEW_H

    #ifndef VECTOR_H
        #include <vector>
    #endif
    #ifndef COMMONCONSTANTS_H
        #include "commonConstants.h"
    #endif

    struct TmildewInput
    {
        float tavg;
        float rain;
        int leafWetness;
        float relativeHumidity;
    };

    struct TmildewOutput
    {
        bool dayInfection;
        bool daySporulation;
        float infectionRate;
        float infectionRisk;
        float aol;
        float col;
    };


    struct TmildewState
    {
        float degreeDays;
        float aic;                              // Ascospores in Chasmothecia
        float currentColonies;                  // total Colony-forming ascospores
        float totalSporulatingColonies;
    };

    struct Tmildew
    {
        TmildewInput input;
        TmildewOutput output;
        TmildewState state;
    };

    void powderyMildew(Tmildew* mildewCore, bool isFirst);
    float computeDegreeDay(float temp);
    float ascosporesReadyFraction(float degreeDay);
    float ascosporeDischargeRate(float temp, float rain, int leafWetness);
    float infectionRate(float temp, float vapourPressure);
    float latencyProgress(float temp);
    float max_vector(const std::vector<float>& v);


#endif // POWDERYMILDEW_H
