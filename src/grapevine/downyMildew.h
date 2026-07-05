#ifndef DOWNYMILDEW_H
#define DOWNYMILDEW_H

    #ifndef VECTOR_H
        #include <vector>
    #endif

    struct TdownyMildewInput
    {
        float tair;
        float rain;
        int leafWetness;
        float relativeHumidity;
    };

    struct TdownyMildewOutput
    {
        float mmo;              // Pool of dormant oospore (Morphologically Mature Oospore)
        bool isInfection;
        float oilSpots;         // oil spots on leaves (the symptom appears at the end of infection)
        float infectionRate;
    };

    struct TdownyMildewState
    {
        int stage;
        float cohort;           // each germinationEvent triggers germination of an oospore cohort
        float rate;             // germination of oospores in stage 1, survival of sporangia in stage 2
        int wetDuration;        // the number of hours of wetness from the start of stage
        float sumT;             // the avg temp from the start of stage
        int nrHours;            // the number of hours since the start of stage
    };

    struct TdownyMildew
    {
        TdownyMildewInput input;
        TdownyMildewOutput output;
        std::vector<TdownyMildewState> state;
        bool isGermination;                 // true when a germination event is active
        float htt;                          // hydro-thermal time value
        float currentPmo;                   // physiologically current mature oospores
    };

    void downyMildew(TdownyMildew* downyMildewCore, bool isFirstJanuary);
    int leafLitterMoisture(float rain, float vpd);
    float hydrothermalTime(float temp, int llm);
    float dormancyBreaking(float htime);
    float survivalRateSporangia(float temp, float relativeHumidity);
    float incubation(float temp);

#endif // DOWNYMILDEW_H

