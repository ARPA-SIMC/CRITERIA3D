#ifndef QUALITY_H
#define QUALITY_H

    #ifndef METEO_H
        #include "meteo.h"
    #endif

    // default
    #define DEF_VALUE_REF_HEIGHT 300
    #define DEF_VALUE_DELTA_T_SUSP 13
    #define DEF_VALUE_DELTA_T_WRONG 26
    #define DEF_VALUE_REL_HUM_TOLERANCE 102


    class Crit3DMeteoPoint;

    namespace quality
    {
        enum qualityType {missing_data, wrong_syntactic, wrong_spatial, wrong_variable, accepted};

        class Range {
            private:
                float max;
                float min;
            public:
                Range();
                Range(float myMin, float myMax);

                float getMin();
                float getMax();
        };
    }


    class Crit3DQuality {

    private:
        quality::Range* qualityHourlyT;
        quality::Range* qualityHourlyTd;
        quality::Range* qualityHourlyP;
        quality::Range* qualityHourlyRH;
        quality::Range* qualityHourlyWInt;
        quality::Range* qualityHourlyWDir;
        quality::Range* qualityHourlyGIrr;
        quality::Range* qualityTransmissivity;
        quality::Range* qualityHourlyET0;
        quality::Range* qualityHourlyleafWetness;

        quality::Range* qualityDailyT;
        quality::Range* qualityDailyP;
        quality::Range* qualityDailyRH;
        quality::Range* qualityDailyWInt;
        quality::Range* qualityDailyWDir;
        quality::Range* qualityDailyGRad;
        quality::Range* qualityDailyET0;

        float referenceHeight;
        float deltaTSuspect;
        float deltaTWrong;
        float relHumTolerance;


    public:

        Crit3DQuality();

        void initialize();

        quality::Range* getQualityRange(meteoVariable myVar);

        void syntacticQualityControl(meteoVariable myVar, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints);

        quality::qualityType syntacticQualitySingleValue(meteoVariable myVar, float myValue);

        float getReferenceHeight() const;

        void setReferenceHeight(float value);

        float getDeltaTSuspect() const;

        void setDeltaTSuspect(float value);

        float getDeltaTWrong() const;

        void setDeltaTWrong(float value);

        float getRelHumTolerance() const;

        void setRelHumTolerance(float value);

        quality::qualityType checkFastValueDaily_SingleValue(meteoVariable myVar, Crit3DClimateParameters *climateParam, float myValue, int month, float height);

        bool wrongValueDaily_SingleValue(meteoVariable myVar, Crit3DClimateParameters *climateParam, float myValue, int month, float height);

        quality::qualityType checkFastValueHourly_SingleValue(meteoVariable myVar, Crit3DClimateParameters* climateParam, float myValue, int month, float height);

        bool wrongValueHourly_SingleValue(meteoVariable myVar, Crit3DClimateParameters* climateParam, float myValue, int month, float height);

    };


#endif // QUALITY_H
