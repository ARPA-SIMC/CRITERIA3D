#ifndef QUALITY_H
#define QUALITY_H

    #ifndef METEO_H
        #include "meteo.h"
    #endif

    // TODO: move in options/quality db
    #define DEF_VALUE_REF_HEIGHT 300
    #define DEF_VALUE_DELTA_T_SUSP 26
    #define DEF_VALUE_DELTA_T_WRONG 13
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

        quality::Range* qualityDailyT;
        quality::Range* qualityDailyP;
        quality::Range* qualityDailyRH;
        quality::Range* qualityDailyWInt;
        quality::Range* qualityDailyWDir;
        quality::Range* qualityDailyGRad;
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
    };


#endif // QUALITY_H
