#ifndef QUALITY_H
#define QUALITY_H

    #ifndef METEO_H
        #include "meteo.h"
    #endif
    #ifndef COMMONCONSTANTS_H
        #include "commonConstants.h"
    #endif

    // default
    // [m]
    #define DEF_VALUE_REF_HEIGHT 300
    // [°C]
    #define DEF_VALUE_DELTA_T_SUSP 13
    #define DEF_VALUE_DELTA_T_WRONG 26
    // [%]
    #define DEF_VALUE_REL_HUM_TOLERANCE 102
    // [cm]
    #define DEF_VALUE_WATERTABLE_MAX_DEPTH 300


    class Crit3DMeteoPoint;

    namespace quality
    {
        enum qualityType {missing_data, wrong_syntactic, wrong_spatial, wrong_variable, accepted};

        class Range {
            private:
                float _min, _max;

            public:
                Range() { _min = NODATA; _max = NODATA; }

                Range(float min, float max): _min(min), _max(max) { }

                float getMin() { return _min; }
                float getMax() { return _max; }
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
        quality::Range* qualityDailyBIC;

        float referenceHeight;
        float deltaTSuspect;
        float deltaTWrong;
        float relHumTolerance;
        float waterTableMaximumDepth;

    public:

        Crit3DQuality();
        ~Crit3DQuality();

        void initialize();

        quality::Range* getQualityRange(meteoVariable myVar);

        void syntacticQualityControl(meteoVariable myVar, Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints);

        quality::qualityType syntacticQualitySingleValue(meteoVariable myVar, float myValue);

        float getReferenceHeight() const { return referenceHeight; }
        void setReferenceHeight(float value) { referenceHeight = value; }

        float getDeltaTSuspect() const { return deltaTSuspect; }
        void setDeltaTSuspect(float value) { deltaTSuspect = value; }

        float getDeltaTWrong() const { return deltaTWrong; }
        void setDeltaTWrong(float value) { deltaTWrong = value; }

        float getRelHumTolerance() const { return relHumTolerance; }
        void setRelHumTolerance(float value) { relHumTolerance = value; }

        float getWaterTableMaximumDepth() const { return waterTableMaximumDepth; }
        void setWaterTableMaximumDepth(float value) { waterTableMaximumDepth = value; }

        quality::qualityType checkFastValueDaily_SingleValue(meteoVariable myVar, Crit3DClimateParameters *climateParam, float myValue, int month, float height);

        bool wrongValueDaily_SingleValue(meteoVariable myVar, Crit3DClimateParameters *climateParam, float myValue, int month, float height);

        quality::qualityType checkFastValueHourly_SingleValue(meteoVariable myVar, Crit3DClimateParameters* climateParam, float myValue, int month, float height);

        bool wrongValueHourly_SingleValue(meteoVariable myVar, Crit3DClimateParameters* climateParam, float myValue, int month, float height);

    };


#endif // QUALITY_H
