#ifndef ELABORATIONSETTINGS_H
#define ELABORATIONSETTINGS_H

    #define DEF_VALUE_ANOMALY_PTS_MAX_DISTANCE 3000
    #define DEF_VALUE_ANOMALY_PTS_MAX_DELTA_Z 50
    #define DEF_VALUE_GRID_MIN_COVERAGE 0
    #define DEF_VALUE_AUTOMATIC_T_MED true
    #define DEF_VALUE_AUTOMATIC_ETP true
    #define DEF_VALUE_MERGE_JOINT_STATIONS true


    class Crit3DElaborationSettings
    {
    public:
        Crit3DElaborationSettings();

        float getAnomalyPtsMaxDistance() const;
        void setAnomalyPtsMaxDistance(float value);

        float getAnomalyPtsMaxDeltaZ() const;
        void setAnomalyPtsMaxDeltaZ(float value);

        float getGridMinCoverage() const;
        void setGridMinCoverage(float value);

        bool getAutomaticTmed() const;
        void setAutomaticTmed(bool value);

        bool getAutomaticETP() const;
        void setAutomaticETP(bool value);

        bool getMergeJointStations() const;
        void setMergeJointStations(bool value);

    private:
        float anomalyPtsMaxDistance;
        float anomalyPtsMaxDeltaZ;
        float gridMinCoverage;
        bool automaticTmed;
        bool automaticETP;
        bool mergeJointStations;

    };

#endif // ELABORATIONSETTINGS_H
