#include "elaborationSettings.h"


Crit3DElaborationSettings::Crit3DElaborationSettings()
{
    anomalyPtsMaxDistance = DEF_VALUE_ANOMALY_PTS_MAX_DISTANCE;
    anomalyPtsMaxDeltaZ = DEF_VALUE_ANOMALY_PTS_MAX_DELTA_Z;
    gridMinCoverage = DEF_VALUE_GRID_MIN_COVERAGE;
    automaticTmed = DEF_VALUE_AUTOMATIC_T_MED;
    automaticETP = DEF_VALUE_AUTOMATIC_ETP;
    mergeJointStations = DEF_VALUE_MERGE_JOINT_STATIONS;
}

float Crit3DElaborationSettings::getAnomalyPtsMaxDistance() const
{
    return anomalyPtsMaxDistance;
}

void Crit3DElaborationSettings::setAnomalyPtsMaxDistance(float value)
{
    anomalyPtsMaxDistance = value;
}

float Crit3DElaborationSettings::getAnomalyPtsMaxDeltaZ() const
{
    return anomalyPtsMaxDeltaZ;
}

void Crit3DElaborationSettings::setAnomalyPtsMaxDeltaZ(float value)
{
    anomalyPtsMaxDeltaZ = value;
}

float Crit3DElaborationSettings::getGridMinCoverage() const
{
    return gridMinCoverage;
}

void Crit3DElaborationSettings::setGridMinCoverage(float value)
{
    gridMinCoverage = value;
}

bool Crit3DElaborationSettings::getAutomaticTmed() const
{
    return automaticTmed;
}

void Crit3DElaborationSettings::setAutomaticTmed(bool value)
{
    automaticTmed = value;
}

bool Crit3DElaborationSettings::getAutomaticETP() const
{
    return automaticETP;
}

void Crit3DElaborationSettings::setAutomaticETP(bool value)
{
    automaticETP = value;
}

bool Crit3DElaborationSettings::getMergeJointStations() const
{
    return mergeJointStations;
}

void Crit3DElaborationSettings::setMergeJointStations(bool value)
{
    mergeJointStations = value;
}

