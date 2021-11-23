#include "pointDataset.h"

PointDataset::PointDataset()
{

}

PointDataset::PointDataset(double lat, double lon, double z)
{
    myLat = lat;
    myLon = lon;
    myZ = z;
}

double PointDataset::getLat() const
{
    return myLat;
}

double PointDataset::getLon() const
{
    return myLon;
}

double PointDataset::getZ() const
{
    return myZ;
}

int PointDataset::getVarIndex(QString var){

    for (int i=0; i<varDatasetList.size(); i++)
    {
        if (varDatasetList[i].getVar() == var)
        {
            // var found
            return i;
        }
    }
    // var not found, add new DailyDataset
    VarDataset newVar(var);
    varDatasetList.append(newVar);
    return (varDatasetList.size()-1);

}

VarDataset* PointDataset::getVarDataset(int varDatasetListIndex)
{
    if (varDatasetListIndex < varDatasetList.size())
        {
            return &varDatasetList[varDatasetListIndex];
        }
        else
            return nullptr;
}
