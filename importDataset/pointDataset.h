#ifndef POINTDATASET_H
#define POINTDATASET_H

#include "varDataset.h"

class PointDataset
{
private:
    QVector<VarDataset> varDatasetList;
    double myLat;
    double myLon;
    double myZ;
public:
    PointDataset();
    PointDataset(double lat, double lon, double z);
    double getLat() const;
    double getLon() const;
    double getZ() const;
    int getVarIndex(QString var);
    VarDataset* getVarDataset(int varDatasetListIndex);
};

#endif // POINTDATASET_H
