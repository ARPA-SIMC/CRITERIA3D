#include "dailyDataset.h"


DailyDataset::DailyDataset(QDate date)
{
    myDate = date;
}

QDate DailyDataset::getDate() const
{
    return myDate;
}

int DailyDataset::getPointIndex(double myLat, double myLon, double myZ){

    for (int i=0; i<pointDatasetList.size(); i++)
    {
        if (pointDatasetList[i].getLat() == myLat && pointDatasetList[i].getLon() == myLon && pointDatasetList[i].getZ() == myZ)
        {
            // point found
            return i;
        }
    }
    // point not found, add new DailyDataset
    PointDataset newPoint(myLat, myLon, myZ);
    pointDatasetList.append(newPoint);
    return (pointDatasetList.size()-1);

}

PointDataset* DailyDataset::getPointDataset(int pointDatasetListIndex)
{
    if (pointDatasetListIndex < pointDatasetList.size())
    {
        return &pointDatasetList[pointDatasetListIndex];
    }
    else
        return nullptr;
}
