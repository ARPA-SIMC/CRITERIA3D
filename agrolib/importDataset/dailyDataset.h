#ifndef DAILYDATASET_H
#define DAILYDATASET_H

#include "pointDataset.h"
#include <QDate>

class DailyDataset
{
private:
    QVector<PointDataset> pointDatasetList;
    QDate myDate;
public:
    DailyDataset(QDate date);
    int getPointIndex(double myLat, double myLon, double myZ);
    QDate getDate() const;
    PointDataset* getPointDataset(int pointDatasetListIndex);
};

#endif // DAILYDATASET_H
