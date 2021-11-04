#ifndef FORECASTDATASET_H
#define FORECASTDATASET_H

#include <QString>
#include <QList>
#include "dailyDataset.h"

class ForecastDataset
{

private:
    QList<DailyDataset> dailyDatasetList;
public:
    ForecastDataset();
    void importForecastData(QString fileName);
    void addDatasetValue(double lat, double lon, double z, QString var, QDate date, int hour, double value);
    int getDateIndex(QDate myDate);
};

#endif // FORECASTDATASET_H
