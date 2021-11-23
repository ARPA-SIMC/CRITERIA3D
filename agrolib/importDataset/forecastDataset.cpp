#include "forecastDataset.h"
#include "commonConstants.h"
#include <QTextStream>
#include <QDate>
#include <QDebug>
#include <QFile>

ForecastDataset::ForecastDataset()
{

}

int ForecastDataset::getDateIndex(QDate myDate){

    for (int i=0; i<dailyDatasetList.size(); i++)
    {
        if (dailyDatasetList[i].getDate() == myDate)
        {
            // date found
            return i;
        }
    }
    // date not found, add new DailyDataset
    DailyDataset newDataset(myDate);
    dailyDatasetList.append(newDataset);
    return (dailyDatasetList.size()-1);

}

void ForecastDataset::importForecastData(QString fileName)
{
    QFile inputFile(fileName);
    if (inputFile.open(QIODevice::ReadOnly))
    {
       QTextStream in(&inputFile);
       while (!in.atEnd())
       {
          QString line = in.readLine();
          QStringList fields = line.split(",");
          double myLat = fields[0].toDouble();
          double myLon = fields[1].toDouble();
          double myZ = fields[2].toDouble();
          QString myVar = fields[3];
          QDate myDate(fields[4].toInt(), fields[5].toInt(), fields[6].toInt());
          int myHour = fields[7].toInt();
          double myValue = fields[8].toDouble();
          addDatasetValue(myLat, myLon, myZ, myVar, myDate, myHour, myValue);
          if (myHour == 0 && myDate > dailyDatasetList[0].getDate())
          {
            // hour 00 - copy 24 day before
            addDatasetValue(myLat, myLon, myZ, myVar, myDate.addDays(-1), 24, myValue);
          }

       }
       inputFile.close();
    }
    // test
    /*
    qDebug() << "date: " << dailyDatasetList[0].getDate();
    qDebug() << "lat: " << dailyDatasetList[0].getPointDataset(0)->getLat();
    qDebug() << "lon: " << dailyDatasetList[0].getPointDataset(0)->getLon();
    qDebug() << "z: " << dailyDatasetList[0].getPointDataset(0)->getZ();

    qDebug() << "var: " << dailyDatasetList[0].getPointDataset(0)->getVarDataset(0)->getVar();
    qDebug() << "value: " << dailyDatasetList[0].getPointDataset(0)->getVarDataset(0)->getHourlyValue(0);

    qDebug() << "date: " << dailyDatasetList[0].getDate();
    qDebug() << "lat: " << dailyDatasetList[0].getPointDataset(0)->getLat();
    qDebug() << "lon: " << dailyDatasetList[0].getPointDataset(0)->getLon();
    qDebug() << "z: " << dailyDatasetList[0].getPointDataset(0)->getZ();

    qDebug() << "var: " << dailyDatasetList[0].getPointDataset(0)->getVarDataset(2)->getVar();
    qDebug() << "value: " << dailyDatasetList[0].getPointDataset(0)->getVarDataset(2)->getHourlyValue(0);
    */

}

void ForecastDataset::addDatasetValue(double lat, double lon, double z, QString var, QDate date, int hour, double value)
{
    int indexDay = getDateIndex(date);
    int indexPoint = dailyDatasetList[indexDay].getPointIndex(lat, lon, z);
    int indexVar = dailyDatasetList[indexDay].getPointDataset(indexPoint)->getVarIndex(var);
    int indexHour = dailyDatasetList[indexDay].getPointDataset(indexPoint)->getVarDataset(indexVar)->getHourlyValueIndex(value);
    if (dailyDatasetList[indexDay].getPointDataset(indexPoint)->getVarDataset(indexVar)->getHourlyValue(indexHour) == NODATA)
    {
        dailyDatasetList[indexDay].getPointDataset(indexPoint)->getVarDataset(indexVar)->setHourlyValue(indexHour, value);
    }
}
