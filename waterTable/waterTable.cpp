#include "waterTable.h"
#include "commonConstants.h"
#include "weatherGenerator.h"

WaterTable::WaterTable(Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints, Crit3DMeteoGrid* meteoGrid, bool isMeteoGridLoaded, Crit3DMeteoSettings meteoSettings, gis::Crit3DGisSettings gisSettings)
    : meteoPoints(meteoPoints), nrMeteoPoints(nrMeteoPoints), meteoGrid(meteoGrid), isMeteoGridLoaded(isMeteoGridLoaded), meteoSettings(meteoSettings), gisSettings(gisSettings)
{

}

QDate WaterTable::getFirstDateWell()
{
    firstDateWell = well.getFirstDate();
    return firstDateWell;
}

QDate WaterTable::getLastDateWell()
{
    lastDateWell = well.getLastDate();
    return lastDateWell;
}

QString WaterTable::getError() const
{
    return error;
}

void WaterTable::initializeWaterTable(Well myWell)
{

    this->well = myWell;
    getFirstDateWell();
    getLastDateWell();
    for (int myMonthIndex = 0; myMonthIndex < 12; myMonthIndex++)
    {
        WTClimateMonthly[myMonthIndex] = NODATA;
    }

    isMeteoPointLinked = false;
    isCWBEquationReady = false;
    isClimateReady = false;

    alpha = NODATA;
    h0 = NODATA;
    R2 = NODATA;
    nrDaysPeriod = NODATA;
    nrObsData = 0;
    EF = NODATA;
    NASH = NODATA;
    RMSE = NODATA;

}

bool WaterTable::computeWaterTable(Well myWell, int maxNrDays, int doy1, int doy2)
{
    if (myWell.getDepthNr() == 0)
    {
        error = "No WaterTable data loaded.";
        return false;
    }

    initializeWaterTable(myWell);
    bool isClimateReady = computeWTClimate();
    isMeteoPointLinked = assignNearestMeteoPoint();
    if (isMeteoPointLinked == false)
    {
        return false;
    }

    if (!computeETP_allSeries())
    {
        return false;
    }

    isCWBEquationReady = computeCWBCorrelation(maxNrDays);
//          If Not .isCWBEquationReady Then Exit Function

//                  computeWaterTableIndices myWell, doy1, doy2

//                  End With

    return true;
}

bool WaterTable::computeWTClimate()
{
    if (well.getDepthNr() < 3)
    {
        error = "Missing data";
        return false;
    }

    std::vector<float> H_sum;
    std::vector<float> H_num;
    for (int myMonthIndex = 0; myMonthIndex < 12; myMonthIndex++)
    {
        H_sum.push_back(0);
        H_num.push_back(0);
    }

    QMap<QDate, int> myDepths = well.getDepths();
    QMapIterator<QDate, int> it(myDepths);
    while (it.hasNext())
    {
        QDate myDate = it.key();
        int myValue = it.value();
        int myMonth = myDate.month();
        int myMonthIndex = myMonth - 1;
        H_sum[myMonthIndex] = H_sum[myMonthIndex] + myValue;
        H_num[myMonthIndex] = H_num[myMonthIndex] + 1;
    }

    for (int myMonthIndex = 0; myMonthIndex < 12; myMonthIndex++)
    {
        if (H_num[myMonthIndex] < 2)
        {
            error = "Missing watertable data: month " + QString::number(myMonthIndex+1);
            return false;
        }
        WTClimateMonthly[myMonthIndex] = H_sum[myMonthIndex] / H_num[myMonthIndex];
        isClimateReady = true;
        cubicSplineYearInterpolate(WTClimateMonthly, WTClimateDaily);
    }
    return true;
}

bool WaterTable::assignNearestMeteoPoint()
{
    float minimumDistance = NODATA;
    bool assignNearestMeteoPoint = false;
    if (isMeteoGridLoaded)
    {
        int zoneNumber;
        for (unsigned row = 0; row < unsigned(meteoGrid->gridStructure().header().nrRows); row++)
        {
            for (unsigned col = 0; col < unsigned(meteoGrid->gridStructure().header().nrCols); col++)
            {
                double utmX = meteoGrid->meteoPointPointer(row,col)->point.utm.x;
                double utmY = meteoGrid->meteoPointPointer(row,col)->point.utm.y;
                if (utmX == NODATA || utmY == NODATA)
                {
                    double lat = meteoGrid->meteoPointPointer(row,col)->latitude;
                    double lon = meteoGrid->meteoPointPointer(row,col)->longitude;
                    gis::latLonToUtm(lat, lon, &utmX, &utmY, &zoneNumber);
                }
                float myDistance = gis::computeDistance(well.getUtmX(), well.getUtmY(), utmX, utmY);
                if (myDistance < MAXWELLDISTANCE )
                {
                    if (myDistance < minimumDistance || minimumDistance == NODATA)
                    {
                        if (assignWTMeteoData(*meteoGrid->meteoPointPointer(row,col) ))
                        {
                            minimumDistance = myDistance;
                            assignNearestMeteoPoint = true;
                            linkedMeteoPoint = (*meteoGrid->meteoPointPointer(row,col));
                        }
                    }
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < nrMeteoPoints; i++)
        {

            double utmX = meteoPoints[i].point.utm.x;
            double utmY = meteoPoints[i].point.utm.y;
            float myDistance = gis::computeDistance(well.getUtmX(), well.getUtmY(), utmX, utmY);
            if (myDistance < MAXWELLDISTANCE )
            {
                if (myDistance < minimumDistance || minimumDistance == NODATA)
                {
                    if (assignWTMeteoData(meteoPoints[i]))
                    {
                        minimumDistance = myDistance;
                        assignNearestMeteoPoint = true;
                        linkedMeteoPoint = meteoPoints[i];
                    }
                }
            }
        }
    }
    return assignNearestMeteoPoint;
}

bool WaterTable::assignWTMeteoData(Crit3DMeteoPoint point)
{
    firstMeteoDate = firstDateWell.addDays(-730); // necessari 24 mesi di dati meteo precedenti il primo dato di falda
    lastMeteoDate = lastDateWell; // ultimo dato di falda CHECK lastDate = max(currentDay, myWell.Well.Items.Item(UBound(myWell.Well.Items.Item)).date_)
    float precPerc = point.getPercValueVariable(Crit3DDate(firstMeteoDate.day(), firstMeteoDate.month(), firstMeteoDate.year()) , Crit3DDate(lastMeteoDate.day(), lastMeteoDate.month(), lastMeteoDate.year()), dailyPrecipitation);
    float tMinPerc = point.getPercValueVariable(Crit3DDate(firstMeteoDate.day(), firstMeteoDate.month(), firstMeteoDate.year()) , Crit3DDate(lastMeteoDate.day(), lastMeteoDate.month(), lastMeteoDate.year()), dailyAirTemperatureMin);
    float tMaxPerc = point.getPercValueVariable(Crit3DDate(firstMeteoDate.day(), firstMeteoDate.month(), firstMeteoDate.year()) , Crit3DDate(lastMeteoDate.day(), lastMeteoDate.month(), lastMeteoDate.year()), dailyAirTemperatureMax);

    float minPercentage = meteoSettings.getMinimumPercentage();
    if (precPerc > minPercentage/100 && tMinPerc > minPercentage/100 && tMaxPerc > minPercentage/100)
    {
        return true;
    }
    else
    {
        error = "Not enough meteo data to analyze watertable period. Try to decrease the required percentage";
        return false;
    }
}

bool WaterTable::computeETP_allSeries()
{
    double myLat;
    double myLon;
    gis::getLatLonFromUtm(gisSettings, well.getUtmX(), well.getUtmY(), &myLat, &myLon);
    double sumCWB = 0;
    int nrValidDays = 0;
    for (QDate myDate = firstMeteoDate; myDate<=lastMeteoDate; myDate=myDate.addDays(1))
    {
        Crit3DDate date(myDate.day(), myDate.month(), myDate.year());
        float Tmin = linkedMeteoPoint.getMeteoPointValueD(date, dailyAirTemperatureMin);
        float Tmax = linkedMeteoPoint.getMeteoPointValueD(date, dailyAirTemperatureMax);
        float prec = linkedMeteoPoint.getMeteoPointValueD(date, dailyPrecipitation);
        float etp = dailyEtpHargreaves(Tmin, Tmax, date, myLat,&meteoSettings);
        if (etp != NODATA && prec != NODATA)
        {
            sumCWB = sumCWB + (prec - etp);
            nrValidDays = nrValidDays + 1;
        }
    }

    if (nrValidDays > 0)
    {
        avgDailyCWB = sumCWB / nrValidDays;
    }
    else
    {
        error = "Missing data: " + QString::fromStdString(linkedMeteoPoint.name);
        return false;
    }

    return true;
}

bool WaterTable::computeCWBCorrelation(int maxNrDays)
{
    float bestR2 = 0;
    int bestNrDays = NODATA;
    QMap<QDate, int> myDepths = well.getDepths();

    for (int nrDays = 90; nrDays <= maxNrDays; nrDays=nrDays+10)
    {
        QMapIterator<QDate, int> it(myDepths);
        while (it.hasNext())
        {
            QDate myDate = it.key();
            int myValue = it.value();
            //float myCWBValue = computeCWB(myDate, nrDays);  // [cm]
            // TO DO
        }
    }
}

double WaterTable::computeCWB(QDate myDate, int nrDays)
{
    // TO DO
    return NODATA;
}

