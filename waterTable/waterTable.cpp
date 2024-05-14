#include "waterTable.h"
#include "commonConstants.h"
#include "weatherGenerator.h"

WaterTable::WaterTable(Crit3DMeteoPoint* meteoPoints, int nrMeteoPoints, Crit3DMeteoGrid* meteoGrid, bool isMeteoGridLoaded)
    : meteoPoints(meteoPoints), nrMeteoPoints(nrMeteoPoints), meteoGrid(meteoGrid), isMeteoGridLoaded(isMeteoGridLoaded)
{

}

QDate WaterTable::getFirstDate()
{
    firstDate = well.getFirstDate();
    return firstDate;
}

QDate WaterTable::getLastDate()
{
    lastDate = well.getLastDate();
    return lastDate;
}

QString WaterTable::getError() const
{
    return error;
}

void WaterTable::initializeWaterTable(Well myWell)
{

    this->well = myWell;
    getFirstDate();
    getLastDate();
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

//          If Not .Well.Items.isLoaded Then Exit Function

//              .isMeteoPointLinked = assignNearestMeteoPoint(myWell, d1, d2)
//          If Not .isMeteoPointLinked Then
//              PragaShell.setErrorMsg "Missing near weather data"
//          Exit Function
//              End If

//                      computeETP_allSeries myWell
//                          .isCWBEquationReady = ComputeCWBCorrelation(myWell, maxNrDays, True)
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
    for (auto it = myDepths.keyValueBegin(); it != myDepths.keyValueEnd(); ++it)
    {
        QDate myDate = it->first;
        int myValue = it->second;
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
                            well.setLinkedMeteoPoint(*meteoGrid->meteoPointPointer(row,col));
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
                        well.setLinkedMeteoPoint(meteoPoints[i]);
                    }
                }
            }
        }
    }
    return assignNearestMeteoPoint;
}

bool WaterTable::assignWTMeteoData(Crit3DMeteoPoint point)
{
    // TO DO
    return true;
}

