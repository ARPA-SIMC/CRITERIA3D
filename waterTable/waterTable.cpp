#include "waterTable.h"
#include "commonConstants.h"

WaterTable::WaterTable()
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
    for (int myMonthIndex = 0; myMonthIndex < 12; myMonthIndex++)
    {
        WTClimateMonthly.push_back(NODATA);
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
        //math.qSplineYearInterpolate .WTClimateMonthly, .WTClimateDaily
    }

    return true;

}

