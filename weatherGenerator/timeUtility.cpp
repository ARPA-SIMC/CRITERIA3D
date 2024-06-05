#include <QDebug>
#include <QDate>
#include <QString>

#include "crit3dDate.h"
#include "timeUtility.h"


int getMonthsInPeriod(int month1, int month2)
{
    int monthsInPeriod = 0;

    if (month2 >= month1)
    {
        // regular period
        monthsInPeriod = (month2 - month1 + 1);
    }
    else
    {
        // not regular period (between years)
        monthsInPeriod = (12 - month1 + 1) + month2;
    }

    return monthsInPeriod;
}


// it checks if observed data includes the last 9 months before wgDoy_1
// it updates wgDoy_1 and nr_DaysBefore if some days are missing (lower than NRDAYSTOLERANCE)
bool checkLastYearDate(const Crit3DDate &inputFirstDate, const Crit3DDate &inputLastDate, int dataLength,
                       int myPredictionYear, int &wgDoy1, int &nrDaysBefore)
{
    Crit3DDate predictionFirstDate = getDateFromDoy (myPredictionYear, wgDoy1);

    if (inputLastDate.addDays(NRDAYSTOLERANCE+1) <  predictionFirstDate)
    {
        qDebug() << "\nObserved days missing are more than NRDAYSTOLERANCE" << NRDAYSTOLERANCE << "\n";
        return false;
    }

    int predictionMonth = predictionFirstDate.month;
    int monthIndex = 0;
    nrDaysBefore = 0;
    for (int i = 0; i < 9; i++)
    {
        monthIndex = (predictionMonth -1) -i;
        if (monthIndex <= 0)
        {
            monthIndex = monthIndex + 12 ;
            myPredictionYear = myPredictionYear - 1;
        }
        nrDaysBefore += getDaysInMonth(monthIndex, myPredictionYear);
    }

    // shift wgDoy1 if there are missing data
    if (inputLastDate < predictionFirstDate)
    {
        int delta = difference(inputLastDate, predictionFirstDate) - 1;
        wgDoy1 -= delta;
        nrDaysBefore -= delta;
    }

    // use or not the observed data in the forecast period
    else if (USEDATA)
    {
        if (inputLastDate > predictionFirstDate.addDays(80))
        {
            qDebug() << "Check your XML: you have already all observed data" << "\n";
            return false;
        }
        if (isLeapYear(predictionFirstDate.year))
        {
            wgDoy1 = (wgDoy1 + (difference(predictionFirstDate, inputLastDate)) + 1 ) % 366;
        }
        else
        {
            wgDoy1 = (wgDoy1 + (difference(predictionFirstDate, inputLastDate)) + 1 ) % 365;
        }

        nrDaysBefore += (difference(predictionFirstDate, inputLastDate)) + 1 ;
    }

    if ( difference(inputFirstDate, predictionFirstDate) < nrDaysBefore || dataLength < (nrDaysBefore-NRDAYSTOLERANCE) )
    {
        // observed data does not include 9 months before wgDoy1 or more than NRDAYSTOLERANCE days missing
        return false;
    }

    return true;
}


bool getDoyFromSeason(const QString &season, int myPredictionYear, int &wgDoy1, int &wgDoy2)
{
    QString period[12] = {"JFM","FMA","MAM","AMJ","MJJ","JJA","JAS","ASO","SON","OND","NDJ","DJF"};
    int i = 0;
    int found = 0;

    for (i = 0; i < 12; i++)
    {
        if (season.compare(period[i]) == 0)
        {
            found = 1;
            break;
        }
    }
    if (found == 0)
    {
        qDebug() << "Wrong season" ;
        return false;
    }

    int month1 = i + 1;             // first month of my season
    int month2 = (i + 3) % 12;      // last month of my season
    if (month2 == 0) month2 = 12;

    Crit3DDate predictionFirstDate;
    predictionFirstDate.year = myPredictionYear;
    predictionFirstDate.month = month1;
    predictionFirstDate.day = 1;

    wgDoy1 = getDoyFromDate(predictionFirstDate);

    // season between 2 years
    if (season.compare(period[10]) == 0 || season.compare(period[11]) == 0)
    {
        myPredictionYear = myPredictionYear + 1 ;
    }

    Crit3DDate predictionLastDate;
    predictionLastDate.year = myPredictionYear;
    predictionLastDate.month = month2;
    predictionLastDate.day = getDaysInMonth(month2, myPredictionYear);

    wgDoy2 = getDoyFromDate(predictionLastDate);

    return true;
}


void setCorrectWgDoy(int wgDoy1, int wgDoy2, int predictionYear, int myYear, int &fixedWgDoy1, int &fixedWgDoy2)
{
    // check if wgDoy1 and wgDoy2 have been computed starting from a leap year and adjust them for standard years
    if (wgDoy1 < wgDoy2)
    {
        if (isLeapYear(predictionYear) && !isLeapYear(myYear))
        {
            // if wgDoy1 or wgDoy2 > 29th Feb.
            if (wgDoy1 >= 60)
                fixedWgDoy1 = wgDoy1-1;

            if (wgDoy1 >= 60)
                fixedWgDoy2 = wgDoy2-1;
        }
        else if ( !isLeapYear(predictionYear) && isLeapYear(myYear))
        {
            // if wgDoy1 or wgDoy2 > 29th Feb.
            if (wgDoy1 >= 60)
                fixedWgDoy1 = wgDoy1+1;

            if (wgDoy1 >= 60)
                fixedWgDoy2 = wgDoy2+1;
        }
        else
        {
            fixedWgDoy1 = wgDoy1;
            fixedWgDoy2 = wgDoy2;
        }
    }
    else
    {
        if (isLeapYear(predictionYear) && !isLeapYear(myYear))
        {
            // if wgDoy1 > 29th Feb.
            if (wgDoy1 >= 60)
                fixedWgDoy1 = wgDoy1-1;
        }

        else if (isLeapYear(predictionYear+1) && !isLeapYear(myYear))
        {
            // if wgDoy2 > 29th Feb.
            if (wgDoy1 >= 60)
                fixedWgDoy2 = wgDoy2-1;
        }
        else
        {
            fixedWgDoy1 = wgDoy1;
            fixedWgDoy2 = wgDoy2;
        }
    }
}

void setAnomalyMonthScenario(QString startingSeason, int *anomalyMonth1, int *anomalyMonth2)
{
    if(startingSeason == "DJF")
    {
        anomalyMonth1[0] = 12; anomalyMonth2[0] = 2;
        anomalyMonth1[1] = 3; anomalyMonth2[1] = 5;
        anomalyMonth1[2] = 6; anomalyMonth2[2] = 8;
        anomalyMonth1[3] = 9; anomalyMonth2[3] = 11;
    }
    else if (startingSeason == "MAM")
    {
        anomalyMonth1[3] = 12; anomalyMonth2[3] = 2;
        anomalyMonth1[0] = 3; anomalyMonth2[0] = 5;
        anomalyMonth1[1] = 6; anomalyMonth2[1] = 8;
        anomalyMonth1[2] = 9; anomalyMonth2[2] = 11;
    }
    else if (startingSeason == "JJA")
    {
        anomalyMonth1[2] = 12; anomalyMonth2[2] = 2;
        anomalyMonth1[3] = 3; anomalyMonth2[3] = 5;
        anomalyMonth1[0] = 6; anomalyMonth2[0] = 8;
        anomalyMonth1[1] = 9; anomalyMonth2[1] = 11;

    }
    else
    {
        anomalyMonth1[1] = 12; anomalyMonth2[1] = 2;
        anomalyMonth1[2] = 3; anomalyMonth2[2] = 5;
        anomalyMonth1[3] = 6; anomalyMonth2[3] = 8;
        anomalyMonth1[0] = 9; anomalyMonth2[0] = 11;
    }
}
