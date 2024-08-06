#include "commonConstants.h"
#include "well.h"
#include <algorithm>

Well::Well()
{
    lat = NODATA;
    lon = NODATA;
    utmX = NODATA;
    utmY = NODATA;

    id = "";
    depths.clear();
}


void Well::insertData(QDate myDate, float myValue)
{
    depths.insert(myDate, myValue);
}

int  Well::getObsDepthNr()
{
    return depths.size();
}



QDate Well::getFirstDate()
{
    QList<QDate> allDates = depths.keys();
    firstDate = allDates[0];
    for (int i = 0; i < allDates.size(); i++)
    {
        if (allDates[i] < firstDate)
        {
            firstDate = allDates[i];
        }
    }
    return firstDate;
}


QDate Well::getLastDate()
{
    QList<QDate> allDates = depths.keys();
    lastDate = allDates[0];
    for (int i = 0; i < allDates.size(); i++)
    {
        if (allDates[i] > lastDate)
        {
            lastDate = allDates[i];
        }
    }
    return lastDate;
}


int Well::minValuesPerMonth()
{
    QMapIterator<QDate, float> it(depths);
    std::vector<int> H_num;
    for (int myMonthIndex = 0; myMonthIndex < 12; myMonthIndex++)
    {
        H_num.push_back(0);
    }
    while (it.hasNext())
    {
        it.next();
        QDate myDate = it.key();
        int myMonth = myDate.month();
        int myMonthIndex = myMonth - 1;
        H_num[myMonthIndex] = H_num[myMonthIndex] + 1;
    }

    auto min = min_element(H_num.begin(), H_num.end());
    return *min;
}
