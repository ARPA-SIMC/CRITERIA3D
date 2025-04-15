#include "commonConstants.h"
#include "well.h"
#include <algorithm>

Well::Well()
{
    _lat = NODATA;
    _lon = NODATA;
    _utmX = NODATA;
    _utmY = NODATA;

    _id = "";
    depths.clear();
}


QDate Well::getFirstObsDate()
{
    QList<QDate> obsDateList = depths.keys();
    _firstDate = obsDateList[0];
    for (int i = 0; i < obsDateList.size(); i++)
    {
        if (obsDateList[i] < _firstDate)
        {
            _firstDate = obsDateList[i];
        }
    }

    return _firstDate;
}


QDate Well::getLastObsDate()
{
    QList<QDate> obsDateList = depths.keys();
    _lastDate = obsDateList[0];
    for (int i = 0; i < obsDateList.size(); i++)
    {
        if (obsDateList[i] > _lastDate)
        {
            _lastDate = obsDateList[i];
        }
    }

    return _lastDate;
}


int Well::minValuesPerMonth()
{
    QMapIterator<QDate, float> it(depths);
    std::vector<int> H_num;
    for (int monthIndex = 0; monthIndex < 12; monthIndex++)
    {
        H_num.push_back(0);
    }
    while (it.hasNext())
    {
        it.next();
        QDate myDate = it.key();
        int myMonth = myDate.month();
        int monthIndex = myMonth - 1;
        H_num[monthIndex] = H_num[monthIndex] + 1;
    }

    auto min = min_element(H_num.begin(), H_num.end());
    return *min;
}
