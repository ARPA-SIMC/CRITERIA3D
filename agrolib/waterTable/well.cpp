#include "commonConstants.h"
#include "well.h"
#include <algorithm>


Well::Well()
{
    initialize();
}


void Well::initialize()
{
    _lat = NODATA;
    _lon = NODATA;
    _utmX = NODATA;
    _utmY = NODATA;

    _id = "";
    depths.clear();
}


void Well::updateDates()
{
    QList<QDate> obsDateList = depths.keys();
    _firstDate = obsDateList[0];
    _lastDate = obsDateList[0];

    for (int i = 0; i < obsDateList.size(); i++)
    {
        if (obsDateList[i] < _firstDate)
        {
            _firstDate = obsDateList[i];
        }
        if (obsDateList[i] > _lastDate)
        {
            _lastDate = obsDateList[i];
        }
    }
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
