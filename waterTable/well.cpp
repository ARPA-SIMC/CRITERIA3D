#include "well.h"
#include <algorithm>

Well::Well()
{

}

QString Well::getId() const
{
    return id;
}

void Well::setId(const QString &newId)
{
    id = newId;
}

double Well::getUtmX() const
{
    return utmX;
}

void Well::setUtmX(double newUtmX)
{
    utmX = newUtmX;
}

double Well::getUtmY() const
{
    return utmY;
}

void Well::setUtmY(double newUtmY)
{
    utmY = newUtmY;
}

void Well::insertData(QDate myDate, float myValue)
{
    depths.insert(myDate, myValue);
}

int  Well::getObsDepthNr()
{
    return depths.size();
}


QMap<QDate, float> Well::getObsDepths()
{
    return depths;
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
