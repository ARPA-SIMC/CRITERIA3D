#include "well.h"

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

void Well::insertData(QDate myDate, int myValue)
{
    depths.insert(myDate, myValue);
}

int  Well::getDepthNr()
{
    return depths.size();
}

QMap<QDate, int> Well::getDepths() const
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
