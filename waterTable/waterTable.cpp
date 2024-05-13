#include "waterTable.h"

WaterTable::WaterTable()
{

}

QDate WaterTable::getFirstDate()
{
    if (wellList.size() == 0)
    {
        return QDate(1800,1,1);
    }
    firstDate = wellList[0].getFirstDate();
    for (int i = 0; i < wellList.size(); i++)
    {
        if (wellList[i].getFirstDate() < firstDate)
        {
            firstDate = wellList[i].getFirstDate();
        }
    }
    return firstDate;
}

QDate WaterTable::getLastDate()
{
    if (wellList.size() == 0)
    {
        return QDate(1800,1,1);
    }
    lastDate = wellList[0].getLastDate();
    for (int i = 0; i < wellList.size(); i++)
    {
        if (wellList[i].getLastDate() > lastDate)
        {
            lastDate = wellList[i].getLastDate();
        }
    }
    return lastDate;
}
