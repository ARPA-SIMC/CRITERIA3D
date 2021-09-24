#include "crit3dDroughtList.h"

Crit3DDroughtList::Crit3DDroughtList()
{

}

bool Crit3DDroughtList::isMeteoGrid() const
{
    return _isMeteoGrid;
}

void Crit3DDroughtList::setIsMeteoGrid(bool isMeteoGrid)
{
    _isMeteoGrid = isMeteoGrid;
}

void Crit3DDroughtList::insertYearStart(int yearStart)
{
    _listYearStart.push_back(yearStart);
}

void Crit3DDroughtList::insertYearEnd(int yearEnd)
{
    _listYearEnd.push_back(yearEnd);
}

void Crit3DDroughtList::insertIndex(droughtIndex index)
{
    _listIndex.push_back(index);
}

void Crit3DDroughtList::insertFileName(QString filename)
{
    _listFileName.push_back(filename);
}

void Crit3DDroughtList::insertDate(QDate date)
{
    _listDate.push_back(date);
}

void Crit3DDroughtList::insertTimescale(int timescale)
{
    _listTimescale.push_back(timescale);
}

std::vector<int> Crit3DDroughtList::listYearStart() const
{
    return _listYearStart;
}

std::vector<int> Crit3DDroughtList::listYearEnd() const
{
    return _listYearEnd;
}

std::vector<droughtIndex> Crit3DDroughtList::listIndex() const
{
    return _listIndex;
}

std::vector<QDate> Crit3DDroughtList::listDate() const
{
    return _listDate;
}

std::vector<int> Crit3DDroughtList::listTimescale() const
{
    return _listTimescale;
}

std::vector<QString> Crit3DDroughtList::listFileName() const
{
    return _listFileName;
}
