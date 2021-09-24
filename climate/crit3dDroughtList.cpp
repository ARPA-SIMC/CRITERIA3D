#include "crit3dDroughtList.h"

Crit3DDroughtList::Crit3DDroughtList()
{

}

void Crit3DDroughtList::reset()
{
    _listAll.clear();
    _listYearEnd.clear();
    _listYearStart.clear();
    _listIndex.clear();
    _listFileName.clear();
    _listDate.clear();
    _listTimescale.clear();
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

std::vector<QString> Crit3DDroughtList::listAll() const
{
    return _listAll;
}

void Crit3DDroughtList::eraseElement(unsigned int index)
{
    if (_listAll.size() > index)
    {
        _listAll.erase(_listAll.begin() + index);
    }
    if (_listYearEnd.size() > index)
    {
        _listYearEnd.erase(_listYearEnd.begin() + index);
    }
    if (_listYearStart.size() > index)
    {
        _listYearStart.erase(_listYearStart.begin() + index);
    }
    if (_listDate.size() > index)
    {
        _listDate.erase(_listDate.begin() + index);
    }
    if (_listFileName.size() > index)
    {
        _listFileName.erase(_listFileName.begin() + index);
    }
    if (_listIndex.size() > index)
    {
        _listIndex.erase(_listIndex.begin() + index);
    }
    if (_listTimescale.size() > index)
    {
        _listTimescale.erase(_listTimescale.begin() + index);
    }

}

void Crit3DDroughtList::addDrought(unsigned int index)
{

    QString yearStart = QString::number(_listYearStart[index]);
    QString yearEnd = QString::number(_listYearEnd[index]);
    QString date = _listDate[index].toString();

    int timeScale = _listTimescale[index];
    droughtIndex thisIndex = _listIndex[index];
    QString indexStr;
    if (thisIndex == INDEX_SPI)
    {
        indexStr = "SPI";
    }
    else if (thisIndex == INDEX_SPEI)
    {
        indexStr = "SPEI";
    }
    else if (thisIndex == INDEX_DECILES)
    {
        indexStr = "DECILES";
    }

    QString droughtAdded = indexStr + "_TIMESCALE" + timeScale +  + "_" + yearStart + "-" + yearEnd + "_" + date ;
    if(std::find(_listAll.begin(), _listAll.end(), droughtAdded) != _listAll.end())
    {
        return;
    }
    else
    {
        _listAll.push_back(droughtAdded);
    }

}
