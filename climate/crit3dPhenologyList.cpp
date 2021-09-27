#include "crit3dPhenologyList.h"

Crit3DPhenologyList::Crit3DPhenologyList()
{

}

void Crit3DPhenologyList::reset()
{
    _listAll.clear();
    _listDateStart.clear();
    _listDateEnd.clear();
    _listComputation.clear();
    _listFileName.clear();
    _listCrop.clear();
    _listVariety.clear();
    _listVernalization.clear();
    _listScale.clear();
}

bool Crit3DPhenologyList::isMeteoGrid() const
{
    return _isMeteoGrid;
}

void Crit3DPhenologyList::setIsMeteoGrid(bool isMeteoGrid)
{
    _isMeteoGrid = isMeteoGrid;
}

void Crit3DPhenologyList::insertFileName(QString filename)
{
    _listFileName.push_back(filename);
}

void Crit3DPhenologyList::insertDateStart(QDate dateStart)
{
    _listDateStart.push_back(dateStart);
}

void Crit3DPhenologyList::insertDateEnd(QDate dateEnd)
{
    _listDateEnd.push_back(dateEnd);
}

void Crit3DPhenologyList::insertComputation(phenoComputation computation)
{
    _listComputation.push_back(computation);
}

void Crit3DPhenologyList::insertCrop(phenoCrop crop)
{
    _listCrop.push_back(crop);
}

void Crit3DPhenologyList::insertVariety(phenoVariety variety)
{
    _listVariety.push_back(variety);
}

void Crit3DPhenologyList::insertVernalization(int vernalization)
{
    _listVernalization.push_back(vernalization);
}

void Crit3DPhenologyList::insertScale(phenoScale scale)
{
    _listScale.push_back(scale);
}

std::vector<QString> Crit3DPhenologyList::listFileName() const
{
    return _listFileName;
}

std::vector<QString> Crit3DPhenologyList::listAll() const
{
    return _listAll;
}

std::vector<QDate> Crit3DPhenologyList::listDateStart() const
{
    return _listDateStart;
}

std::vector<QDate> Crit3DPhenologyList::listDateEnd() const
{
    return _listDateEnd;
}

std::vector<phenoComputation> Crit3DPhenologyList::listComputation() const
{
    return _listComputation;
}

std::vector<phenoCrop> Crit3DPhenologyList::listCrop() const
{
    return _listCrop;
}

std::vector<phenoVariety> Crit3DPhenologyList::listVariety() const
{
    return _listVariety;
}

std::vector<int> Crit3DPhenologyList::listVernalization() const
{
    return _listVernalization;
}

std::vector<phenoScale> Crit3DPhenologyList::listScale() const
{
    return _listScale;
}

void Crit3DPhenologyList::eraseElement(unsigned int index)
{
    if (_listAll.size() > index)
    {
        _listAll.erase(_listAll.begin() + index);
    }
    if (_listDateStart.size() > index)
    {
        _listDateStart.erase(_listDateStart.begin() + index);
    }
    if (_listDateEnd.size() > index)
    {
        _listDateEnd.erase(_listDateEnd.begin() + index);
    }
    if (_listComputation.size() > index)
    {
        _listComputation.erase(_listComputation.begin() + index);
    }
    if (_listFileName.size() > index)
    {
        _listFileName.erase(_listFileName.begin() + index);
    }
    if (_listCrop.size() > index)
    {
        _listCrop.erase(_listCrop.begin() + index);
    }
    if (_listVariety.size() > index)
    {
        _listVariety.erase(_listVariety.begin() + index);
    }
    if (_listVernalization.size() > index)
    {
        _listVernalization.erase(_listVernalization.begin() + index);
    }
    if (_listScale.size() > index)
    {
        _listScale.erase(_listScale.begin() + index);
    }

}

void Crit3DPhenologyList::addPhenology(unsigned int index)
{
    // TO DO
    /*
    QString dateStart = _listDateStart[index].toString("dd/MM/yyyy");
    QString dateEnd = _listDateEnd[index].toString("dd/MM/yyyy");

    QString phenologyAdded = _listComputation[index] + "_" + _listCrop[index] + "_" + _listVariety[index] + "_" + _listScale[index] + "_VERN:" + _listVernalization[index] + "_" + dateStart + "_"+ dateEnd;
    if(std::find(_listAll.begin(), _listAll.end(), phenologyAdded) != _listAll.end())
    {
        return;
    }
    else
    {
        _listAll.push_back(phenologyAdded);
    }
    */

}
