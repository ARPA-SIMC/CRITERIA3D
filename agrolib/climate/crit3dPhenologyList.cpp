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


bool Crit3DPhenologyList::addPhenology(unsigned int index)
{
    QString dateStart = _listDateStart[index].toString("yyyy-MM-dd");
    QString dateEnd = _listDateEnd[index].toString("yyyy-MM-dd");
    QString computation;
    QString variety;
    QString scale;

    if (_listComputation[index] == currentStage)
    {
        computation = "Current";
    }
    else if(_listComputation[index] == anomalyDays)
    {
        computation = "Anomaly";
    }
    else if(_listComputation[index] == differenceStages)
    {
        computation = "Difference";
    }

    if (_listVariety[index] == precocissima)
    {
        variety = "precocissima";
    }
    else if(_listVariety[index] == precoce)
    {
        variety = "precoce";
    }
    else if(_listVariety[index] == media)
    {
        variety = "media";
    }
    else if(_listVariety[index] == tardive)
    {
        variety = "tardive";
    }

    if (_listScale[index] == ARPA)
    {
        scale = "ARPA";
    }
    else if(_listScale[index] == BBCH)
    {
        scale = "BBCH";
    }

    QString crop = QString::fromStdString(getStringMapPhenoCrop(MapPhenoCropToString, _listCrop[index]));

    QString phenologyAdded = computation + "_" + crop + "_" + variety + "_" + scale + "_VERN:" + QString::number(_listVernalization[index]) + "_" + dateStart + "_"+ dateEnd;

    /*if(std::find(_listAll.begin(), _listAll.end(), phenologyAdded) != _listAll.end())
    {
        return false;
    }
    else
    {*/

    _listAll.push_back(phenologyAdded);
    return true;

}
