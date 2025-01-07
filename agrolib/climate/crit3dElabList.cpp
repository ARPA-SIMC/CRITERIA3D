#include <QString>
#include <QDate>
#include <QDebug>

#include "crit3dElabList.h"
#include "commonConstants.h"


Crit3DElabList::Crit3DElabList()
{

}

Crit3DElabList::~Crit3DElabList()
{
}

bool Crit3DElabList::isMeteoGrid() const
{
    return _isMeteoGrid;
}

void Crit3DElabList::setIsMeteoGrid(bool isMeteoGrid)
{
    _isMeteoGrid = isMeteoGrid;
}

QList<QString> Crit3DElabList::listAll() const
{
    return _listAll;
}

void Crit3DElabList::setListAll(const QList<QString> &listElab)
{
    _listAll = listElab;
}

void Crit3DElabList::reset()
{
    _listAll.clear();
    _listElab1.clear();
    _listElab2.clear();
    _listDateStart.clear();
    _listDateEnd.clear();
    _listNYears.clear();
    _listParam1.clear();
    _listParam1ClimateField.clear();
    _listParam1IsClimate.clear();
    _listParam2.clear();
    _listPeriodStr.clear();
    _listPeriodType.clear();
    _listVariable.clear();
    _listYearEnd.clear();
    _listYearStart.clear();
    _listDailyCumulated.clear();
}

void Crit3DElabList::eraseElement(unsigned int index)
{
    if (_listAll.size() > index)
    {
        _listAll.erase(_listAll.begin() + index);
    }
    if (_listElab1.size() > index)
    {
        _listElab1.erase(_listElab1.begin() + index);
    }
    if (_listElab2.size() > index)
    {
        _listElab2.erase(_listElab2.begin() + index);
    }
    if (_listDateStart.size() > index)
    {
        _listDateStart.erase(_listDateStart.begin() + index);
    }
    if (_listDateEnd.size() > index)
    {
        _listDateEnd.erase(_listDateEnd.begin() + index);
    }
    if (_listNYears.size() > index)
    {
        _listNYears.erase(_listNYears.begin() + index);
    }
    if (_listParam1.size() > index)
    {
        _listParam1.erase(_listParam1.begin() + index);
    }
    if (_listParam1ClimateField.size() > index)
    {
        _listParam1ClimateField.erase(_listParam1ClimateField.begin() + index);
    }
    if (_listParam1IsClimate.size() > index)
    {
        _listParam1IsClimate.erase(_listParam1IsClimate.begin() + index);
    }
    if (_listParam2.size() > index)
    {
        _listParam2.erase(_listParam2.begin() + index);
    }
    if (_listPeriodStr.size() > index)
    {
        _listPeriodStr.erase(_listPeriodStr.begin() + index);
    }
    if (_listPeriodType.size() > index)
    {
        _listPeriodType.erase(_listPeriodType.begin() + index);
    }
    if (_listVariable.size() > index)
    {
        _listVariable.erase(_listVariable.begin() + index);
    }
    if (_listYearEnd.size() > index)
    {
        _listYearEnd.erase(_listYearEnd.begin() + index);
    }
    if (_listYearStart.size() > index)
    {
        _listYearStart.erase(_listYearStart.begin() + index);
    }
    if (_listDailyCumulated.size() > index)
    {
        _listDailyCumulated.erase(_listDailyCumulated.begin() + index);
    }
}

std::vector<int> Crit3DElabList::listYearStart() const
{
    return _listYearStart;
}

void Crit3DElabList::setListYearStart(const std::vector<int> &listYearStart)
{
    _listYearStart = listYearStart;
}

void Crit3DElabList::insertYearStart(int yearStart)
{
    _listYearStart.push_back(yearStart);
}

std::vector<int> Crit3DElabList::listYearEnd() const
{
    return _listYearEnd;
}

void Crit3DElabList::setListYearEnd(const std::vector<int> &listYearEnd)
{
    _listYearEnd = listYearEnd;
}

void Crit3DElabList::insertYearEnd(int yearEnd)
{
    _listYearEnd.push_back(yearEnd);
}

std::vector<meteoVariable> Crit3DElabList::listVariable() const
{
    return _listVariable;
}

void Crit3DElabList::setListVariable(const std::vector<meteoVariable> &listVariable)
{
    _listVariable = listVariable;
}

void Crit3DElabList::insertVariable(meteoVariable variable)
{
    _listVariable.push_back(variable);
}

std::vector<QString> Crit3DElabList::listPeriodStr() const
{
    return _listPeriodStr;
}

void Crit3DElabList::setListPeriodStr(const std::vector<QString> &listPeriodStr)
{
    _listPeriodStr = listPeriodStr;
}

void Crit3DElabList::insertPeriodStr(QString period)
{
    _listPeriodStr.push_back(period);
}

std::vector<period> Crit3DElabList::listPeriodType() const
{
    return _listPeriodType;
}

void Crit3DElabList::setListPeriodType(const std::vector<period> &listPeriodType)
{
    _listPeriodType = listPeriodType;
}

void Crit3DElabList::insertPeriodType(period period)
{
    _listPeriodType.push_back(period);
}

std::vector<QDate> Crit3DElabList::listDateStart() const
{
    return _listDateStart;
}

void Crit3DElabList::setListDateStart(const std::vector<QDate> &listDateStart)
{
    _listDateStart = listDateStart;
}

void Crit3DElabList::insertDateStart(QDate dateStart)
{
    _listDateStart.push_back(dateStart);
}

std::vector<QDate> Crit3DElabList::listDateEnd() const
{
    return _listDateEnd;
}

void Crit3DElabList::setListDateEnd(const std::vector<QDate> &listDateEnd)
{
    _listDateEnd = listDateEnd;
}

void Crit3DElabList::insertDateEnd(QDate dateEnd)
{
    _listDateEnd.push_back(dateEnd);
}

std::vector<int> Crit3DElabList::listNYears() const
{
    return _listNYears;
}

void Crit3DElabList::setListNYears(const std::vector<int> &listNYears)
{
    _listNYears = listNYears;
}

void Crit3DElabList::insertNYears(int nYears)
{
    _listNYears.push_back(nYears);
}

std::vector<QString> Crit3DElabList::listElab1() const
{
    return _listElab1;
}

void Crit3DElabList::setListElab1(const std::vector<QString> &listElab1)
{
    _listElab1 = listElab1;
}

void Crit3DElabList::insertElab1(QString elab1)
{
    _listElab1.push_back(elab1);
}

std::vector<float> Crit3DElabList::listParam1() const
{
    return _listParam1;
}

void Crit3DElabList::setListParam1(const std::vector<float> &listParam1)
{
    _listParam1 = listParam1;
}

void Crit3DElabList::insertParam1(float param1)
{
    _listParam1.push_back(param1);
}

std::vector<bool> Crit3DElabList::listParam1IsClimate() const
{
    return _listParam1IsClimate;
}

void Crit3DElabList::setListParam1IsClimate(const std::vector<bool> &listParam1IsClimate)
{
    _listParam1IsClimate = listParam1IsClimate;
}

void Crit3DElabList::insertParam1IsClimate(bool param1IsClimate)
{
    _listParam1IsClimate.push_back(param1IsClimate);
}

std::vector<QString> Crit3DElabList::listParam1ClimateField() const
{
    return _listParam1ClimateField;
}

void Crit3DElabList::setListParam1ClimateField(const std::vector<QString> &listParam1ClimateField)
{
    _listParam1ClimateField = listParam1ClimateField;
}

void Crit3DElabList::insertParam1ClimateField(QString param1ClimateField)
{
    _listParam1ClimateField.push_back(param1ClimateField);
}

std::vector<QString> Crit3DElabList::listElab2() const
{
    return _listElab2;
}

void Crit3DElabList::setListElab2(const std::vector<QString> &listElab2)
{
    _listElab2 = listElab2;
}

void Crit3DElabList::insertElab2(QString elab2)
{
    _listElab2.push_back(elab2);
}

std::vector<float> Crit3DElabList::listParam2() const
{
    return _listParam2;
}

void Crit3DElabList::setListParam2(const std::vector<float> &listParam2)
{
    _listParam2 = listParam2;
}

void Crit3DElabList::insertParam2(float param2)
{
    _listParam2.push_back(param2);
}

void Crit3DElabList::insertDailyCumulated(bool dailyCumulated)
{
    _listDailyCumulated.push_back(dailyCumulated);
}

std::vector<bool> Crit3DElabList::listDailyCumulated() const
{
    return _listDailyCumulated;
}


bool Crit3DElabList::addElab(unsigned int index)
{

    QString yearStart = QString::number(_listYearStart[index]);
    QString yearEnd = QString::number(_listYearEnd[index]);
    QString variable = QString::fromStdString(MapDailyMeteoVarToString.at(_listVariable[index])).remove("_");
    if (_listDailyCumulated[index] == true)
    {
        variable = variable+"CUMULATED";
    }
    QString period = _listPeriodStr[index];
    QString periodStartDay = QString::number(_listDateStart[index].day());
    QString periodStartMonth = QString::number(_listDateStart[index].month());
    QString periodEndDay = QString::number(_listDateEnd[index].day());
    QString periodEndMonth = QString::number(_listDateEnd[index].month());
    QString nYear = QString::number(_listNYears[index]);
    QString elab1 = _listElab1[index];
    QString secondElab = _listElab2[index];
    float elab1Param = _listParam1[index];
    float elab2Param = _listParam2[index];
    QString elab1ParamFromdB = _listParam1ClimateField[index];

    QString elabAdded = yearStart + "-" + yearEnd + "_" + variable + "_" + period;
    if (periodStartDay != "0" && periodStartMonth != "0" && periodEndDay != "0" && periodEndMonth != "0")
    {
        elabAdded = elabAdded + "_" + periodStartDay + "of" + periodStartMonth + "-" + periodEndDay + "of" + periodEndMonth;
    }
    if (nYear != "0")
    {
        elabAdded = elabAdded + "-+" + nYear + "y";
    }

    if (!secondElab.isEmpty())
    {
        elabAdded = elabAdded + "_" + secondElab;

        if (elab2Param != NODATA)
        {
            elabAdded = elabAdded + "_" + QString::number(elab2Param);
        }
    }
    if (elab1 != "")
    {
        elabAdded = elabAdded + "_" + elab1;
        if (elab1Param != NODATA)
        {
            elabAdded = elabAdded + "_" + QString::number(elab1Param);
        }
        else if(_listParam1IsClimate[index] == true && !elab1ParamFromdB.isEmpty())
        {
            elabAdded = elabAdded + "_|" + elab1ParamFromdB + "||";
        }
    }
    else
    {
        qDebug() << "elab1 is empty " ;
        elabAdded = elabAdded + "_" + "noMeteoComp";
    }

    if (_listAll.contains(elabAdded)!= 0)
    {
        qDebug() << "return false elabAdded: " << elabAdded;
        return false;
    }

    _listAll.append(elabAdded);
    return true;
}

std::vector<QString> Crit3DElabList::listFileName() const
{
    return _listFileName;
}

void Crit3DElabList::setListFileName(const std::vector<QString> &listFileName)
{
    _listFileName = listFileName;
}

void Crit3DElabList::insertFileName(QString filename)
{
    _listFileName.push_back(filename);
}



