#include <QString>
#include <QDate>

#include "commonConstants.h"
#include "crit3dClimateList.h"
#include "climate.h"


Crit3DClimateList::Crit3DClimateList()
{

}

Crit3DClimateList::~Crit3DClimateList()
{
}

QList<QString> Crit3DClimateList::listClimateElab() const
{
    return _listClimateElab;
}

void Crit3DClimateList::setListClimateElab(const QList<QString> &listClimateElab)
{
    _listClimateElab = listClimateElab;
}

void Crit3DClimateList::resetListClimateElab()
{
    _listElab1.clear();
    _listElab2.clear();
    _listGenericPeriodDateStart.clear();
    _listGenericPeriodDateEnd.clear();
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
}

std::vector<int> Crit3DClimateList::listYearStart() const
{
    return _listYearStart;
}

void Crit3DClimateList::setListYearStart(const std::vector<int> &listYearStart)
{
    _listYearStart = listYearStart;
}

std::vector<int> Crit3DClimateList::listYearEnd() const
{
    return _listYearEnd;
}

void Crit3DClimateList::setListYearEnd(const std::vector<int> &listYearEnd)
{
    _listYearEnd = listYearEnd;
}

std::vector<meteoVariable> Crit3DClimateList::listVariable() const
{
    return _listVariable;
}

void Crit3DClimateList::setListVariable(const std::vector<meteoVariable> &listVariable)
{
    _listVariable = listVariable;
}

std::vector<QString> Crit3DClimateList::listPeriodStr() const
{
    return _listPeriodStr;
}

void Crit3DClimateList::setListPeriodStr(const std::vector<QString> &listPeriodStr)
{
    _listPeriodStr = listPeriodStr;
}

std::vector<period> Crit3DClimateList::listPeriodType() const
{
    return _listPeriodType;
}

void Crit3DClimateList::setListPeriodType(const std::vector<period> &listPeriodType)
{
    _listPeriodType = listPeriodType;
}

std::vector<QDate> Crit3DClimateList::listGenericPeriodDateStart() const
{
    return _listGenericPeriodDateStart;
}

void Crit3DClimateList::setListGenericPeriodDateStart(const std::vector<QDate> &listGenericPeriodDateStart)
{
    _listGenericPeriodDateStart = listGenericPeriodDateStart;
}

std::vector<QDate> Crit3DClimateList::listGenericPeriodDateEnd() const
{
    return _listGenericPeriodDateEnd;
}

void Crit3DClimateList::setListGenericPeriodDateEnd(const std::vector<QDate> &listGenericPeriodDateEnd)
{
    _listGenericPeriodDateEnd = listGenericPeriodDateEnd;
}

std::vector<int> Crit3DClimateList::listNYears() const
{
    return _listNYears;
}

void Crit3DClimateList::setListNYears(const std::vector<int> &listNYears)
{
    _listNYears = listNYears;
}

std::vector<QString> Crit3DClimateList::listElab1() const
{
    return _listElab1;
}

void Crit3DClimateList::setListElab1(const std::vector<QString> &listElab1)
{
    _listElab1 = listElab1;
}

std::vector<float> Crit3DClimateList::listParam1() const
{
    return _listParam1;
}

void Crit3DClimateList::setListParam1(const std::vector<float> &listParam1)
{
    _listParam1 = listParam1;
}

std::vector<bool> Crit3DClimateList::listParam1IsClimate() const
{
    return _listParam1IsClimate;
}

void Crit3DClimateList::setListParam1IsClimate(const std::vector<bool> &listParam1IsClimate)
{
    _listParam1IsClimate = listParam1IsClimate;
}

std::vector<QString> Crit3DClimateList::listParam1ClimateField() const
{
    return _listParam1ClimateField;
}

void Crit3DClimateList::setListParam1ClimateField(const std::vector<QString> &listParam1ClimateField)
{
    _listParam1ClimateField = listParam1ClimateField;
}

std::vector<QString> Crit3DClimateList::listElab2() const
{
    return _listElab2;
}

void Crit3DClimateList::setListElab2(const std::vector<QString> &listElab2)
{
    _listElab2 = listElab2;
}

std::vector<float> Crit3DClimateList::listParam2() const
{
    return _listParam2;
}

void Crit3DClimateList::setListParam2(const std::vector<float> &listParam2)
{
    _listParam2 = listParam2;
}

void Crit3DClimateList::insertDailyCumulated(bool dailyCumulated)
{
   _listDailyCumulated.push_back(dailyCumulated);
}

std::vector<bool> Crit3DClimateList::listDailyCumulated() const
{
   return _listDailyCumulated;
}


void Crit3DClimateList::parserElaboration(QString &errorStr)
{
    for (int i = 0; i < _listClimateElab.size(); i++)
    {
        int pos = 0;

        QString climateElab = _listClimateElab[i];

        QList<QString> words = climateElab.split('_');

        if (words.isEmpty())
        {
            _listClimateElab.replace(i, "NULL");
            continue;
        }

        // split years
        QString periodElabList = words.at(pos);
        QList<QString> myYearWords = periodElabList.split('-');

        bool isOk = false;
        int firstYear = myYearWords[0].toInt(&isOk);
        if (! isOk)
        {
            errorStr += "Wrong year: " + myYearWords[0] + "\n";
            _listClimateElab.replace(i, "NULL");
            continue;
        }

        int lastYear = myYearWords[1].toInt(&isOk);
        if (! isOk)
        {
            errorStr += "Wrong year: " + myYearWords[1] + "\n";
            _listClimateElab.replace(i, "NULL");
            continue;
        }

        _listYearStart.push_back(firstYear);
        _listYearEnd.push_back(lastYear);

        pos = pos + 1;

        if (words.size() == pos)
        {
            errorStr += "Missing parameters in line: " + climateElab + "\n";
            _listClimateElab.replace(i, "NULL");
            continue;
        }

        meteoVariable var = noMeteoVar;
        QString varStr = words[pos];
        if (varStr != "")
        {
            if (varStr.contains("CUMULATED"))
            {
                _listDailyCumulated.push_back(true);
                varStr.remove("CUMULATED");
            }
            else
            {
                _listDailyCumulated.push_back(false);
            }
            var = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, varStr.toUpper().toStdString());
            if (var == noMeteoVar)
            {
                var = getKeyMeteoVarMeteoMapWithoutUnderscore(MapDailyMeteoVarToString, varStr.toUpper().toStdString());
            }
        }

        if (var == noMeteoVar)
        {
            errorStr += "Wrong variable: " + varStr + "\n";
            _listClimateElab.replace(i, "NULL");
            continue;
        }
        else
        {
            _listVariable.push_back(var);
        }

        pos = pos + 1;

        if (words.size() == pos)
        {
            errorStr += "Missing parameters in line: " + climateElab + "\n";
            _listClimateElab.replace(i, "NULL");
            continue;
        }

        QString periodTypeStr = words[pos];
        _listPeriodType.push_back(getPeriodTypeFromString(periodTypeStr));

        pos = pos + 1; // pos = 3

        if (words.size() == pos)
        {
            errorStr += "Missing parameters in line: " + climateElab + "\n";
            _listClimateElab.replace(i, "NULL");
            continue;
        }

        if ( (_listPeriodType[i] == genericPeriod) && ( (words[pos].at(0)).isDigit() ))
        {
            _listPeriodStr.push_back(words[pos]);
            parserGenericPeriodString(i);
            pos = pos + 1; // pos = 4
        }
        else
        {
            _listPeriodStr.push_back(periodTypeStr);
            _listGenericPeriodDateStart.push_back( QDate(0,  0,  0) );
            _listGenericPeriodDateEnd.push_back( QDate(0,  0,  0) );
            _listNYears.push_back(0);
        }

        if (words.size() == pos)
        {
            errorStr += "Missing parameters in line: " + climateElab + "\n";
            _listClimateElab.replace(i, "NULL");
            continue;
        }

        QString elab = words[pos];
        bool param1IsClimate;
        QString param1ClimateField;

        meteoComputation elabMeteoComputation = getMeteoCompFromString(MapMeteoComputation, elab.toStdString());
        if (elabMeteoComputation == noMeteoComp)
        {
            errorStr += "Wrong elaboration: " + elab + "\n";
            _listClimateElab.replace(i, "NULL");
            continue;
        }

        float param = NODATA;
        int nrParam = nParameters(elabMeteoComputation);

        if (nrParam > 0)
        {
            pos = pos + 1;
            if ( words[pos].at(0) == '|' )
            {
                param1IsClimate = true;
                param1ClimateField = words[pos];
                param1ClimateField.remove(0,1);

                pos = pos + 1;
                if ( words[pos].right(2) == "||" )
                {
                     _listClimateElab.replace(i, "NULL");
                }

                while ( words[pos].right(2) != "||" )
                {
                    param1ClimateField = param1ClimateField + "_" + words[pos];
                    pos = pos + 1;
                }
                param1ClimateField = param1ClimateField + "_" + words[pos].left(words[pos].size() - 2);
                param =  NODATA;
            }
            else
            {
                param1IsClimate = false;
                param1ClimateField = "";
                param = words[pos].toFloat();
            }
        }
        else
        {
            param1IsClimate = false;
            param1ClimateField = "";
            param =  NODATA;
        }

        pos = pos + 1;
        QString elab1;
        // second elab present
        if (words.size() > pos)
        {
            _listElab2.push_back(elab);
            _listParam2.push_back(param);


            elab1 = words[pos];
            _listElab1.push_back(elab1);
            elabMeteoComputation = getMeteoCompFromString(MapMeteoComputation, elab1.toStdString());

            if (elabMeteoComputation == noMeteoComp)
            {
                errorStr += "Wrong elaboration: " + elab1 + "\n";
                _listClimateElab.replace(i, "NULL");
                continue;
            }

            nrParam = nParameters(elabMeteoComputation);

            if (nrParam > 0)
            {
                pos = pos + 1;
                if ( words[pos].at(0) == '|' )
                {
                    param1IsClimate = true;
                    param1ClimateField = words[pos];
                    param1ClimateField.remove(0,1);

                    pos = pos + 1;
                    if ( words[pos].right(2) == "||" )
                    {
                         _listClimateElab.replace(i, "NULL");
                    }

                    while ( words[pos].right(2) != "||" )
                    {
                        param1ClimateField = param1ClimateField + "_" + words[pos];
                        pos = pos + 1;
                    }
                    param1ClimateField = param1ClimateField + "_" + words[pos].left(words[pos].size() - 2);

                    _listParam1.push_back(NODATA);
                }
                else
                {
                    param1IsClimate = false;
                    param1ClimateField = "";
                    _listParam1.push_back( words[pos].toFloat() );
                }
            }
            else
            {
                param1IsClimate = false;
                param1ClimateField = "";
                _listParam1.push_back(NODATA);
            }
            _listParam1IsClimate.push_back(param1IsClimate);
            _listParam1ClimateField.push_back(param1ClimateField);

        }
        else
        {
            _listElab1.push_back(elab);
            _listParam1.push_back(param);
            _listElab2.push_back(nullptr);
            _listParam2.push_back(NODATA);

            _listParam1IsClimate.push_back(param1IsClimate);
            _listParam1ClimateField.push_back(param1ClimateField);
        }
    }
}


bool Crit3DClimateList::parserGenericPeriodString(int index)
{

    QString periodString = _listPeriodStr.at(index);

    if ( periodString.isEmpty())
    {
        return false;
    }

    QString day = periodString.mid(0,2);
    QString month = periodString.mid(3,2);
    int year = 2000;
    _listGenericPeriodDateStart.push_back( QDate(year,  month.toInt(),  day.toInt()) );

    day = periodString.mid(6,2);
    month = periodString.mid(9,2);

    _listGenericPeriodDateEnd.push_back( QDate(year,  month.toInt(),  day.toInt()) );

    if ( periodString.size() > 11 )
    {
        _listNYears.push_back((periodString.mid(13,2)).toInt());
    }
    return true;

}
