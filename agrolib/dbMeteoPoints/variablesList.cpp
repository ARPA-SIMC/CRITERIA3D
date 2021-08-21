#include "variablesList.h"


VariablesList::VariablesList(int id, int arkId, QString varName, int frequency)
{
    _id = id;
    _arkId = arkId;
    _varName = varName;
    _frequency = frequency;
}

int VariablesList::id() const
{
    return _id;
}

int VariablesList::arkId() const
{
    return _arkId;
}

QString VariablesList::varName() const
{
    return _varName;
}

int VariablesList::frequency() const
{
    return _frequency;
}

