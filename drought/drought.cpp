#include "drought.h"

Drought::Drought()
{

}

meteoVariable Drought::getVar() const
{
    return var;
}

void Drought::setVar(const meteoVariable &value)
{
    var = value;
}

droughtIndex Drought::getIndex() const
{
    return index;
}

void Drought::setIndex(const droughtIndex &value)
{
    index = value;
}

int Drought::getTimeScale() const
{
    return timeScale;
}

void Drought::setTimeScale(int value)
{
    timeScale = value;
}

int Drought::getFirstYear() const
{
    return firstYear;
}

void Drought::setFirstYear(int value)
{
    firstYear = value;
}

int Drought::getLastYear() const
{
    return lastYear;
}

void Drought::setLastYear(int value)
{
    lastYear = value;
}

bool Drought::getUseClima() const
{
    return useClima;
}

void Drought::setUseClima(bool value)
{
    useClima = value;
}

bool Drought::getComputeAll() const
{
    return computeAll;
}

void Drought::setComputeAll(bool value)
{
    computeAll = value;
}
