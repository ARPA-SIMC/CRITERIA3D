#include "varDataset.h"
#include "commonConstants.h"

VarDataset::VarDataset()
{

}

VarDataset::VarDataset(QString var)
{
    myVar = var;
}

QString VarDataset::getVar() const
{
    return myVar;
}

int VarDataset::getHourlyValueIndex(double value)
{

    if (hourlyValueList.contains(value))
    {
        // hourly value found
        return hourlyValueList.indexOf(value);
    }
    // hourly value not found, add new hourly value
    hourlyValueList.append(value);
    return (hourlyValueList.size()-1);

}

double VarDataset::getHourlyValue(int index)
{
    if (index > hourlyValueList.size())
        return NODATA;
    else
        return hourlyValueList[index];
}

void VarDataset::setHourlyValue(int index, double value)
{
    if (index > hourlyValueList.size())
        hourlyValueList.append(value);
    else
        hourlyValueList[index] = value;
}
