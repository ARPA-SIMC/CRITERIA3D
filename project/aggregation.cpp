#include "aggregation.h"


std::string getKeyStringAggregationMethod(aggregationMethod value)
{
    std::map<std::string, aggregationMethod>::const_iterator it;
    std::string key = "";

    for (it = aggregationMethodToString.begin(); it != aggregationMethodToString.end(); ++it)
    {
        if (it->second == value)
        {
            key = it->first;
            break;
        }
    }
    return key;
}


aggregationMethod getAggregationMethod(const std::string& value)
{
    std::map<aggregationMethod, std::string>::const_iterator it;
    aggregationMethod key = noAggrMethod;

    for (it = aggregationStringToMethod.begin(); it != aggregationStringToMethod.end(); ++it)
    {
        if (it->second == value)
        {
            key = it->first;
            break;
        }
    }
    return key;
}
