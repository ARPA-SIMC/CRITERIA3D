#ifndef AGGREGATION_H
#define AGGREGATION_H

    #ifndef STATISTICS_H
        #include "statistics.h"
    #endif

    #include <string>
    #include <map>

    const std::map<std::string, aggregationMethod> aggregationMethodToString = {
        { "AVG", aggrAverage },
        { "MEDIAN", aggrMedian },
        { "STDDEV", aggrStdDeviation },
        { "MIN", aggrMin },
        { "MAX", aggrMax },
        { "SUM", aggrSum },
        { "PREVAILING", aggrPrevailing },
        { "CENTER", aggrCenter },
        { "PERC95", aggr95Perc }
    };

    const std::map<aggregationMethod, std::string> aggregationStringToMethod = {
        { aggrAverage, "AVG" },
        { aggrMedian, "MEDIAN" },
        { aggrStdDeviation, "STDDEV" },
        { aggrMin, "MIN" },
        { aggrMax, "MAX" },
        { aggrSum, "SUM" },
        { aggrPrevailing, "PREVAILING" },
        { aggrCenter, "CENTER" },
        { aggr95Perc, "PERC95" }
    };

    std::string getKeyStringAggregationMethod(aggregationMethod value);
    aggregationMethod getAggregationMethod(const std::string& value);


#endif // AGGREGATION_H
