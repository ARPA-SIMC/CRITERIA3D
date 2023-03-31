#ifndef FILEUTILITY_H
#define FILEUTILITY_H

    class QString;
    struct ToutputDailyMeteo;
    struct TinputObsData;
    #include <vector>

    bool readMeteoDataCsv (QString namefile, char valuesSeparator,
                          double noData,  TinputObsData* inputData);

    bool writeMeteoDataCsv (QString namefile, char valueSeparator,
                           ToutputDailyMeteo* mydailyData, long dataLenght);

    bool writeMeteoDataCsv(QString namefile, char valueSeparator,
                           std::vector<ToutputDailyMeteo>& dailyData);


#endif // FILEUTILITY_H

