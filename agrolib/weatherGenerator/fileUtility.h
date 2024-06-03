#ifndef FILEUTILITY_H
#define FILEUTILITY_H

    #include <vector>
    #include <QString>

    struct ToutputDailyMeteo;
    struct TinputObsData;

    bool readMeteoDataCsv (const QString &fileName, char separator, double noData, TinputObsData &inputData);

    bool writeMeteoDataCsv(const QString &fileName, char separator, std::vector<ToutputDailyMeteo> &dailyData, bool isWaterTable);


#endif // FILEUTILITY_H

