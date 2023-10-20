#ifndef FILEUTILITY_H
#define FILEUTILITY_H

    struct ToutputDailyMeteo;
    struct TinputObsData;
    #include <vector>
    #include <QString>

    bool readMeteoDataCsv (QString &fileName, char separator, double noData, TinputObsData* inputData);

    bool writeMeteoDataCsv(QString &fileName, char separator, std::vector<ToutputDailyMeteo> &dailyData);


#endif // FILEUTILITY_H

