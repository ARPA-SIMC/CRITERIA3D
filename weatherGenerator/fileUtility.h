#ifndef FILEUTILITY_H
#define FILEUTILITY_H

    #include <vector>
    #include <QString>

    struct ToutputDailyMeteo;
    struct TinputObsData;

    bool readMeteoDataCsv (QString &fileName, char separator, double noData, TinputObsData &inputData);

    bool writeMeteoDataCsv(QString &fileName, char separator, std::vector<ToutputDailyMeteo> &dailyData);


#endif // FILEUTILITY_H

