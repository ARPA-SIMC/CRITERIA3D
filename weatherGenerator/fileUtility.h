#ifndef FILEUTILITY_H
#define FILEUTILITY_H

    class QString;
    struct ToutputDailyMeteo;
    struct TinputObsData;

    bool readMeteoDataCsv (QString namefile, char separator, double noData,  TinputObsData* inputData);

    bool writeMeteoDataCsv (QString namefile, char separator, ToutputDailyMeteo* mydailyData, long dataLenght);

#endif // FILEUTILITY_H

