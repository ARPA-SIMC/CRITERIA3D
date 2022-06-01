#ifndef OUTPUTPOINTS_H
#define OUTPUTPOINTS_H

    #ifndef GIS_H
        #include "gis.h"
    #endif
    #include <QString>

    bool importOutputPointsCsv(QString csvFileName, QList<QList<QString>> &data, QString &errorString);

    bool loadOutputPointListCsv(QString csvFileName, std::vector<gis::Crit3DOutputPoint> &outputPointList,
                             int utmZone, QString &errorString);

    bool writeOutputPointListCsv(QString csvFileName, std::vector<gis::Crit3DOutputPoint> &outputPointList, QString &errorString);


#endif // OUTPUTPOINTS_H
