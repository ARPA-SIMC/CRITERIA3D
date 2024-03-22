#ifndef OUTPUTPOINTS_H
#define OUTPUTPOINTS_H

    #ifndef GIS_H
        #include "gis.h"
    #endif
    #include <QString>

    bool importPointPropertiesCsv(const QString &csvFileName, QList<QList<QString>> &pointsProperties,
                                  QString &errorString);

    bool loadOutputPointListCsv(const QString &csvFileName, std::vector<gis::Crit3DOutputPoint> &outputPointList,
                                int utmZone, QString &errorString);

    bool writeOutputPointListCsv(const QString &csvFileName, std::vector<gis::Crit3DOutputPoint> &outputPointList,
                                 QString &errorString);


#endif // OUTPUTPOINTS_H
