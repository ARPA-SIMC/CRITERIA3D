#ifndef IMPORTDATA_H
#define IMPORTDATA_H

    #include <QString>
    #ifndef WELL_H
        #include "well.h"
    #endif
    #ifndef GIS_H
        #include "gis.h"
    #endif

    bool loadWaterTableLocationCsv(const QString &csvFileName, std::vector<Well> &wellList,
                               const gis::Crit3DGisSettings &gisSettings, QString &errorStr, int &wrongLines);

    bool loadWaterTableDepthCsv(const QString &csvFileName, std::vector<Well> &wellList, int waterTableMaximumDepth,
                                QString &errorStr, int &wrongLines);

    bool loadCsvDepthsSingleWell(const QString &csvFileName, Well* well, int waterTableMaximumDepth, QString &errorStr, int &wrongLines);

#endif // IMPORTDATA_H

