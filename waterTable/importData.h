#ifndef IMPORTDATA_H
#define IMPORTDATA_H

#include <QString>
#ifndef WELL_H
    #include "well.h"
#endif

bool loadWaterTableLocationCsv(const QString &csvFileName, std::vector<Well> &wellList, QString &errorStr, int &wrongLines);
bool loadWaterTableDepthCsv(const QString &csvFileName, std::vector<Well> &wellList, int waterTableMaximumDepth, QString &errorStr, int &wrongLines);
bool loadCsvDepthsSingleWell(QString csvDepths, Well* well, int waterTableMaximumDepth, QDate climateObsFirstDate, QDate climateObsLastDate, QString &errorStr, int &wrongLines);

#endif // IMPORTDATA_H

