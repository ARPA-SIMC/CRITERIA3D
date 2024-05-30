#ifndef IMPORTDATA_H
#define IMPORTDATA_H

#include <QString>
#ifndef WELL_H
    #include "well.h"
#endif

<<<<<<< HEAD
bool loadWaterTableLocationCsv(const QString &csvFileName, std::vector<Well> &wellList, QString &errorStr, int &wrongLines);
bool loadWaterTableDepthCsv(const QString &csvFileName, std::vector<Well> &wellList, int waterTableMaximumDepth, QString &errorStr, int &wrongLines);
bool loadCsvDepthsSingleWell(QString csvDepths, Well well, int waterTableMaximumDepth, QString &errorStr, int &wrongLines);
=======
bool loadCsvRegistry(QString csvRegistry, std::vector<Well> &wellList, QString &errorStr, int &wrongLines);
bool loadCsvDepths(QString csvDepths, std::vector<Well> &wellList, int waterTableMaximumDepth, QString &errorStr, int &wrongLines);
bool loadCsvDepthsSingleWell(QString csvDepths, Well* well, int waterTableMaximumDepth, QDate climateObsFirstDate, QDate climateObsLastDate, QString &errorStr, int &wrongLines);
>>>>>>> 2a84db442537ad935d81f540ab67574523de1a13

#endif // IMPORTDATA_H

