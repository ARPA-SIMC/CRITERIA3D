#ifndef IMPORTDATA_H
#define IMPORTDATA_H

#include <QString>
#ifndef WELL_H
    #include "well.h"
#endif

bool loadCsvRegistry(QString csvRegistry, std::vector<Well> &wellList, QString &errorStr, int &wrongLines);
bool loadCsvDepths(QString csvDepths, std::vector<Well> &wellList, int waterTableMaximumDepth, QString &errorStr, int &wrongLines);

#endif // IMPORTDATA_H

