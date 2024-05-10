#ifndef IMPORTDATA_H
#define IMPORTDATA_H

#include <QString>
#ifndef WELL_H
    #include "well.h"
#endif

bool loadCsvRegistry(QString csvRegistry, QList<Well> *wellList, QString *errorStr);

#endif // IMPORTDATA_H

