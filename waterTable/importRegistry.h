#ifndef IMPORTREGISTRY_H
#define IMPORTREGISTRY_H

#include <QString>
#ifndef WELL_H
    #include "well.h"
#endif

bool loadCsvRegistry(QString csvRegistry, QList<Well> *wellList, QString *errorStr);

#endif // IMPORTREGISTRY_H

