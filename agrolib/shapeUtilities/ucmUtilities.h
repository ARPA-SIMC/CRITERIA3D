#ifndef UCMUTILITIES_H
#define UCMUTILITIES_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #include <QString>

    bool writeUcmListToDb(Crit3DShapeHandler* shapeHandler, QString dbName, std::string *error);
    bool shapeFromCsv(Crit3DShapeHandler* refShapeFile, Crit3DShapeHandler* outputShapeFile, QString csvFileName,
                      QString fieldListFileName, QString outputFileName, QString &error);

#endif // UCMUTILITIES_H
