#ifndef UCMUTILITIES_H
#define UCMUTILITIES_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #include <QString>

    bool writeUcmListToDb(Crit3DShapeHandler* shapeHandler, QString dbName, std::string *error);
    bool shapeFromCsv(Crit3DShapeHandler* shapeHandler, Crit3DShapeHandler* outputShape, QString fileCsv, QString fileCsvRef, QString outputName, std::string *error, bool showInfo);

#endif // UCMUTILITIES_H
