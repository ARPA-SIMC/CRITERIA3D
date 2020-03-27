#ifndef UCMUTILITIES_H
#define UCMUTILITIES_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #include <QString>

    bool writeUCMListToDb(Crit3DShapeHandler* shapeHandler, QString dbName, std::string *error);
    bool shapeFromCSV(Crit3DShapeHandler* shapeHandler, Crit3DShapeHandler* outputShape, QString fileCSV, QString fileCSVRef, QString outputName, std::string *error);

#endif // UCMUTILITIES_H
