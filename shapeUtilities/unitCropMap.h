#ifndef UNITCROPMAP_H
#define UNITCROPMAP_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #include <QString>

    bool computeUcmPrevailing(Crit3DShapeHandler &ucm, Crit3DShapeHandler &crop, Crit3DShapeHandler &soil, Crit3DShapeHandler &meteo,
                 std::string idCrop, std::string idSoil, std::string idMeteo, double cellSize, double threshold,
                 QString ucmFileName, std::string &error, bool showInfo);

    bool fillUcmIdCase(Crit3DShapeHandler &ucm, std::string idCrop, std::string idSoil, std::string idMeteo);

    bool writeUcmListToDb(Crit3DShapeHandler& shapeHandler, QString dbName, QString &error);

#endif // UNITCROPMAP_H
