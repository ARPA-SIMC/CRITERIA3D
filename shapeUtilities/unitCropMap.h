#ifndef UNITCROPMAP_H
#define UNITCROPMAP_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #include <QString>

    bool computeUcmPrevailing(Crit3DShapeHandler &ucm, Crit3DShapeHandler &crop, Crit3DShapeHandler &soil, Crit3DShapeHandler &meteo,
                 std::string idCrop, std::string idSoil, std::string idMeteo, double cellSize,
                 QString ucmFileName, std::string &error, bool showInfo);

    bool fillIDCase(Crit3DShapeHandler *ucm, std::string idCrop, std::string idSoil, std::string idMeteo);

#endif // UNITCROPMAP_H
