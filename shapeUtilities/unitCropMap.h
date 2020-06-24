#ifndef UNITCROPMAP_H
#define UNITCROPMAP_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #include <QString>

    bool computeUcmPrevailing(Crit3DShapeHandler *ucm, Crit3DShapeHandler *crop, Crit3DShapeHandler *soil, Crit3DShapeHandler *meteo,
                        std::string idCrop, std::string idSoil, std::string idMeteo, double cellSize, QString ucmFileName,
                        std::string *error, bool showInfo);
    bool computeUcmIntersection(Crit3DShapeHandler *ucm, Crit3DShapeHandler *crop, Crit3DShapeHandler *soil, Crit3DShapeHandler *meteo,
                     std::string idCrop, std::string idSoil, std::string idMeteo,
                     QString ucmFileName, std::string *error, bool showInfo);
    // bool shapeIntersection(Crit3DShapeHandler *intersecHandler, Crit3DShapeHandler *firstHandler, Crit3DShapeHandler *secondHandler, std::string fieldNameFirst, std::string fieldNameSecond, std::string *error, bool showInfo);
    bool fillIDCase(Crit3DShapeHandler *ucm, std::string idCrop, std::string idSoil, std::string idMeteo);


#endif // UNITCROPMAP_H
