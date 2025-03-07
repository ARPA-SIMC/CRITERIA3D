#ifndef SHAPEUTILITIES_H
#define SHAPEUTILITIES_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #include <QString>

    QString cloneShapeFile(QString refFileName, QString newFileName);
    QString copyShapeFile(QString refFileName, QString newFileName);

    bool cleanShapeFile(Crit3DShapeHandler &shapeHandler);

    bool computeAnomaly(Crit3DShapeHandler *shapeAnomaly, Crit3DShapeHandler *shape1, Crit3DShapeHandler *shape2,
                        std::string id, std::string field1, std::string field2,
                        QString fileName, QString &errorStr);

#endif // SHAPEUTILITIES_H
