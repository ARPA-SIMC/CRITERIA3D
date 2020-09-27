#ifndef SHAPEUTILITIES_H
#define SHAPEUTILITIES_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #include <QString>

    QString cloneShapeFile(QString refFileName, QString newFileName);
    QString copyShapeFile(QString refFileName, QString newFileName);
    bool cleanShapeFile(Crit3DShapeHandler &shapeHandler);

#endif // SHAPEUTILITIES_H
