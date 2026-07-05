#ifndef SHAPEUTILITIES_H
#define SHAPEUTILITIES_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #include <QString>

    QString cloneShapeFile(const QString &refFileName, const QString &newFileName);

    QString copyShapeFile(const QString &refFileName, const QString &newFileName);

    bool cleanShapeFile(Crit3DShapeHandler &shapeHandler);

    bool computeAnomaly(Crit3DShapeHandler *shapeAnomaly, Crit3DShapeHandler *shape1, Crit3DShapeHandler *shape2,
                        const std::string &idStr, const std::string &field1, const std::string &field2,
                        const QString &fileName, QString &errorStr);

#endif // SHAPEUTILITIES_H
