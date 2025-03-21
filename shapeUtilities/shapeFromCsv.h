#ifndef SHAPEFROMCSV_H
#define SHAPEFROMCSV_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #include <QString>
    #include <QMap>

    long getFileLenght(const QString &fileName, QString &errorStr);

    bool getShapeFieldList(const QString &fileName, QMap<QString, QList<QString>>& fieldList, QString &error);

    bool shapeFromCsv(const Crit3DShapeHandler &refShapeFile, const QString &csvFileName,
                      const QString &fieldListFileName, QString &outputFileName, QString &errorStr);

#endif // SHAPEFROMCSV_H
