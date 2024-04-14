#ifndef SHAPEFROMCSV_H
#define SHAPEFROMCSV_H

    #ifndef SHAPEHANDLER_H
        #include "shapeHandler.h"
    #endif
    #include <QString>

    bool getFieldList(QString fieldListFileName, QMap<QString, QList<QString>>& fieldList, QString &error);

    bool shapeFromCsv(Crit3DShapeHandler &refShapeFile, QString csvFileName,
                      QString fieldListFileName, QString outputFileName, QString &errorStr);

#endif // SHAPEFROMCSV_H
