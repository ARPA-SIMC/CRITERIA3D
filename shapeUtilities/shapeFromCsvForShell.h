#ifndef SHAPEFROMCSVFORSHELL_H
#define SHAPEFROMCSVFORSHELL_H

#ifndef SHAPEHANDLER_H
    #include "shapeHandler.h"
#endif
#include <QString>

bool shapeFromCsvForShell(Crit3DShapeHandler* shapeHandler, Crit3DShapeHandler* outputShape, QString fileCsv, QString fileCsvRef, QString outputName, std::string *error);

#endif // SHAPEFROMCSVFORSHELL_H
