#include "shapeUtilities.h"
#include <QFile>
#include <QFileInfo>


/*! cloneShapeFile
 * \brief make a copy of shapefile (delete old version)
 * and return filename of the cloned shapefile
 */
QString cloneShapeFile(QString refFileName, QString newFileName)
{
    QFileInfo refFileInfo(refFileName);
    QFileInfo newFileInfo(newFileName);

    QString refFile = refFileInfo.absolutePath() + "/" + refFileInfo.baseName();
    QString newFile = newFileInfo.absolutePath() + "/" + newFileInfo.baseName();

    QFile::remove(newFile + ".dbf");
    QFile::copy(refFile +".dbf", newFile +".dbf");

    QFile::remove(newFile +".shp");
    QFile::copy(refFile +".shp", newFile +".shp");

    QFile::remove(newFile +".shx");
    QFile::copy(refFile +".shx", newFile +".shx");

    QFile::remove(newFile +".prj");
    QFile::copy(refFile +".prj", newFile +".prj");

    return(newFile + ".shp");
}

/*! copyShapeFile
 * \brief make a copy of shapefile (keep original version)
 * and return filename of the cloned shapefile
 */
QString copyShapeFile(QString refFileName, QString newFileName)
{
    QFileInfo refFileInfo(refFileName);
    QFileInfo newFileInfo(newFileName);

    QString refFile = refFileInfo.absolutePath() + "/" + refFileInfo.baseName();
    QString newFile = newFileInfo.absolutePath() + "/" + newFileInfo.baseName();

    QFile::copy(refFile +".dbf", newFile +".dbf");
    QFile::copy(refFile +".shp", newFile +".shp");
    QFile::copy(refFile +".shx", newFile +".shx");
    QFile::copy(refFile +".prj", newFile +".prj");

    return(newFile + ".shp");
}


bool cleanShapeFile(Crit3DShapeHandler &shapeHandler)
{
    if (! shapeHandler.existRecordDeleted()) return true;

    QFileInfo fileInfo(QString::fromStdString(shapeHandler.getFilepath()));
    QString refFile = fileInfo.absolutePath() + "/" + fileInfo.baseName();
    QString tmpFile = refFile + "_temp";

    shapeHandler.packSHP(tmpFile.toStdString());
    shapeHandler.packDBF(tmpFile.toStdString());
    shapeHandler.close();

    QFile::remove(refFile + ".dbf");
    QFile::copy(tmpFile + ".dbf", refFile + ".dbf");
    QFile::remove(tmpFile + ".dbf");

    QFile::remove(refFile + ".shp");
    QFile::copy(tmpFile + ".shp", refFile + ".shp");
    QFile::remove(tmpFile + ".shp");

    QFile::remove(refFile + ".shx");
    QFile::copy(tmpFile + ".shx", refFile + ".shx");
    QFile::remove(tmpFile + ".shx");

    return shapeHandler.open(shapeHandler.getFilepath());
}
