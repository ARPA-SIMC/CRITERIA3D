#include "commonConstants.h"
#include "shapeUtilities.h"
#include "shapeHandler.h"
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

    if (QFile::exists(newFile + ".shp"))
    {
        if (! QFile::remove(newFile + ".shp"))
        {
            return "";
        }
    }
    QFile::copy(refFile +".shp", newFile +".shp");

    QFile::remove(newFile +".dbf");
    QFile::copy(refFile +".dbf", newFile +".dbf");

    QFile::remove(newFile +".shx");
    QFile::copy(refFile +".shx", newFile +".shx");

    QFile::remove(newFile +".prj");
    QFile::copy(refFile +".prj", newFile +".prj");

    return newFile + ".shp";
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


// shape1 and shape2 must have the same polygons and IDs
bool computeAnomaly(Crit3DShapeHandler *shapeAnomaly, Crit3DShapeHandler *shape1, Crit3DShapeHandler *shape2,
                    std::string id, std::string field1, std::string field2, QString fileName, QString &errorStr)
{
    QString newShapeFileName = cloneShapeFile(QString::fromStdString(shape1->getFilepath()), fileName);
    if (newShapeFileName == "")
    {
        errorStr = "Error in create/open shapefile: " + newShapeFileName;
        return false;
    }

    if (! shapeAnomaly->open(newShapeFileName.toStdString()))
    {
        errorStr = "Error in create/open shapefile: " + newShapeFileName;
        return false;
    }

    // remove fields (not ID)
    int nrFields = shape1->getFieldNumbers();
    for (int i = 0; i < nrFields; i++)
    {
        std::string fieldName = shape1->getFieldName(i);
        if (fieldName != id)
        {
            int fieldPos = shapeAnomaly->getFieldPos(fieldName);
            if (fieldPos != -1)
            {
                if (! shapeAnomaly->removeField(fieldPos))
                {
                    errorStr = "Error in delete field: " + QString::fromStdString(fieldName);
                    return false;
                }
            }
        }
    }

    // add anomaly field
    if (! shapeAnomaly->addField("anomaly", FTDouble, 10, 1))
    {
        errorStr = "Error in create field 'anomaly'.";
        return false;
    }
    int anomalyPos = shapeAnomaly->getFieldPos("anomaly");

    // check field position
    int pos1 = shape1->getFieldPos(field1);
    int pos2 = shape2->getFieldPos(field2);
    if (pos1 == -1 || pos2 == -1)
    {
        errorStr = "Missing field.";
        return false;
    }

    // compute values
    for (int i = 0; i < shape1->getShapeCount(); i++)
    {
        double value1 = shape1->getNumericValue(i, pos1);
        double value2 = shape2->getNumericValue(i, pos2);
        if (value1 != NODATA && value2 != NODATA)
        {
            shapeAnomaly->writeDoubleAttribute(i, anomalyPos, value2 - value1);
        }
    }

    return true;
}
