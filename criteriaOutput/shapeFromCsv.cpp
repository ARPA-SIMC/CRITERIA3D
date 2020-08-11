/*! shapeFromCsv
* import data on a shapeFile from a csv
*
* This file is part of CRITERIA1D distribution
*/


#include "shapeFromCsv.h"
#include "commonConstants.h"
#include <QtSql>


/*! cloneShapeFile
 * \brief make a copy of shapefile and return filename of the cloned shapefile
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


/*! shapeFromCsv
 * \brief import data on a shapeFile from a csv
 *
 * \param refShapeFile is the handler to reference shapeFile (will be cloned)
 * \param outputShapeFile is the handler to output shapeFile
 * \param csvFileName is the filename od input data (csv)
 * \param fieldListFileName is the filename of the field list to export (csv)
 * \param outputFileName is the filename of output shapefile
 *
 * \details fieldList format:
 * output field (shapefile), input variable (csv), field type, field lenght, decimals nr.
 * example:
 * CropName,CROP,STRING,20,
 * FcstIrr7d,forecast7daysIRR,FLOAT,10,1
 *
 * \return true if all is correct
*/
bool shapeFromCsv(Crit3DShapeHandler* refShapeFile, Crit3DShapeHandler* outputShapeFile,
                  QString csvFileName, QString fieldListFileName, QString outputFileName, QString &error)
{
    int csvRefRequiredInfo = 5;
    int defaultStringLenght = 20;
    int defaultDoubleLenght = 10;
    int defaultDoubleDecimals = 2;

    QString refFileName = QString::fromStdString(refShapeFile->getFilepath());
    QFileInfo csvFileInfo(csvFileName);
    QFileInfo refFileInfo(refFileName);

    // make a copy of shapefile and return cloned shapefile complete path
    QString ucmShapeFile = cloneShapeFile(refFileName, outputFileName);

    if (!outputShapeFile->open(ucmShapeFile.toStdString()))
    {
        error = "Load shapefile failed: " + ucmShapeFile;
        return false;
    }

    // read fieldList and fill mapCsvShapeFields
    QMap<QString, QStringList> mapCsvShapeFields;
    QFile fileRef(fieldListFileName);
    if ( !fileRef.open(QFile::ReadOnly | QFile::Text) ) {
        error = "Field list not exists: " + fieldListFileName;
        return false;
    }
    else
    {
        QTextStream in(&fileRef);
        // skip header
        QString line = in.readLine();
        while (!in.atEnd())
        {
            QString line = in.readLine();
            QStringList items = line.split(",");
            if (items.size() < csvRefRequiredInfo)
            {
                error = "invalid field list: missing parameters";
                return false;
            }
            QString key = items[1];
            items.removeAt(1);
            if (key.isEmpty() || items[0].isEmpty())
            {
                error = "invalid field list: missing field name";
                return false;
            }
            mapCsvShapeFields.insert(key,items);
        }
    }

    QFile file(csvFileName);
    if ( !file.open(QFile::ReadOnly | QFile::Text) )
    {
        error = "CSV data not exists: " + csvFileName;
        return false;
    }

    // Create a thread to retrieve data from a file
    QTextStream inputStream(&file);
    // Reads first row
    QString firstRow = inputStream.readLine();
    QStringList newFields = firstRow.split(",");

    int type;
    int nWidth;
    int nDecimals;

    QMap<int, int> myPosMap;

    int idCaseIndexShape = outputShapeFile->getFieldPos("ID_CASE");
    int idCaseIndexCsv = NODATA;

    for (int i = 0; i < newFields.size(); i++)
    {
        if (newFields[i] == "ID_CASE")
        {
            idCaseIndexCsv = i;
        }
        if (mapCsvShapeFields.contains(newFields[i]))
        {
            QStringList valuesList = mapCsvShapeFields.value(newFields[i]);
            QString field = valuesList[0];
            if (valuesList[1] == "STRING")
            {
                type = FTString;
                if (valuesList[2].isEmpty())
                {
                    nWidth = defaultStringLenght;
                }
                else
                {
                    nWidth = valuesList[2].toInt();
                }
                nDecimals = 0;
            }
            else
            {
                type = FTDouble;
                if (valuesList[2].isEmpty())
                {
                    nWidth = defaultDoubleLenght;
                }
                else
                {
                    nWidth = valuesList[2].toInt();
                }
                if (valuesList[3].isEmpty())
                {
                    nDecimals = defaultDoubleDecimals;
                }
                else
                {
                    nDecimals = valuesList[3].toInt();
                }
            }
            outputShapeFile->addField(field.toStdString().c_str(), type, nWidth, nDecimals);
            myPosMap.insert(i,outputShapeFile->getFieldPos(field.toStdString()));
        }
    }

    if (idCaseIndexCsv == NODATA)
    {
        error = "invalid CSV: missing ID_CASE";
        return false;
    }

    // Reads the data up to the end of file
    QString line;
    QStringList items;
    std::string idCaseStr;
    int nrShapes = outputShapeFile->getShapeCount();
    QMapIterator<int, int> iterator(myPosMap);

    while (!inputStream.atEnd())
    {
        line = inputStream.readLine();
        items = line.split(",");
        idCaseStr = items[idCaseIndexCsv].toStdString();

        for (int shapeIndex = 0; shapeIndex < nrShapes; shapeIndex++)
        {
            // check ID_CASE
            if (outputShapeFile->readStringAttribute(shapeIndex, idCaseIndexShape) == idCaseStr)
            {
                iterator.toFront();
                while (iterator.hasNext())
                {
                    iterator.next();
                    QString valueToWrite = items[iterator.key()];
                    if (outputShapeFile->getFieldType(iterator.value()) == FTString)
                    {
                        outputShapeFile->writeStringAttribute(shapeIndex, iterator.value(), valueToWrite.toStdString().c_str());
                    }
                    else
                    {
                        outputShapeFile->writeDoubleAttribute(shapeIndex, iterator.value(), valueToWrite.toDouble());
                    }
                }
            }
        }
    }

    file.close();
    return true;
}
