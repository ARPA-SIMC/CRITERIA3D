#include "shapeFromCsvForShell.h"
#include "shapeUtilities.h"
#include "commonConstants.h"

#include <QtSql>

/* output format file:
 * CsvFieldName, ShapeFieldName, type, lenght, decimals nr
 * example:
 * CROP,CROP,STRING,20,
 * deficit,DEFICIT,FLOAT,10,1
 * forecast7daysIRR,FcstIrr7d,FLOAT,10,1
*/
bool shapeFromCsvForShell(Crit3DShapeHandler* shapeHandler, Crit3DShapeHandler* outputShape,
                  QString fileCsv, QString fileCsvRef, QString outputName, std::string *error)
{
    int csvRefRequiredInfo = 5;
    int defaultStringLenght = 20;
    int defaultDoubleLenght = 10;
    int defaultDoubleDecimals = 2;

    QString refFileName = QString::fromStdString(shapeHandler->getFilepath());
    QFileInfo csvFileInfo(fileCsv);
    QFileInfo refFileInfo(refFileName);

    // make a copy of shapefile and return cloned shapefile complete path
    QString ucmShapeFile = cloneShapeFile(refFileName, outputName);

    if (!outputShape->open(ucmShapeFile.toStdString()))
    {
        *error = "Load shapefile failed: " + ucmShapeFile.toStdString();
        return false;
    }

    // read fileCsvRef and fill MapCsvShapeFields
    QMap<QString, QStringList> MapCsvShapeFields;
    QFile fileRef(fileCsvRef);
    if ( !fileRef.open(QFile::ReadOnly | QFile::Text) ) {
        qDebug() << "File not exists";
    }
    else
    {
        QTextStream in(&fileRef);
        while (!in.atEnd())
        {
            QString line = in.readLine();
            QStringList items = line.split(",");
            if (items.size() < csvRefRequiredInfo)
            {
                *error = "invalid output format CSV, missing reference data";
                return false;
            }
            QString key = items[0];
            items.removeFirst();
            if (key.isEmpty() || items[0].isEmpty())
            {
                *error = "invalid output format CSV, missing field name";
                return false;
            }
            MapCsvShapeFields.insert(key,items);
        }
    }

    int nShape = outputShape->getShapeCount();

    QFile file(fileCsv);
    if ( !file.open(QFile::ReadOnly | QFile::Text) )
    {
        qDebug() << "data file not exists:" << fileCsv;
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

    int idCaseIndex = outputShape->getFieldPos("ID_CASE");
    int idCaseCsv = NODATA;

    for (int i = 0; i < newFields.size(); i++)
    {
        if (newFields[i] == "ID_CASE")
        {
            idCaseCsv = i;
        }
        if (MapCsvShapeFields.contains(newFields[i]))
        {
            QStringList valuesList = MapCsvShapeFields.value(newFields[i]);
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
            outputShape->addField(field.toStdString().c_str(), type, nWidth, nDecimals);
            myPosMap.insert(i,outputShape->getFieldPos(field.toStdString()));
        }

    }

    if (idCaseCsv == NODATA)
    {
        *error = "invalid CSV, missing ID_CASE";
        return false;
    }


    // Reads the data up to the end of file
    QString line;
    QStringList items;
    while (!inputStream.atEnd())
    {

        line = inputStream.readLine();
        items = line.split(",");

        for (int shapeIndex = 0; shapeIndex < nShape; shapeIndex++)
        {
            // check right ID_CASE
            if (outputShape->readStringAttribute(shapeIndex, idCaseIndex) == items[idCaseCsv].toStdString())
            {
                QMapIterator<int, int> i(myPosMap);
                while (i.hasNext()) {
                    i.next();
                    QString valueToWrite = items[i.key()];
                    if (outputShape->getFieldType(i.value()) == FTString)
                    {
                        outputShape->writeStringAttribute(shapeIndex, i.value(), valueToWrite.toStdString().c_str());
                    }
                    else
                    {
                        outputShape->writeDoubleAttribute(shapeIndex, i.value(), valueToWrite.toDouble());
                    }
                }
            }
        }

    }

    file.close();
    return true;
}
