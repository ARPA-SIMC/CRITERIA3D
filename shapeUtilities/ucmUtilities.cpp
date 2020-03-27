#include "ucmUtilities.h"
#include "shapeUtilities.h"
#include "ucmDb.h"
#include "commonConstants.h"

#include <QtSql>


bool writeUCMListToDb(Crit3DShapeHandler* shapeHandler, QString dbName, std::string *error)
{
    UcmDb* unitList = new UcmDb(dbName);

    QStringList idCase, idCrop, idMeteo, idSoil;
    QList<double> ha;

    int nShape = shapeHandler->getShapeCount();

    for (int i = 0; i < nShape; i++)
    {
        QString key = QString::fromStdString(shapeHandler->getStringValue(signed(i), "ID_CASE"));
        if (key.isEmpty()) continue;

        if ( !idCase.contains(key) )
        {
            idCase << key;
            idCrop << QString::fromStdString(shapeHandler->getStringValue(signed(i), "ID_CROP"));
            idMeteo << QString::fromStdString(shapeHandler->getStringValue(signed(i), "ID_METEO"));
            idSoil << QString::fromStdString(shapeHandler->getStringValue(signed(i), "ID_SOIL"));
            ha << shapeHandler->getNumericValue(signed(i), "HA");
        }
        else
        {
            // TODO search value and sum ha
        }
    }

    bool res = unitList->writeListToUnitsTable(idCase, idCrop, idMeteo, idSoil, ha);
    *error = unitList->getError().toStdString();

    delete unitList;

    return res;
}


/* output format file:
 * CSVfieldName, ShapeFieldName, type, lenght, decimals nr
 * example:
 * CROP,CROP,STRING,20,
 * deficit,DEFICIT,FLOAT,10,1
 * forecast7daysIRR,FcstIrr7d,FLOAT,10,1
*/
bool shapeFromCSV(Crit3DShapeHandler* shapeHandler, Crit3DShapeHandler* outputShape, QString fileCSV, QString fileCSVRef, QString outputName, std::string *error)
{
    int CSVRefRequiredInfo = 5;
    int defaultStringLenght = 20;
    int defaultDoubleLenght = 10;
    int defaultDoubleDecimals = 2;

    QString refFileName = QString::fromStdString(shapeHandler->getFilepath());
    QFileInfo csvFileInfo(fileCSV);
    QFileInfo refFileInfo(refFileName);

    // make a copy of shapefile and return cloned shapefile complete path
    QString ucmShapeFile = cloneShapeFile(refFileName, outputName);

    if (!outputShape->open(ucmShapeFile.toStdString()))
    {
        *error = "Load shapefile failed: " + ucmShapeFile.toStdString();
        return false;
    }

    // read fileCSVRef and fill MapCSVShapeFields
    QMap<QString, QStringList> MapCSVShapeFields;
    QFile fileRef(fileCSVRef);
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
            if (items.size() < CSVRefRequiredInfo)
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
            MapCSVShapeFields.insert(key,items);
        }
    }

    int nShape = outputShape->getShapeCount();

    QFile file(fileCSV);
    if ( !file.open(QFile::ReadOnly | QFile::Text) ) {
        qDebug() << "File not exists";
    }
    else
    {
        // Create a thread to retrieve data from a file
        QTextStream in(&file);
        // Reads first row
        QString firstRow = in.readLine();
        QStringList newFields = firstRow.split(",");

        int type;
        int nWidth;
        int nDecimals;

        QMap<int, int> myPosMap;

        int idCaseIndex = outputShape->getFieldPos("ID_CASE");
        int idCaseCSV = NODATA;

        for (int i = 0; i < newFields.size(); i++)
        {
            if (newFields[i] == "ID_CASE")
            {
                idCaseCSV = i;
            }
            if (MapCSVShapeFields.contains(newFields[i]))
            {
                QStringList valuesList = MapCSVShapeFields.value(newFields[i]);
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

        if (idCaseCSV == NODATA)
        {
            *error = "invalid CSV, missing ID_CASE";
            return false;
        }

        //Reads the data up to the end of file
        while (!in.atEnd())
        {
            QString line = in.readLine();
            QStringList items = line.split(",");
            //qDebug() << "items " << items;
            for (int shapeIndex = 0; shapeIndex < nShape; shapeIndex++)
            {
                // check right ID_CASE
                if (outputShape->readStringAttribute(shapeIndex, idCaseIndex) == items[idCaseCSV].toStdString())
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
    }

    return true;
}
