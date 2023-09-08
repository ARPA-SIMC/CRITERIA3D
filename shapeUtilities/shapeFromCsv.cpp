#include "shapeFromCsv.h"
#include "shapeUtilities.h"
#include "commonConstants.h"

#include <math.h>
#include <iostream>
#include <fstream>

#include <QtSql>

long getFileLenght(QString fileName)
{
    QFile file(fileName);
    if ( !file.open(QFile::ReadOnly | QFile::Text) )
    {
        qDebug() << "data file not exists";
        return 0;
    }

    QTextStream inputStream(&file);

    long nrRows = 0;
    while( !inputStream.atEnd())
    {
        inputStream.readLine();
        nrRows++;
    }

    file.close();

    return nrRows;
}


bool getFieldList(QString fieldListFileName, QMap<QString, QList<QString>>& fieldList, QString &error)
{
    int requiredItems = 5;

    // check fieldList
    if (fieldListFileName.isEmpty())
    {
        error = "Missing field list.";
        return false;
    }

    QFile fileRef(fieldListFileName);
    if ( !fileRef.open(QFile::ReadOnly | QFile::Text) ) {
        error = "Field list not exists: " + fieldListFileName;
        return false;
    }

    QTextStream in(&fileRef);
    // skip header
    QString line = in.readLine();
    while (!in.atEnd())
    {
        QString line = in.readLine();
        QList<QString> items = line.split(",");
        if (items.size() < requiredItems)
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
        fieldList.insert(key,items);
    }

    return true;
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
bool shapeFromCsv(Crit3DShapeHandler &refShapeFile, QString csvFileName,
                  QString fieldListFileName, QString outputFileName, QString &error)
{
    int defaultStringLenght = 20;
    int defaultDoubleLenght = 10;
    int defaultDoubleDecimals = 2;

    // check csv data
    long nrRows = getFileLenght(csvFileName);
    if (nrRows < 2)
    {
        error = "CSV data file is void: " + csvFileName;
        return false;
    }

    QFile csvFile(csvFileName);
    if ( !csvFile.open(QFile::ReadOnly | QFile::Text) )
    {
        error = "CSV data file not exists: " + csvFileName;
        return false;
    }

    // make a copy of shapefile and return cloned shapefile complete path
    QString refShapeFileName = QString::fromStdString(refShapeFile.getFilepath());
    cloneShapeFile(refShapeFileName, outputFileName);
    Crit3DShapeHandler outputShapeFile;
    if (!outputShapeFile.open(outputFileName.toStdString()))
    {
        error = "Load shapefile failed: " + outputFileName;
        return false;
    }

    // Create a thread to retrieve data from a file
    QTextStream inputStream(&csvFile);

    // read first row (header)
    QString firstRow = inputStream.readLine();
    QList<QString> newFields = firstRow.split(",");

    // read field list
    QMap<QString, QList<QString>> fieldList;
    if (fieldListFileName.isEmpty())
    {
        // fill fieldList with default values (call from GEO)
        QString key = newFields.last();
        QList<QString> items;
        items << "outputVar" << "FLOAT" << "8";

        // fraction of available water [0-1] requires 3 decimal digits
        if (key == "FRACTION_AW" || key.left(3) == "FAW")
        {
            items << "3";
        }
        else
        {
            items << "1";
        }
        fieldList.insert(key, items);
    }
    else
    {
        if (! getFieldList(fieldListFileName, fieldList, error))
            return false;
    }

    int type;
    int nWidth;
    int nDecimals;

    QMap<int, int> myPosMap;

    int idCaseIndexShape = outputShapeFile.getFieldPos("ID_CASE");
    int idCaseIndexCsv = NODATA;

    for (int i = 0; i < newFields.size(); i++)
    {
        if (newFields[i] == "ID_CASE")
        {
            idCaseIndexCsv = i;
        }
        if (fieldList.contains(newFields[i]))
        {
            QList<QString> valuesList = fieldList.value(newFields[i]);
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
                if (valuesList[1] == "INTEGER")
                {
                    type = FTInteger;
                    if (valuesList[2].isEmpty())
                    {
                        nWidth = defaultDoubleLenght;
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
            }

            outputShapeFile.addField(field.toStdString().c_str(), type, nWidth, nDecimals);
            myPosMap.insert(i,outputShapeFile.getFieldPos(field.toStdString()));
        }
    }

    if (idCaseIndexCsv == NODATA)
    {
        error = "invalid CSV: missing ID_CASE";
        return false;
    }

    // Reads the data and write to output shapefile
    QString line;
    QList<QString> items;
    QString idCase;
    std::string idCaseStr;
    int nrShapes = outputShapeFile.getShapeCount();
    QMapIterator<int, int> iterator(myPosMap);

    int step = nrRows * 0.01;
    int currentRow = 0;
    while (!inputStream.atEnd())
    {
        // counter
        if (currentRow % step == 0)
        {
            int percentage = round(currentRow * 100.0 / nrRows);
            std::cout << percentage << "..";
        }

        line = inputStream.readLine();
        items = line.split(",");
        idCase = items[idCaseIndexCsv];
        idCaseStr = idCase.toStdString();

        for (int shapeIndex = 0; shapeIndex < nrShapes; shapeIndex++)
        {
            // check ID_CASE
            if (outputShapeFile.readStringAttribute(shapeIndex, idCaseIndexShape) == idCaseStr)
            {
                iterator.toFront();
                while (iterator.hasNext())
                {
                    iterator.next();
                    QString valueToWrite = items[iterator.key()];
                    bool writeOK = false;
                    if (outputShapeFile.getFieldType(iterator.value()) == FTString)
                    {
                        writeOK = outputShapeFile.writeStringAttribute(shapeIndex, iterator.value(), valueToWrite.toStdString().c_str());
                    }
                    else
                    {
                        writeOK = outputShapeFile.writeDoubleAttribute(shapeIndex, iterator.value(), valueToWrite.toDouble());
                    }
                    if (!writeOK)
                    {
                        error = "Error in write this cases: " + idCase;
                        outputShapeFile.close();
                        csvFile.close();
                        return false;
                    }
                }
            }
        }

        currentRow++;
        if (currentRow == (nrRows-1))
        {
            std::cout << "100\n";
        }
    }

    outputShapeFile.close();
    csvFile.close();
    return true;
}

