#include <shapelib/shapefil.h>
#include "shapeHandler.h"
#include "shapeFromCsv.h"
#include "shapeUtilities.h"
#include "commonConstants.h"
#include "basicMath.h"

#include <math.h>
#include <iostream>
#include <fstream>

#include <QFile>
#include <QTextStream>


long getFileLenght(const QString &fileName, QString &errorStr)
{
    QFile file(fileName);
    if (! file.open(QFile::ReadOnly | QFile::Text) )
    {
        errorStr = "data file not exists";
        return 0;
    }

    QTextStream inputStream(&file);

    long nrRows = 0;
    while(! inputStream.atEnd())
    {
        inputStream.readLine();
        nrRows++;
    }

    file.close();

    return nrRows;
}


bool getShapeFieldList(const QString &fileName, QMap<QString, QList<QString>> &fieldList, QString &error)
{
    int requiredItems = 5;

    // check fieldList
    if (fileName.isEmpty())
    {
        error = "Missing field list.";
        return false;
    }

    QFile fileRef(fileName);
    if (! fileRef.open(QFile::ReadOnly | QFile::Text) )
    {
        error = "Field list not exists: " + fileName;
        return false;
    }

    QTextStream in(&fileRef);
    // skip header
    in.readLine();

    while (! in.atEnd())
    {
        QString line = in.readLine();
        QList<QString> items = line.split(",");
        if (items.size() < requiredItems)
        {
            error = "Invalid field list, missing parameters in line: " + line;
            return false;
        }
        if (items[0].isEmpty() || items[1].isEmpty())
        {
            error = "Invalid field list, missing field name in line: " + line;
            return false;
        }

        QString key = items[1];
        items.removeAt(1);

        fieldList.insert(key,items);
    }

    return true;
}


// call from Criteria GEO: fieldsList will be filled with default values
bool prepareFieldsList(const QString &keyVariable, QMap<QString, QList<QString>> &fieldsList)
{
    QList<QString> items;
    items << "outputVar" << "FLOAT" << "9";

    // decimal digits
    // fraction [0-1] requires 3 decimal digits
    if (keyVariable == "FRACTION_AW" || keyVariable.left(3) == "FAW" || keyVariable.left(3) == "SWI")
    {
        items << "3";
    }
    else
    {
        items << "1";
    }
    fieldsList.insert(keyVariable, items);

    return true;
}


/*! shapeFromCsv
 * \brief import data on a shapeFile from a csv
 *
 * \param refShapeFile is the handler to reference shapeFile (will be cloned)
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
bool shapeFromCsv(const Crit3DShapeHandler &refShapeFile, const QString &csvFileName,
                  const QString &fieldListFileName, QString &outputFileName, QString &errorStr)
{
    int defaultStringLenght = 20;
    int defaultDoubleLenght = 10;
    int defaultDoubleDecimals = 2;

    // check csv data
    long nrRows = getFileLenght(csvFileName, errorStr);
    if (nrRows < 2)
    {
        errorStr = "CSV data file is empty: " + csvFileName;
        return false;
    }

    QFile csvFile(csvFileName);
    if (! csvFile.open(QFile::ReadOnly | QFile::Text))
    {
        errorStr = "CSV data file not exists: " + csvFileName;
        return false;
    }

    // make a copy of shapefile and return cloned shapefile complete path
    QString refShapeFileName = QString::fromStdString(refShapeFile.getFilepath());
    outputFileName = cloneShapeFile(refShapeFileName, outputFileName);
    if (outputFileName == "")
    {
        errorStr = "Error in create/open shapefile: " + outputFileName;
        return false;
    }

    Crit3DShapeHandler outputShapeFile;
    if (! outputShapeFile.open(outputFileName.toStdString()))
    {
        errorStr = "Load shapefile failed: " + outputFileName;
        return false;
    }

    // Create a thread to retrieve data from a file
    QTextStream inputCsvStream(&csvFile);

    // read first row (header)
    QString firstRow = inputCsvStream.readLine();
    QList<QString> headerList = firstRow.split(",");

    // read field list
    QMap<QString, QList<QString>> fieldsList;
    if (fieldListFileName.isEmpty())
    {
        // filename doesn't exist (call from GEO)
        QString variable = headerList.last();
        prepareFieldsList (variable, fieldsList);
    }
    else
    {
        if (! getShapeFieldList(fieldListFileName, fieldsList, errorStr))
        {
            errorStr += "\nError in reading file: " + fieldListFileName;
            return false;
        }
    }

    int type, nWidth, nDecimals;
    QMap<int, int> myPosMap;

    int idCaseIndexShape = outputShapeFile.getFieldPos("ID_CASE");
    bool isIdCasePresent = false;
    int idCaseIndex = NODATA;

    for (int i = 0; i < headerList.size(); i++)
    {
        if (headerList[i] == "ID_CASE")
        {
            isIdCasePresent = true;
            idCaseIndex = i;
        }
        if (fieldsList.contains(headerList[i]))
        {
            QList<QString> valuesList = fieldsList.value(headerList[i]);
            QString field = valuesList[0];
            if (valuesList[1] == "STRING" || valuesList[1] == "TEXT")
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
            myPosMap.insert(i, outputShapeFile.getFieldPos(field.toStdString()));
        }
    }

    if (! isIdCasePresent)
    {
        errorStr = "Invalid CSV: missing ID_CASE";
        return false;
    }

    // Reads the data and write to output shapefile
    QString line;
    QList<QString> items;
    QString idCase;
    std::string idCaseStr;
    int nrShapes = outputShapeFile.getShapeCount();
    QMapIterator<int, int> iterator(myPosMap);

    // save the attribute idCase in a list (to speed search)
    std::vector<std::string> idCaseList;
    idCaseList.resize(nrShapes);
    for (int i = 0; i < nrShapes; i++)
    {
        idCaseList[i] = outputShapeFile.readStringAttribute(i, idCaseIndexShape);
    }

    // main cycle
    int step = nrRows * 0.1;
    int currentRow = 0;
    while (! inputCsvStream.atEnd())
    {
        // counter
        if (currentRow % step == 0)
        {
            int percentage = round(currentRow * 100. / nrRows);
            std::cout << percentage << "...";
        }

        line = inputCsvStream.readLine();
        items = line.split(",");
        idCase = items[idCaseIndex];
        idCaseStr = idCase.toStdString();

        for (int shapeIndex = 0; shapeIndex < nrShapes; shapeIndex++)
        {
            // check ID_CASE
            if (idCaseList[shapeIndex] == idCaseStr)
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
                    if (! writeOK)
                    {
                        errorStr = "Error in write this cases: " + idCase;
                        outputShapeFile.close();
                        csvFile.close();
                        return false;
                    }
                }
            }
        }
        currentRow++;
    }

    outputShapeFile.close();
    csvFile.close();

    std::cout << " done.\n";
    return true;
}

