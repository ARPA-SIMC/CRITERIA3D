#include "shapeFromCsv.h"
#include "shapeUtilities.h"
#include "commonConstants.h"

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
    int csvRefRequiredInfo = 5;
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
    QTextStream inputStream(&file);
    // read first row (header)
    QString firstRow = inputStream.readLine();
    QStringList newFields = firstRow.split(",");

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
    QStringList items;
    QString idCase;
    std::string idCaseStr;
    int nrShapes = outputShapeFile.getShapeCount();
    QMapIterator<int, int> iterator(myPosMap);

    while (!inputStream.atEnd())
    {
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
                        file.close();
                        return false;
                    }
                }
            }
        }
    }

    outputShapeFile.close();
    file.close();
    return true;
}

