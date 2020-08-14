#include "ucmUtilities.h"
#include "shapeUtilities.h"
#include "ucmDb.h"
#include "commonConstants.h"

#include <QtSql>
#include "formInfo.h"



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
bool shapeFromCsv(Crit3DShapeHandler* refShapeFile, Crit3DShapeHandler* outputShapeFile, QString csvFileName,
                  QString fieldListFileName, QString outputFileName, QString &error, bool showInfo)
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

    long nrRows = getFileLenght(csvFileName);
    if (nrRows < 2)
    {
        error = "CSV data is void: " + csvFileName;
        return false;
    }

    QFile file(csvFileName);
    if ( !file.open(QFile::ReadOnly | QFile::Text) )
    {
        error = "CSV data not exists: " + csvFileName;
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

    // show info
    FormInfo formInfo;
    long count = 0;
    int step = 0;
    if (showInfo)
    {
        step = formInfo.start("Create shape... ", nrRows);
    }

    // Reads the data and write to shapefile
    QString line;
    QStringList items;
    std::string idCaseStr;
    int nrShapes = outputShapeFile->getShapeCount();
    QMapIterator<int, int> iterator(myPosMap);

    while (!inputStream.atEnd())
    {
        count++;
        if (showInfo && (count%step == 0)) formInfo.setValue(count);

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

    if (showInfo) formInfo.close();

    file.close();
    return true;
}



/*
bool shapeFromCsvGUI(Crit3DShapeHandler* shapeHandler, Crit3DShapeHandler* outputShape,
                  QString fileCsv, QString fileCsvRef, QString outputName, std::string *error, bool showInfo)
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

    long nrRows = getFileLenght(fileCsv);
    if (nrRows == 0) return false;

    QFile file(fileCsv);
    if ( !file.open(QFile::ReadOnly | QFile::Text) )
    {
        qDebug() << "data file not exists";
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

    FormInfo formInfo;
    long count = 0;
    int step = 0;
    if (showInfo)
    {
        step = formInfo.start("create shape... ", nrRows);
    }

    // Reads the data up to the end of file
    QString line;
    QStringList items;
    while (!inputStream.atEnd())
    {
        count++;
        if (showInfo && (count%step == 0)) formInfo.setValue(count);

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

    if (showInfo) formInfo.close();

    file.close();
    return true;
}
*/


