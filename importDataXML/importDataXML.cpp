#include "importDataXML.h"
#include "commonConstants.h"
#include <QTextStream>
#include <QFile>
#include <QFileInfo>



ImportDataXML::ImportDataXML(bool isGrid, Crit3DMeteoPointsDbHandler *meteoPointsDbHandler, Crit3DMeteoGridDbHandler *meteoGridDbHandler, QString xmlFileName)
{
    this->isGrid = isGrid;
    this->xmlFileName = xmlFileName;
    this->meteoPointsDbHandler = meteoPointsDbHandler;
    this->meteoGridDbHandler = meteoGridDbHandler;
    this->format_headerRow = 0;
}

bool ImportDataXML::parseXMLFile(QDomDocument* xmlDoc, QString *error)
{
    if (xmlFileName == "")
    {
        *error = "Missing XML file.";
        return false;
    }

    QFile myFile(xmlFileName);
    if (!myFile.open(QIODevice::ReadOnly))
    {
        *error = "Open XML failed: " + xmlFileName + "\n " + myFile.errorString();
        return (false);
    }

    QString myError;
    int myErrLine, myErrColumn;
    if (!xmlDoc->setContent(&myFile, &myError, &myErrLine, &myErrColumn))
    {
       *error = "Parse xml failed:" + xmlFileName
                + " Row: " + QString::number(myErrLine)
                + " - Column: " + QString::number(myErrColumn)
                + "\n" + myError;
        myFile.close();
        return(false);
    }

    myFile.close();
    return true;
}

bool ImportDataXML::parserXML(QString *myError)
{
    QDomDocument xmlDoc;
    if (!parseXMLFile(&xmlDoc, myError)) return false;

    QDomNode child;
    QDomNode secondChild;
    QDomNode thirdChild;

    QDomNode ancestor = xmlDoc.documentElement().firstChild();
    QString myTag;
    QString mySecondTag;
    QString myThirdTag;

    while(!ancestor.isNull())
    {
        if (ancestor.toElement().tagName().toUpper() == "FILENAME")
        {
            child = ancestor.firstChild();
            while( !child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "PATH")
                {
                    fileName_path = child.toElement().text();
                }
                else if (myTag == "FIELD")
                {
                    secondChild = child.firstChild();
                    while( !secondChild.isNull())
                    {
                        mySecondTag = secondChild.toElement().tagName().toUpper();

                        if (mySecondTag == "PRAGANAME" || mySecondTag == "PRAGAFIELD")
                        {
                            fileName_pragaName[fileName_pragaName.size()-1] = secondChild.toElement().text();
                        }
                        else if (mySecondTag == "TEXT" || mySecondTag == "FIXEDTEXT")
                        {
                            fileName_fixedText[fileName_fixedText.size()-1] = secondChild.toElement().text();
                        }
                        else if (mySecondTag == "NRCHAR" || mySecondTag == "NR_CHAR")
                        {
                            fileName_nrChar[fileName_nrChar.size()-1] = secondChild.toElement().text().toInt();
                        }
                    }

                }
                child = child.nextSibling();
            }
        }
        else if (ancestor.toElement().tagName().toUpper() == "FORMAT")
        {
            child = ancestor.firstChild();
            while( !child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "TYPE")
                {
                    if (child.toElement().text().toUpper().simplified() == "FIXED" || child.toElement().text().toUpper().simplified() == "XMLFORMATFIXED")
                    {
                        format_type = XMLFORMATFIXED;
                    }
                    else if (child.toElement().text().toUpper().simplified() == "COMMASEPARATED" || child.toElement().text().toUpper().simplified() == "DELIMITED"
                             || child.toElement().text().toUpper().simplified() == "CSV")
                    {
                        format_type = XMLFORMATDELIMITED;
                    }
                }
                else if (myTag == "ATTRIBUTE")
                {
                    if (child.toElement().text().toUpper().simplified() == "SINGLEPOINT")
                    {
                        format_isSinglePoint = true;
                    }
                    else
                    {
                        format_isSinglePoint = false;
                    }
                }
                else if (myTag == "HEADER" || myTag == "HEADERROWS" || myTag == "NUMHEADERROWS")
                {
                    format_headerRow = child.toElement().text().toInt();
                }
                else if (myTag == "MISSINGVALUE" || myTag == "MISSING_VALUE" || myTag == "NODATA")
                {
                    format_missingValue = child.toElement().text().toFloat();
                }
                else if (myTag == "DELIMITER")
                {
                    if (child.toElement().text() == "")
                    {
                        format_delimiter = " ";
                    }
                    else
                    {
                        format_delimiter = child.toElement().text();
                    }
                }
                else if (myTag == "DECIMALSEPARATOR")
                {
                    format_decimalSeparator = child.toElement().text();
                }
                child = child.nextSibling();
            }
        }
        else if (ancestor.toElement().tagName().toUpper() == "POINTCODE")
        {
            child = ancestor.firstChild();
            while( !child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "TYPE" || myTag == "NAME")
                {
                    pointCode.setType(child.toElement().text());
                }
                else if (myTag == "FORMAT")
                {
                    pointCode.setFormat(child.toElement().text());
                }
                else if (myTag == "ATTRIBUTE")
                {
                    pointCode.setAttribute(child.toElement().text());
                }
                else if (myTag == "FIELD" || myTag == "POSITION")
                {
                    pointCode.setPosition(child.toElement().text().toInt());
                }
                else if (myTag == "FIRST_CHAR" || myTag == "FIRSTCHAR")
                {
                    pointCode.setFirstChar(child.toElement().text().toInt());
                }
                else if (myTag == "NR_CHAR" || myTag == "NUMCHAR" || myTag == "NRCHAR")
                {
                    pointCode.setNrChar(child.toElement().text().toInt());
                }
                else if (myTag == "ALIGN" || myTag == "ALIGNMENT")
                {
                    pointCode.setAlignment(child.toElement().text());
                }
                else if (myTag == "PREFIX" || myTag == "FIXEDTEXT")
                {
                    pointCode.setPrefix(child.toElement().text());
                }
                child = child.nextSibling();
            }
        }
        else if (ancestor.toElement().tagName().toUpper() == "TIME")
        {
            child = ancestor.firstChild();
            while( !child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "TYPE" || myTag == "NAME")
                {
                    time.setType(child.toElement().text());
                }
                else if (myTag == "FORMAT")
                {
                    time.setFormat(child.toElement().text());
                }
                else if (myTag == "ATTRIBUTE")
                {
                    time.setAttribute(child.toElement().text());
                }
                else if (myTag == "FIELD" || myTag == "POSITION")
                {
                    time.setPosition(child.toElement().text().toInt());
                }
                else if (myTag == "FIRST_CHAR" || myTag == "FIRSTCHAR")
                {
                    time.setFirstChar(child.toElement().text().toInt());
                }
                else if (myTag == "NR_CHAR" || myTag == "NUMCHAR" || myTag == "NRCHAR")
                {
                    time.setNrChar(child.toElement().text().toInt());
                }
                else if (myTag == "ALIGN" || myTag == "ALIGNMENT")
                {
                    time.setAlignment(child.toElement().text());
                }
                else if (myTag == "PREFIX" || myTag == "FIXEDTEXT")
                {
                    time.setPrefix(child.toElement().text());
                }
                child = child.nextSibling();
            }
        }
        else if (ancestor.toElement().tagName().toUpper() == "VARIABLECODE")
        {
            child = ancestor.firstChild();
            while( !child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "TYPE" || myTag == "NAME")
                {
                    variableCode.setType(child.toElement().text());
                }
                else if (myTag == "FORMAT")
                {
                    variableCode.setFormat(child.toElement().text());
                }
                else if (myTag == "ATTRIBUTE")
                {
                    variableCode.setAttribute(child.toElement().text());
                }
                else if (myTag == "FIELD" || myTag == "POSITION")
                {
                    variableCode.setPosition(child.toElement().text().toInt());
                }
                else if (myTag == "FIRST_CHAR" || myTag == "FIRSTCHAR")
                {
                    variableCode.setFirstChar(child.toElement().text().toInt());
                }
                else if (myTag == "NR_CHAR" || myTag == "NUMCHAR" || myTag == "NRCHAR")
                {
                    variableCode.setNrChar(child.toElement().text().toInt());
                }
                else if (myTag == "ALIGN" || myTag == "ALIGNMENT")
                {
                    variableCode.setAlignment(child.toElement().text());
                }
                else if (myTag == "PREFIX" || myTag == "FIXEDTEXT")
                {
                    variableCode.setPrefix(child.toElement().text());
                }
                child = child.nextSibling();
            }
        }
        else if (ancestor.toElement().tagName().toUpper() == "VARIABLE")
        {
            VariableXML var;
            variable.push_back(var);
            child = ancestor.firstChild();
            while( !child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "FIELD")
                {
                    secondChild = child.firstChild();
                    while( !secondChild.isNull())
                    {
                        mySecondTag = secondChild.toElement().tagName().toUpper();
                        if (mySecondTag == "TYPE" || mySecondTag == "NAME")
                        {
                            variable[variable.size()-1].varField.setType(secondChild.toElement().text());
                        }
                        else if (mySecondTag == "FORMAT")
                        {
                            variable[variable.size()-1].varField.setFormat(secondChild.toElement().text());
                        }
                        else if (mySecondTag == "ATTRIBUTE")
                        {
                            variable[variable.size()-1].varField.setAttribute(secondChild.toElement().text());
                        }
                        else if (mySecondTag == "FIELD" || mySecondTag == "POSITION")
                        {
                            variable[variable.size()-1].varField.setPosition(secondChild.toElement().text().toInt());
                        }
                        else if (mySecondTag == "FIRST_CHAR" || mySecondTag == "FIRSTCHAR")
                        {
                            variable[variable.size()-1].varField.setFirstChar(secondChild.toElement().text().toInt());
                        }
                        else if (mySecondTag == "NR_CHAR" || mySecondTag == "NUMCHAR" || mySecondTag == "NRCHAR")
                        {
                            variable[variable.size()-1].varField.setNrChar(secondChild.toElement().text().toInt());
                        }
                        else if (mySecondTag == "ALIGN" || mySecondTag == "ALIGNMENT")
                        {
                            variable[variable.size()-1].varField.setAlignment(secondChild.toElement().text());
                        }
                        else if (mySecondTag == "PREFIX" || mySecondTag == "FIXEDTEXT")
                        {
                            variable[variable.size()-1].varField.setPrefix(secondChild.toElement().text());
                        }
                        secondChild = secondChild.nextSibling();
                    }
                }
                else if (myTag == "FLAG")
                {
                    secondChild = child.firstChild();
                    while( !secondChild.isNull())
                    {
                        mySecondTag = secondChild.toElement().tagName().toUpper();
                        if (mySecondTag == "FIELD")
                        {
                            thirdChild = secondChild.firstChild();
                            while( !thirdChild.isNull())
                            {
                                myThirdTag = thirdChild.toElement().tagName().toUpper();
                                if (myThirdTag == "TYPE" || myThirdTag == "NAME")
                                {
                                    variable[variable.size()-1].flagField.setType(thirdChild.toElement().text());
                                }
                                else if (myThirdTag == "FORMAT")
                                {
                                    variable[variable.size()-1].flagField.setFormat(thirdChild.toElement().text());
                                }
                                else if (myThirdTag == "ATTRIBUTE")
                                {
                                    variable[variable.size()-1].flagField.setAttribute(thirdChild.toElement().text());
                                }
                                else if (myThirdTag == "FIELD" || myThirdTag == "POSITION")
                                {
                                    variable[variable.size()-1].flagField.setPosition(thirdChild.toElement().text().toInt());
                                }
                                else if (myThirdTag == "FIRST_CHAR" || myThirdTag == "FIRSTCHAR")
                                {
                                    variable[variable.size()-1].flagField.setFirstChar(thirdChild.toElement().text().toInt());
                                }
                                else if (myThirdTag == "NR_CHAR" || myThirdTag == "NUMCHAR" || myThirdTag == "NRCHAR")
                                {
                                    variable[variable.size()-1].flagField.setNrChar(thirdChild.toElement().text().toInt());
                                }
                                else if (myThirdTag == "ALIGN" || myThirdTag == "ALIGNMENT")
                                {
                                    variable[variable.size()-1].flagField.setAlignment(thirdChild.toElement().text());
                                }
                                else if (myThirdTag == "PREFIX" || myThirdTag == "FIXEDTEXT")
                                {
                                    variable[variable.size()-1].flagField.setPrefix(thirdChild.toElement().text());
                                }
                                thirdChild = thirdChild.nextSibling();
                            }
                        }
                        else if (mySecondTag == "ACCEPTED" || mySecondTag == "VALUE")
                        {
                            variable[variable.size()-1].flagAccepted = secondChild.toElement().text();
                        }
                        secondChild = secondChild.nextSibling();
                    }
                }
                else if (myTag == "NR_REPLICATIONS" || myTag == "REPLICATION")
                {
                    variable[variable.size()-1].nReplication = child.toElement().text().toInt();
                }
                child = child.nextSibling();
            }
        }
        ancestor = ancestor.nextSibling();
    }
    xmlDoc.clear();

    if (variableCode.getType().toUpper() == "FIXED")
    {
        for (int i = 0; i<variable.size(); i++)
        {
            if (variable[i].nReplication > 1)
            {
                numVarFields = numVarFields + variable[i].nReplication;
            }
            else
            {
                numVarFields = numVarFields + 1;
            }
        }
    }
    else if (variable.size() > 0)
    {
        numVarFields = 1;
    }
    else
    {
        *myError = "Error in XML File: missing variables definition.";
        return false;
    }
    return true;
}

bool ImportDataXML::importData(QString fileName, QString *error)
{
    if (fileName == "")
    {
        *error = "Missing data file.";
        return false;
    }

    dataFileName = fileName;
    if (format_type == XMLFORMATFIXED)
    {
        return importXMLDataFixed(error);
    }
    else
    {
        return importXMLDataDelimited(error);
    }
}

bool ImportDataXML::importXMLDataFixed(QString *error)
{
    QFile myFile(dataFileName);
    QFileInfo myFileInfo(myFile.fileName());
    if (!myFile.open(QIODevice::ReadOnly))
    {
        *error = "Open file failed: " + dataFileName + "\n " + myFile.errorString();
        return false;
    }

    QString myPointCode = "";

    if (format_isSinglePoint)
    {
        if (pointCode.getType().toUpper() == "FILENAMEDEFINED")
        {
            myPointCode = parseXMLPointCode(myFileInfo.baseName());
            if (myPointCode.isEmpty())
            {
                *error = "Point code not found for file: " + dataFileName;
                return false;
            }
            // check if myPointCode exists
            if (isGrid)
            {
                if(!meteoGridDbHandler->meteoGrid()->existsMeteoPointFromId(myPointCode.toStdString()))
                {
                    *error = "Point code: " + myPointCode + "not exists for file: " + dataFileName;
                    return false;
                }
            }
            else
            {
                if (!meteoPointsDbHandler->existIdPoint(myPointCode))
                {
                    *error = "Point code: " + myPointCode + "not exists for file: " + dataFileName;
                    return false;
                }
            }
        }
        else
        {
            *error = "Open: " + dataFileName + " format is SinglePoint but pointCode is not FILENAMEDEFINED";
            return false;
        }
    }

    QTextStream in(&myFile);
    int nRow = 0;
    QVariant myFlagAccepted;
    QVariant myFlag;
    QVariant myValue;
    int nErrors = 0;
    int nReplication = 0;  // LC è sempre 0,eliminare?

    QMultiMap<QString, QList<QString>> mapIdValues;
    QList<QString> listEntries;

    while (!in.atEnd())
    {
      QString line = in.readLine();
      if (nRow >= format_headerRow && !line.isEmpty())
      {
          if (!format_isSinglePoint)
          {
              //multipoint
              QString previousPointCode = "";
              myPointCode = parseXMLPointCode(line);
              if (myPointCode.isEmpty())
              {
                  *error = "Point code not found for file: " + dataFileName;
                  return false;
              }
              if (myPointCode != previousPointCode)
              {
                  // check if myPointCode exists
                  if (isGrid)
                  {
                      if(!meteoGridDbHandler->meteoGrid()->existsMeteoPointFromId(myPointCode.toStdString()))
                      {
                          *error = "Point code: " + myPointCode + "not exists for file: " + dataFileName;
                          return false;
                      }
                  }
                  else
                  {
                      if (!meteoPointsDbHandler->existIdPoint(myPointCode))
                      {
                          *error = "Point code: " + myPointCode + "not exists for file: " + dataFileName;
                          return false;
                      }
                  }
                  previousPointCode = myPointCode;
              }
          }
        if (time.getType().toUpper() == "DAILY")
        {
            QDate myDate = parseXMLDate(line);
            if (!myDate.isValid() || myDate.year() == 1800)
            {
                *error = "Date not found or not valid for file: " + dataFileName;
                return false;
            }
            for (int i = 0; i<variable.size(); i++)
            {
                if (variable[i].nReplication > 1)
                {
                    // TO DO (anche in vb)
                }
                else
                {
                    // FLAG
                    if (!variable[i].flagAccepted.isEmpty())
                    {
                        QString format = variable[i].flagField.getFormat();
                        if (format.isEmpty() || format == "%s" || format == "%d")
                        {
                            myFlagAccepted = variable[i].flagAccepted;
                            myFlag = parseXMLFixedValue(line, nReplication, variable[i].flagField);
                        }
                    }
                    else
                    {
                        myFlagAccepted = 0;
                        myFlag = 0;
                    }
                    myValue  = parseXMLFixedValue(line, nReplication, variable[i].varField);
                    if (myValue.toString() == "ERROR")
                    {
                        nErrors = nErrors + 1;
                        myValue = format_missingValue;
                    }
                    else if (myFlag != myFlagAccepted)
                    {
                        myValue = format_missingValue;
                    }

                } // end flag if
                if (myValue != format_missingValue)
                {
                    // write myValue
                    meteoVariable var = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, variable[i].varField.getType().toStdString());
                    if (var == noMeteoVar)
                    {
                        *error = "Meteovariable not found or not valid for file:\n" + dataFileName;
                        return false;
                    }
                    if (isGrid)
                    {
                        listEntries.push_back(QString("('%1',%2,%3)").arg(myDate.toString("yyyy-MM-dd")).arg(meteoGridDbHandler->getDailyVarCode(var)).arg(myValue.toFloat()));
                    }
                    else
                    {
                        listEntries.push_back(QString("('%1',%2,%3)").arg(myDate.toString("yyyy-MM-dd")).arg(meteoPointsDbHandler->getIdfromMeteoVar(var)).arg(myValue.toFloat()));
                    }
                    mapIdValues.insert(myPointCode, listEntries);
                    // TO DO isFixedFields non è ottimizzata la scrittura, struttura non piu' utilizzata
                    if (isGrid && meteoGridDbHandler->meteoGrid()->gridStructure().isFixedFields())
                    {
                        if (!meteoGridDbHandler->saveCellCurrentGridDailyFF(error, myPointCode, myDate, QString::fromStdString(meteoGridDbHandler->getDailyPragaName(var)), myValue.toFloat()))
                        {
                            return false;
                        }
                    }
                }
            }
        } // end daily
        else if (time.getType().toUpper() == "HOURLY")
        {
            QDateTime myDateTime = parseXMLDateTime(line);
            if (!myDateTime.isValid() || myDateTime.date().year() == 1800)
            {
                *error = "Date not found or not valid for file: " + dataFileName + "\n" + line;
                return false;
            }
            for (int i = 0; i<variable.size(); i++)
            {
                if (variable[i].nReplication > 1)
                {
                    // TO DO (anche in vb)
                }
                else
                {
                    // FLAG
                    if (!variable[i].flagAccepted.isEmpty())
                    {
                        QString format = variable[i].flagField.getFormat();
                        if (format.isEmpty() || format == "%s" || format == "%d")
                        {
                            myFlagAccepted = variable[i].flagAccepted;
                            myFlag = parseXMLFixedValue(line, nReplication, variable[i].flagField);
                        }
                    }
                    else
                    {
                        myFlagAccepted = 0;
                        myFlag = 0;
                    }
                    myValue  = parseXMLFixedValue(line, nReplication, variable[i].varField);
                    if (myValue.toString() == "ERROR")
                    {
                        nErrors = nErrors + 1;
                        myValue = format_missingValue;
                    }
                    else if (myFlag != myFlagAccepted)
                    {
                        myValue = format_missingValue;
                    }

                } // end flag if
                if (myValue != format_missingValue)
                {
                    // write myValue
                    meteoVariable var = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, variable[i].varField.getType().toStdString());
                    if (var == noMeteoVar)
                    {
                        *error = "Meteovariable not found or not valid for file:\n" + dataFileName;
                        return false;
                    }
                    if (isGrid)
                    {
                        listEntries.push_back(QString("('%1',%2,%3)").arg(myDateTime.toString("yyyy-MM-dd hh:mm:ss")).arg(meteoGridDbHandler->getHourlyVarCode(var)).arg(myValue.toFloat()));
                    }
                    else
                    {
                        listEntries.push_back(QString("('%1',%2,%3)").arg(myDateTime.toString("yyyy-MM-dd hh:mm:ss")).arg(meteoPointsDbHandler->getIdfromMeteoVar(var)).arg(myValue.toFloat()));
                    }
                    mapIdValues.insert(myPointCode, listEntries);
                    // TO DO isFixedFields non è ottimizzata la scrittura, struttura non piu' utilizzata
                    if (isGrid && meteoGridDbHandler->meteoGrid()->gridStructure().isFixedFields())
                    {
                        if (!meteoGridDbHandler->saveCellCurrentGridHourlyFF(error, myPointCode, myDateTime, QString::fromStdString(meteoGridDbHandler->getHourlyPragaName(var)), myValue.toFloat()))
                        {
                            return false;
                        }
                    }
                }
            }
        }
        else
        {
            *error = "Unknown time type" + time.getType().toUpper() + "for file: " + dataFileName;
            return false;
        }
      }
      nRow = nRow + 1;
    }
    myFile.close();

    if (isGrid && !meteoGridDbHandler->meteoGrid()->gridStructure().isFixedFields())
    {
        QList<QString> keys = mapIdValues.uniqueKeys();
        if (time.getType().toUpper() == "DAILY")
        {
            for (int i = 0; i<keys.size(); i++)
            {
                if (!meteoGridDbHandler->saveCellCurrentGridDailyList(keys[i], mapIdValues.value(keys[i]), error))
                {
                    return false;
                }
            }
        }
        else if (time.getType().toUpper() == "HOURLY")
        {
            for (int i = 0; i<keys.size(); i++)
            {
                if (!meteoGridDbHandler->saveCellCurrentGridHourlyList(keys[i], mapIdValues.value(keys[i]), error))
                {
                    return false;
                }
            }
        }
    }
    else
    {
        QList<QString> keys = mapIdValues.uniqueKeys();
        if (time.getType().toUpper() == "DAILY")
        {
            for (int i = 0; i<keys.size(); i++)
            {
                if (!meteoPointsDbHandler->writeDailyDataList(keys[i], mapIdValues.value(keys[i]), error))
                {
                    return false;
                }
            }
        }
        else if (time.getType().toUpper() == "HOURLY")
        {
            for (int i = 0; i<keys.size(); i++)
            {
                if (!meteoPointsDbHandler->writeHourlyDataList(keys[i], mapIdValues.value(keys[i]), error))
                {
                    return false;
                }
            }
        }
    }

    if (nErrors != 0)
    {
        *error = QString::number(nErrors);
    }
    return true;
}

bool ImportDataXML::importXMLDataDelimited(QString *error)
{
    QFile myFile(dataFileName);
    QFileInfo myFileInfo(myFile.fileName());
    if (!myFile.open(QIODevice::ReadOnly))
    {
        *error = "Open file failed: " + dataFileName + "\n " + myFile.errorString();
        return (false);
    }

    QString myPointCode = "";

    if (format_isSinglePoint)
    {
        if (pointCode.getType().toUpper() == "FILENAMEDEFINED")
        {
            myPointCode = parseXMLPointCode(myFileInfo.baseName());
            if (myPointCode.isEmpty())
            {
                *error = "Point code not found for file: " + dataFileName;
                return false;
            }
            // check if myPointCode exists
            if (isGrid)
            {
                if(!meteoGridDbHandler->meteoGrid()->existsMeteoPointFromId(myPointCode.toStdString()))
                {
                    *error = "Point code: " + myPointCode + "not exists for file: " + dataFileName;
                    return false;
                }
            }
            else
            {
                if (!meteoPointsDbHandler->existIdPoint(myPointCode))
                {
                    *error = "Point code: " + myPointCode + "not exists for file: " + dataFileName;
                    return false;
                }
            }
        }
        else
        {
            *error = "Open: " + dataFileName + " format is SinglePoint but pointCode is not FILENAMEDEFINED";
            return false;
        }
    }

    QTextStream in(&myFile);
    int nRow = 0;
    QVariant myValue;
    int nReplication = 0;  // TO DO
    int nErrors = 0;
    QMultiMap<QString, QList<QString>> mapIdValues;
    QList<QString> listEntries;

    while (!in.atEnd())
    {
        QString line = in.readLine();
        if (nRow >= format_headerRow && !line.isEmpty())
        {
            if (!format_isSinglePoint)
            {
              QList<QString> myFields = line.split(format_delimiter);
              QString previousPointCode = "";
              if (pointCode.getPosition()-1 < myFields.size())
              {
                  myPointCode = parseXMLPointCode(line);
              }
              if (myPointCode.isEmpty())
              {
                *error = "Point code not found for file: " + dataFileName;
                return false;
              }
              if (myPointCode != previousPointCode)
              {
                // check if myPointCode exists
                if (isGrid)
                {
                    if(!meteoGridDbHandler->meteoGrid()->existsMeteoPointFromId(myPointCode.toStdString()))
                    {
                      *error = "Point code: " + myPointCode + "not exists for file: " + dataFileName;
                      return false;
                    }
                 }
                 else
                 {
                    if (!meteoPointsDbHandler->existIdPoint(myPointCode))
                    {
                       *error = "Point code: " + myPointCode + "not exists for file: " + dataFileName;
                       return false;
                    }
                 }
                 previousPointCode = myPointCode;
              }
            } // end multiPoint case
            if (time.getType().toUpper() == "DAILY")
            {
                QList<QString> myFields = line.split(format_delimiter);
                QDate myDate(1800,1,1);

                if (time.getPosition()-1 < myFields.size())
                {
                    myDate = parseXMLDate(myFields[time.getPosition()-1]);
                }

                if (!myDate.isValid() || myDate.year() == 1800)
                {
                    *error = "Date not found or not valid for file: " + dataFileName;
                    return false;
                }
                for (int i = 0; i<variable.size(); i++)
                {
                    if (variable[i].nReplication > 1)
                    {
                        // TO DO (anche in vb)
                    }
                    else
                    {
                        if (!variable[i].flagAccepted.isEmpty() && variable[i].flagField.getPosition()>0 && variable[i].flagField.getPosition()-1 < myFields.size())
                        {
                            if (variable[i].varField.getPosition() > 0 && variable[i].varField.getPosition()-1 < myFields.size())
                            {
                                myValue  = parseXMLFixedValue(myFields[variable[i].varField.getPosition()-1], nReplication, variable[i].varField);
                                if (myValue.toString() == "ERROR")
                                {
                                    nErrors = nErrors + 1;
                                    myValue = format_missingValue;
                                }
                            }
                            else
                            {
                                nErrors = nErrors + 1;
                                myValue = format_missingValue;
                                return false;
                            }
                            // check FLAG
                            if (myFields[variable[i].flagField.getPosition()-1] != variable[i].flagAccepted)
                            {
                                myValue = format_missingValue;
                            }
                        }
                        else
                        {
                            if (variable[i].varField.getPosition() > 0 && variable[i].varField.getPosition()-1 < myFields.size())
                            {
                                myValue  = parseXMLFixedValue(myFields[variable[i].varField.getPosition()-1], nReplication, variable[i].varField);
                                if (myValue.toString() == "ERROR")
                                {
                                    nErrors = nErrors + 1;
                                    myValue = format_missingValue;
                                }
                            }
                            else
                            {
                                nErrors = nErrors + 1;
                                myValue = format_missingValue;
                                return false;
                            }
                        }
                        if (myValue != format_missingValue && myValue != NODATA)
                        {
                            // write myValue
                            meteoVariable var = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, variable[i].varField.getType().toStdString());
                            if (var == noMeteoVar)
                            {
                                *error = "Meteovariable not found or not valid for file:\n" + dataFileName;
                                return false;
                            }
                            if (isGrid)
                            {
                                listEntries.push_back(QString("('%1',%2,%3)").arg(myDate.toString("yyyy-MM-dd")).arg(meteoGridDbHandler->getDailyVarCode(var)).arg(myValue.toFloat()));
                            }
                            else
                            {
                                listEntries.push_back(QString("('%1',%2,%3)").arg(myDate.toString("yyyy-MM-dd")).arg(meteoPointsDbHandler->getIdfromMeteoVar(var)).arg(myValue.toFloat()));
                            }
                            mapIdValues.insert(myPointCode, listEntries);
                            // TO DO isFixedFields non è ottimizzata la scrittura, struttura non piu' utilizzata
                            if (isGrid && meteoGridDbHandler->meteoGrid()->gridStructure().isFixedFields())
                            {
                                if (!meteoGridDbHandler->saveCellCurrentGridDailyFF(error, myPointCode, myDate, QString::fromStdString(meteoGridDbHandler->getDailyPragaName(var)), myValue.toFloat()))
                                {
                                    return false;
                                }
                            }
                        }

                    }

                }
            } // end daily
            else if (time.getType().toUpper() == "HOURLY")
            {
                QList<QString> myFields = line.split(format_delimiter);
                QDateTime myDateTime(QDate(1800,1,1), QTime(0,0,0), Qt::UTC);

                if (time.getPosition()-1 < myFields.size())
                {
                     myDateTime = parseXMLDateTime(myFields[time.getPosition()-1]);
                }

                if (!myDateTime.isValid() || myDateTime.date().year() == 1800)
                {
                    *error = "Date not found or not valid for file: " + dataFileName + "\n" + line;
                    return false;
                }
                for (int i = 0; i<variable.size(); i++)
                {
                    if (variable[i].nReplication > 1)
                    {
                        // TO DO (anche in vb)
                    }
                    else
                    {
                        if (!variable[i].flagAccepted.isEmpty() && variable[i].flagField.getPosition()>0 && variable[i].flagField.getPosition()-1 < myFields.size())
                        {
                            if (variable[i].varField.getPosition() > 0 && variable[i].varField.getPosition()-1 < myFields.size())
                            {
                                myValue  = parseXMLFixedValue(myFields[variable[i].varField.getPosition()-1], nReplication, variable[i].varField);
                                if (myValue.toString() == "ERROR")
                                {
                                    nErrors = nErrors + 1;
                                    myValue = format_missingValue;
                                }
                            }
                            else
                            {
                                nErrors = nErrors + 1;
                                myValue = format_missingValue;
                                return false;
                            }
                            // check FLAG
                            if (myFields[variable[i].flagField.getPosition()-1] != variable[i].flagAccepted)
                            {
                                myValue = format_missingValue;
                            }
                        }
                        else
                        {
                            if (variable[i].varField.getPosition() > 0 && variable[i].varField.getPosition()-1 < myFields.size())
                            {
                                myValue  = parseXMLFixedValue(myFields[variable[i].varField.getPosition()-1], nReplication, variable[i].varField);
                                if (myValue.toString() == "ERROR")
                                {
                                    nErrors = nErrors + 1;
                                    myValue = format_missingValue;
                                }
                            }
                            else
                            {
                                nErrors = nErrors + 1;
                                myValue = format_missingValue;
                                return false;
                            }
                        }
                        if (myValue != format_missingValue && myValue != NODATA)
                        {
                            // write myValue
                            meteoVariable var = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, variable[i].varField.getType().toStdString());
                            if (var == noMeteoVar)
                            {
                                *error = "Meteovariable not found or not valid for file:\n" + dataFileName;
                                return false;
                            }
                            if (isGrid)
                            {
                                listEntries.push_back(QString("('%1',%2,%3)").arg(myDateTime.toString("yyyy-MM-dd hh:mm:ss")).arg(meteoGridDbHandler->getHourlyVarCode(var)).arg(myValue.toFloat()));
                            }
                            else
                            {
                                listEntries.push_back(QString("('%1',%2,%3)").arg(myDateTime.toString("yyyy-MM-dd hh:mm:ss")).arg(meteoPointsDbHandler->getIdfromMeteoVar(var)).arg(myValue.toFloat()));
                            }
                            mapIdValues.insert(myPointCode, listEntries);
                            // TO DO isFixedFields non è ottimizzata la scrittura, struttura non piu' utilizzata
                            if (isGrid && meteoGridDbHandler->meteoGrid()->gridStructure().isFixedFields())
                            {
                                if (!meteoGridDbHandler->saveCellCurrentGridHourlyFF(error, myPointCode, myDateTime, QString::fromStdString(meteoGridDbHandler->getDailyPragaName(var)), myValue.toFloat()))
                                {
                                    return false;
                                }
                            }
                        }

                    }
                }
            } // end hourly
            else
            {
                *error = "Unknown time type" + time.getType().toUpper() + "for file: " + dataFileName;
                return false;
            }
      }
      nRow = nRow + 1;
    }
    myFile.close();

    if (isGrid && !meteoGridDbHandler->meteoGrid()->gridStructure().isFixedFields())
    {
        QList<QString> keys = mapIdValues.uniqueKeys();
        if (time.getType().toUpper() == "DAILY")
        {
            for (int i = 0; i<keys.size(); i++)
            {
                if (!meteoGridDbHandler->saveCellCurrentGridDailyList(keys[i], mapIdValues.value(keys[i]), error))
                {
                    return false;
                }
            }
        }
        else if (time.getType().toUpper() == "HOURLY")
        {
            for (int i = 0; i<keys.size(); i++)
            {
                if (!meteoGridDbHandler->saveCellCurrentGridHourlyList(keys[i], mapIdValues.value(keys[i]), error))
                {
                    return false;
                }
            }
        }
    }
    else
    {
        QList<QString> keys = mapIdValues.uniqueKeys();
        if (time.getType().toUpper() == "DAILY")
        {
            for (int i = 0; i<keys.size(); i++)
            {
                if (!meteoPointsDbHandler->writeDailyDataList(keys[i], mapIdValues.value(keys[i]), error))
                {
                    return false;
                }
            }
        }
        else if (time.getType().toUpper() == "HOURLY")
        {
            for (int i = 0; i<keys.size(); i++)
            {
                if (!meteoPointsDbHandler->writeHourlyDataList(keys[i], mapIdValues.value(keys[i]), error))
                {
                    return false;
                }
            }
        }
    }

    if (nErrors != 0)
    {
        *error = "Not valid or missing data: " + QString::number(nErrors);
    }
    return true;
}

QString ImportDataXML::parseXMLPointCode(QString text)
{
    QString myPointCode = "";

    if (pointCode.getType().toUpper() == "FIELDDEFINED" || pointCode.getType().toUpper() == "FIELDEFINED" || pointCode.getType().toUpper() == "FILENAMEDEFINED")
    {
        if (format_type == XMLFORMATFIXED || (format_type == XMLFORMATDELIMITED && pointCode.getType().toUpper() == "FILENAMEDEFINED"))
        {
            QString substring = text.mid(pointCode.getFirstChar()-1,pointCode.getNrChar());
            if (pointCode.getFormat().isEmpty() || pointCode.getFormat() == "%s")
            {
                // pointCode is a string
                myPointCode = substring;
            }
        }
        else if (format_type == XMLFORMATDELIMITED)
        {
            QList<QString> list = text.split(format_delimiter);
            if (format_isSinglePoint)
            {
                myPointCode = list[pointCode.getPosition()];
            }
            else
            {
                myPointCode = list[pointCode.getPosition()-1];
            }
        }
    }

    if (myPointCode != "")
    {
        myPointCode = myPointCode.trimmed();
    }

    return myPointCode;
}

QDate ImportDataXML::parseXMLDate(QString text)
{
    QDate myDate(1800,1,1);

    QString myDateStr = text.mid(time.getFirstChar()-1,time.getNrChar());
    QString format = time.getFormat();
    myDate = QDate::fromString(myDateStr,format);

    return myDate;
}


QDateTime ImportDataXML::parseXMLDateTime(QString text)
{
    QString myDateStr = text.mid(time.getFirstChar()-1,time.getNrChar());
    QString format = time.getFormat();

    QDateTime myDateTime = QDateTime::fromString(myDateStr,format);
    myDateTime.setTimeSpec(Qt::UTC);

    return myDateTime;
}


QVariant ImportDataXML::parseXMLFixedValue(QString text, int nReplication, FieldXML myField)
{
    QVariant myValue = "ERROR";
    QString mySubstring;
    if (myField.getNrChar() == 0 || myField.getNrChar() == NODATA)
    {
        mySubstring = text;
    }
    else
    {
        mySubstring = text.mid(myField.getFirstChar()-1 + (nReplication*myField.getNrChar()),myField.getNrChar());
    }
    if (!mySubstring.isEmpty())
    {
        QString format = myField.getFormat();
        if (format.isEmpty() || format == "%s")
        {
            // is a string
            myValue = mySubstring;
        }
        else if (format == "%d")
        {
            // is a int
            mySubstring = mySubstring.trimmed();
            myValue = mySubstring.toInt();
        }
        else if (format.endsWith("f"))
        {
            // is a float
            bool ok;
            mySubstring = mySubstring.trimmed();
            float myFloat = mySubstring.toFloat(&ok);
            if(ok)
            {
                if (format.contains("."))
                {
                    // decimals
                    int startPos = format.indexOf('.')+1;
                    int endPos = format.indexOf('f');
                    int nChar = endPos - startPos;
                    QString nDecimasStr = format.mid(startPos, nChar);
                    int nDecimals = nDecimasStr.toInt();
                    QString strFormat = QString("%1").arg(myFloat,0,'f',nDecimals);
                    myFloat = strFormat.toFloat();
                }
                myValue = myFloat;
            }
        }
    }
    return myValue;
}
