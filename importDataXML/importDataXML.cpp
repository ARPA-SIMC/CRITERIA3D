#include "importDataXML.h"

#include <QFile>

ImportDataXML::ImportDataXML(bool isGrid, QString xmlFileName)
{
    this->isGrid = isGrid;
    this->xmlFileName = xmlFileName;
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
        *error = "Open XML failed:\n" + xmlFileName + "\n" + myFile.errorString();
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
                    if (child.toElement().text().toUpper().simplified() == "FIXED")
                    {
                        format_isFixed = true;
                    }
                    else
                    {
                        format_isFixed = false;
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
                    pointCode.setField(child.toElement().text());
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
                    time.setField(child.toElement().text());
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
                    variableCode.setField(child.toElement().text());
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
                            variable[variable.size()-1].varField.setField(secondChild.toElement().text());
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
                                    variable[variable.size()-1].flagField.setField(thirdChild.toElement().text());
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
                            if (secondChild.toElement().text().toInt() == 0)
                                variable[variable.size()-1].fieldAccepted == false;
                            else
                                variable[variable.size()-1].fieldAccepted == true;
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


}
