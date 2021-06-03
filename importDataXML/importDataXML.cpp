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

    QDomNode ancestor = xmlDoc.documentElement().firstChild();
    QString myTag;
    QString mySecondTag;

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
                        isSinglePoint = true;
                    }
                    else
                    {
                        isSinglePoint = false;
                    }
                }
                else if (myTag == "HEADER" || myTag == "HEADERROWS" || myTag == "NUMHEADERROWS")
                {
                    headerRow = child.toElement().text().toInt();
                }
                else if (myTag == "MISSINGVALUE" || myTag == "MISSING_VALUE" || myTag == "NODATA")
                {
                    missingValue = child.toElement().text().toFloat();
                }
                else if (myTag == "DELIMITER")
                {
                    if (child.toElement().text() == "")
                    {
                        delimiter = " ";
                    }
                    else
                    {
                        delimiter = child.toElement().text();
                    }
                }
                else if (myTag == "DECIMALSEPARATOR")
                {
                    decimalSeparator = child.toElement().text();
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
        ancestor = ancestor.nextSibling();
    }
    xmlDoc.clear();


}
