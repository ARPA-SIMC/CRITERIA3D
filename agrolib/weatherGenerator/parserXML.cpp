#include <QDebug>
#include <QFile>
#include <QString>

#include "commonConstants.h"
#include "parserXML.h"


XMLSeasonalAnomaly::XMLSeasonalAnomaly()
{
    this->initialize();
}


void XMLSeasonalAnomaly::initialize()
{
   point.name = "";
   point.code = "";
   point.latitude = NODATA;
   point.longitude = NODATA;
   point.info = "";

   forecast.clear();

   climatePeriod.yearFrom = NODATA;
   climatePeriod.yearTo = NODATA;

   modelNumber = NODATA;

   modelName.clear();
   modelMember.clear();

   repetitions = NODATA;
   anomalyYear = NODATA;
   anomalySeason = "";
}


void XMLSeasonalAnomaly::printInfo()
{
   qDebug() << "point.name = " << point.name;
   qDebug() << "point.longitude = " << point.longitude;
   qDebug() << "point.latitude = " << point.latitude;
   qDebug() << "climate first year = " << climatePeriod.yearFrom;
   qDebug() << "climate last year = " << climatePeriod.yearTo;
   qDebug() << "number of models = " << modelNumber;
   qDebug() << "models = " << modelName;
   qDebug() << "number of members = " << modelMember;
   qDebug() << "number of repetitions = " << repetitions;
   qDebug() << "anomaly year = " << anomalyYear;
   qDebug() << "anomaly season = " << anomalySeason;
   qDebug() << "";
}


bool parseXMLFile(QString xmlFileName, QDomDocument* xmlDoc)
{
    if (xmlFileName == "")
    {
        qDebug() << "Missing XML file.";
        return(false);
    }

    QFile myFile(xmlFileName);
    if (!myFile.open(QIODevice::ReadOnly))
    {
        qDebug() << "Open XML failed:" << xmlFileName;
        qDebug() << myFile.errorString();
        return (false);
    }

    QString myError;
    int myErrLine, myErrColumn;
    if (!xmlDoc->setContent(&myFile, &myError, &myErrLine, &myErrColumn))
    {
       qDebug() << "Parse xml failed:" << xmlFileName
                << " Row: " << QString::number(myErrLine)
                << " - Column: " << QString::number(myErrColumn)
                << "\n" << myError;
        myFile.close();
        return(false);
    }

    myFile.close();
    return true;
}


bool parseXMLSeasonal(QString xmlFileName, XMLSeasonalAnomaly* XMLAnomaly)
{
    QDomDocument xmlDoc;

    XMLAnomaly->initialize();

     if (!parseXMLFile(xmlFileName, &xmlDoc))
    {
        qDebug() << "parseXMLSeasonal error";
        return false;
    }

    QDomNode child;
    QDomNode secondChild;
    TXMLValuesList valuelist;

    QDomNode ancestor = xmlDoc.documentElement().firstChild();
    QString myTag;
    QString mySecondTag;
    int nrTokens = 0;
    const int nrRequiredToken = 31;

    QString models;
    QString members;
    QString values;

    while(!ancestor.isNull())
    {
        if (ancestor.toElement().tagName().toUpper() == "POINT")
        {
            child = ancestor.firstChild();
            while( !child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "NAME")
                {
                    XMLAnomaly->point.name = child.toElement().text();
                    // remove white spaces
                    XMLAnomaly->point.name = XMLAnomaly->point.name.simplified();
                    nrTokens++;
                }
                else if (myTag == "CODE")
                {
                    XMLAnomaly->point.code = child.toElement().text();
                    // remove white spaces
                    XMLAnomaly->point.code = XMLAnomaly->point.code.simplified();
                    nrTokens++;
                }
                else if ((myTag == "LAT") || (myTag == "LATITUDE"))
                {
                    XMLAnomaly->point.latitude = child.toElement().text().toFloat();
                    nrTokens++;
                }
                else if ((myTag == "LON") || (myTag == "LONGITUDE"))
                {
                    XMLAnomaly->point.longitude = child.toElement().text().toFloat();
                    nrTokens++;
                }
                else if (myTag == "INFO")
                {
                    XMLAnomaly->point.info = child.toElement().text();
                    // remove white spaces
                    XMLAnomaly->point.info = XMLAnomaly->point.info.simplified();
                    nrTokens++;
                }
                child = child.nextSibling();
            }

        }
        else if (ancestor.toElement().tagName().toUpper() == "CLIMATE")
        {
            child = ancestor.firstChild();
            while( !child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "FROM")
                     {XMLAnomaly->climatePeriod.yearFrom = child.toElement().text().toInt(); nrTokens++;}
                if (myTag == "TO")
                     {XMLAnomaly->climatePeriod.yearTo = child.toElement().text().toInt(); nrTokens++;}
                child = child.nextSibling();
            }

        }
        else if (ancestor.toElement().tagName().toUpper() == "MODELS")
        {
            child = ancestor.firstChild();
            while( !child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "NUMBER")
                     {XMLAnomaly->modelNumber = child.toElement().text().toInt(); nrTokens++;}
                else if (myTag == "NAME")
                {
                    models = child.toElement().text();
                    XMLAnomaly->modelName = models.split(",");
                    nrTokens++;
                }
                else if (myTag == "MEMBERS")
                {
                    members = child.toElement().text();
                    XMLAnomaly->modelMember = members.split(",");
                    nrTokens++;
                }
                else if (myTag == "REPETITIONS")
                     {XMLAnomaly->repetitions = child.toElement().text().toInt(); nrTokens++;}
                else if (myTag == "YEAR")
                     {XMLAnomaly->anomalyYear = child.toElement().text().toInt(); nrTokens++;}
                else if (myTag == "SEASON")
                     {XMLAnomaly->anomalySeason = child.toElement().text(); nrTokens++;}
                child = child.nextSibling();
            }

        }
        else if (ancestor.toElement().tagName().toUpper() == "FORECAST")
        {
            child = ancestor.firstChild();
            while( !child.isNull())
            {
                myTag = child.toElement().tagName().toUpper();
                if (myTag == "VAR")
                {
                    secondChild = child.firstChild();
                    XMLAnomaly->forecast.push_back(valuelist);
                    while( !secondChild.isNull())
                    {
                        mySecondTag = secondChild.toElement().tagName().toUpper();
                        if (mySecondTag == "TYPE")
                        {
                            XMLAnomaly->forecast[XMLAnomaly->forecast.size()-1].type = secondChild.toElement().text();
                            // remove white spaces
                            XMLAnomaly->forecast[XMLAnomaly->forecast.size()-1].type = XMLAnomaly->forecast[XMLAnomaly->forecast.size()-1].type.simplified();
                            nrTokens++;
                        }

                        if (mySecondTag == "ATTRIBUTE")
                        {
                            XMLAnomaly->forecast[XMLAnomaly->forecast.size()-1].attribute = secondChild.toElement().text();
                            // remove white spaces
                            XMLAnomaly->forecast[XMLAnomaly->forecast.size()-1].attribute = XMLAnomaly->forecast[XMLAnomaly->forecast.size()-1].attribute.simplified();
                            nrTokens++;
                        }

                        if (mySecondTag == "VALUE")
                        {
                            values = secondChild.toElement().text();
                            XMLAnomaly->forecast[XMLAnomaly->forecast.size()-1].value = values.split(",");
                            nrTokens++;
                        }

                        secondChild = secondChild.nextSibling();
                    }
                }

                child = child.nextSibling();
            }
        }

        ancestor = ancestor.nextSibling();
    }
    xmlDoc.clear();

    if (nrTokens < nrRequiredToken)
    {
        int missingTokens = nrRequiredToken - nrTokens;
        qDebug() << "Missing " + QString::number(missingTokens) + " key informations.";
        return false;
    }

    return true;
}

