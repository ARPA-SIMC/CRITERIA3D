#include "outputPoints.h"
#include "commonConstants.h"

#include <QFile>
#include <QTextStream>
#include <QFileInfo>



bool loadOutputPointListCsv(QString csvFileName, std::vector<gis::Crit3DOutputPoint> &outputPointList,
                         int utmZone, QString &errorString)
{
    QList<QList<QString>> data;
    if (!importOutputPointsCsv(csvFileName, data, errorString))
        return false;

    for (int i = 0; i < data.size(); i++)
    {
        gis::Crit3DOutputPoint p;
        QString id = data.at(i)[0];
        QString lat = data.at(i)[1];
        QString lon = data.at(i)[2];
        QString z = data.at(i)[3];
        QString activeStr = data.at(i)[4];
        bool active = (activeStr.trimmed() == "1");

        p.initialize(id.toStdString(), active, lat.toDouble(), lon.toDouble(), z.toDouble(), utmZone);
        outputPointList.push_back(p);
    }

    return true;
}


bool writeOutputPointListCsv(QString csvFileName, std::vector<gis::Crit3DOutputPoint> &outputPointList, QString &errorString)
{
    errorString.clear();
    if (csvFileName == "")
    {
        errorString = "Missing csv filename";
        return false;
    }

    QList<QList<QString>> data;
    for (unsigned int i = 0; i < outputPointList.size(); i++)
    {
        QList<QString> pointData;
        pointData.clear();
        pointData.append(QString::fromStdString(outputPointList[i].id));
        pointData.append(QString::number(outputPointList[i].latitude, 'g', 8));
        pointData.append(QString::number(outputPointList[i].longitude, 'g', 8));
        pointData.append(QString::number(outputPointList[i].z));
        if (outputPointList[i].active)
        {
            pointData.append("1");
        }
        else
        {
            pointData.append("0");
        }
        data.append(pointData);
    }

    QFile myFile(csvFileName);
    if (!myFile.open(QIODevice::WriteOnly | QFile::Truncate))
    {
        errorString = "Open CSV failed: " + csvFileName + "\n " + myFile.errorString();
        return false;
    }

    QTextStream myStream (&myFile);

    QString header = "id,latitude,longitude,height,active";
    myStream << header << "\n";

    for (int i = 0; i < data.size(); i++)
    {
        myStream << data[i].join(",") << "\n";
    }

    myFile.close();

    return true;
}


bool importOutputPointsCsv(QString csvFileName, QList<QList<QString>> &data, QString &errorString)
{
    errorString.clear();
    if (csvFileName == "")
    {
        errorString = "Missing CSV file.";
        return false;
    }

    QFile myFile(csvFileName);
    if (!myFile.open(QIODevice::ReadOnly))
    {
        errorString = "Open CSV failed: " + csvFileName + "\n " + myFile.errorString();
        return (false);
    }

    QTextStream myStream (&myFile);
    QList<QString> line;

    int CSVRequiredInfo = 5;
    if (myStream.atEnd())
    {
        errorString += "\nFile is void.";
        myFile.close();
        return false;
    }

    QList<QString> header = myStream.readLine().split(',');
    if (header[0].trimmed() != "id" || header[1].trimmed() != "latitude" || header[2].trimmed() != "longitude" || header[3].trimmed() != "height" || header[4].trimmed() != "active")
    {
        errorString += "invalid CSV header.";
        myFile.close();
        return false;
    }

    int nrLine = 0;
    while(!myStream.atEnd())
    {
        nrLine++;
        line = myStream.readLine().split(',');
        if (line.size() < CSVRequiredInfo)
        {
            errorString += "invalid format CSV, missing data.";
            myFile.close();
            return false;
        }

        // check id
        if (line[0].isEmpty())
        {
            errorString += "id field is empty, \nLine nr: " + QString::number(nrLine);
            myFile.close();
            return false;
        }

        // check lat
        bool isOk = false;
        double lat = line[1].toDouble(&isOk);
        if (!isOk || abs(lat) > 90.)
        {
            errorString += "invalid latitude. \nLine nr: " + QString::number(nrLine);
            myFile.close();
            return false;
        }

        // check lon
        isOk = false;
        double lon = line[2].toDouble(&isOk);
        if (!isOk || abs(lon) > 180.)
        {
            errorString += "invalid longitude. \nLine nr: " + QString::number(nrLine);
            myFile.close();
            return false;
        }

        // check height
        isOk = false;
        line[3].toDouble(&isOk);
        if (!isOk)
        {
            errorString += "invalid height. \nLine nr: " + QString::number(nrLine);
            myFile.close();
            return false;
        }

        // check active
        isOk = false;
        int active = line[4].toInt(&isOk);
        if (!isOk || (active != 0 && active != 1))
        {
            errorString += "invalid value in field active. \nLine nr: " + QString::number(nrLine);
            myFile.close();
            return false;
        }
        data.append(line);
    }
    myFile.close();

    return true;
}
