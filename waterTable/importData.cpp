#include "importData.h"
#include "commonConstants.h"
#include <QFile>
#include <QTextStream>
#include <QRegularExpression>
#include "well.h"
#include "gis.h"


bool loadWaterTableLocationCsv(const QString &csvFileName, std::vector<Well> &wellList,
                               const gis::Crit3DGisSettings& gisSettings, QString &errorStr, int &wrongLines)
{
    wellList.clear();

    QFile myFile(csvFileName);
    if (! myFile.open(QFile::ReadOnly | QFile::Text) )
    {
        errorStr = "Csv file does not exist:\n" + csvFileName;
        return false;
    }

    QTextStream in(&myFile);
    int nrRequiredFields = 3;

    // check header
    QString line = in.readLine();
    QList<QString> headerItems = line.split(",");
    if (headerItems.size() != nrRequiredFields)
    {
        errorStr = "Wrong data! Required well ID, utm X, utm Y.";
        return false;
    }
    errorStr = "";

    QList<QString> idList;
    QList<QString> errorList;
    int posId = 0;
    int posUtmx = 1;
    int posUtmy = 2;
    int validLines = 0;

    while (! in.atEnd())
    {
        line = in.readLine();
        QList<QString> items = line.split(",");
        items.removeAll({});
        if (items.size() < nrRequiredFields)
        {
            errorList.append(items[posId]);
            wrongLines++;
            continue;
        }

        items[posId] = items[posId].simplified();
        QString id = items[posId].remove(QChar('"'));
        if (idList.contains(id))
        {
            // id already saved
            errorList.append(id);
            wrongLines++;
            continue;
        }
        idList.append(id);

        bool isOk;
        items[posUtmx] = items[posUtmx].simplified();
        double utmX = items[posUtmx].remove(QChar('"')).toDouble(&isOk);
        if (! isOk)
        {
            errorList.append(id);
            wrongLines++;
            continue;
        }

        items[posUtmy] = items[posUtmy].simplified();
        double utmY = items[posUtmy].remove(QChar('"')).toDouble(&isOk);
        if (! isOk)
        {
            errorList.append(id);
            wrongLines++;
            continue;
        }

        double lat, lon;
        gis::getLatLonFromUtm(gisSettings, utmX, utmY, &lat, &lon);

        Well newWell;
        newWell.setId(id);
        newWell.setUtmX(utmX);
        newWell.setUtmY(utmY);
        newWell.setLatitude(lat);
        newWell.setLongitude(lon);
        wellList.push_back(newWell);

        validLines++;
    }

    myFile.close();

    if (validLines == 0)
    {
        errorStr = "Wrong wells location:\n" + csvFileName;
        return false;
    }

    if (wrongLines > 0)
    {
        errorStr = "ID repeated or with invalid coordinates: " + errorList.join(",");
    }

    return true;
}


bool loadWaterTableDepthCsv(const QString &csvFileName, std::vector<Well> &wellList,
                            int waterTableMaximumDepth, QString &errorStr, int &wrongLines)
{
    errorStr = "";
    QFile myFile(csvFileName);
    QList<QString> errorList;

    int posId = 0;
    int posDate = 1;
    int posDepth = 2;

    int nrRequiredFields = 3;
    int validLines = 0;
    bool ok;

    if (! myFile.open(QFile::ReadOnly | QFile::Text) )
    {
        errorStr = "Csv file does not exist:\n" + csvFileName;
        return false;
    }
    else
    {
        QTextStream in(&myFile);

        // check header
        QString line = in.readLine();
        QList<QString> headerItems = line.split(",");
        if (headerItems.size() != nrRequiredFields)
        {
            errorStr = "Wrong data! Required well ID, date, depth.";
            return false;
        }

        while (!in.atEnd())
        {
            line = in.readLine();
            QList<QString> items = line.split(",");
            items.removeAll({});
            if (items.size() < nrRequiredFields)
            {
                errorList.append(line);
                wrongLines++;
                continue;
            }
            items[posId] = items[posId].simplified();
            QString id = items[posId].remove(QChar('"'));
            bool found = false;
            int index = NODATA;
            for (int i = 0; i < wellList.size(); i++)
            {
                if (wellList[i].getId() == id)
                {
                    found = true;
                    index = i;
                    break;
                }
            }
            if (found == false)
            {
                // id does not exist
                errorList.append(line);
                wrongLines++;
                continue;
            }
            items[posDate] = items[posDate].simplified();
            QDate date = QDate::fromString(items[posDate].remove(QChar('"')),"yyyy-MM-dd");
            if (! date.isValid())
            {
                errorList.append(line);
                wrongLines++;
                continue;
            }

            items[posDepth] = items[posDepth].simplified();
            float value = items[posDepth].remove(QChar('"')).toFloat(&ok);
            if (!ok || value == NODATA || value < 0 || value > waterTableMaximumDepth)
            {
                errorList.append(line);
                wrongLines++;
                continue;
            }

            wellList[index].insertData(date, value);
            validLines++;
        }
    }
    myFile.close();

    if (validLines == 0)
    {
        errorStr = "Wrong water table depth:\n" + csvFileName;
        return false;
    }

    if (wrongLines > 0)
    {
        errorStr = "ID not existing or with invalid data or value:\n" + errorList.join("\n");
    }

    return true;
}


bool loadCsvDepthsSingleWell(QString csvDepths, Well* well, int waterTableMaximumDepth, QString &errorStr, int &wrongLines)
{
    QFile myFile(csvDepths);
    QList<QString> errorList;

    int posDate = 0;
    int posDepth = 1;

    int nFields = 2;
    bool ok;
    errorStr = "";

    if (! myFile.open(QFile::ReadOnly | QFile::Text) )
    {
        errorStr = "csvFileName file does not exist";
        return false;
    }
    else
    {
        QTextStream in(&myFile);
        //skip header
        QString line = in.readLine();
        while (!in.atEnd())
        {
            line = in.readLine();
            QList<QString> items = line.split(",");
            items.removeAll({});
            if (items.size() < nFields)
            {
                errorList.append(line);
                wrongLines++;
                continue;
            }
            items[posDate] = items[posDate].simplified();
            QDate date = QDate::fromString(items[posDate].remove(QChar('"')),"yyyy-MM-dd");
            if (! date.isValid())
            {
                errorList.append(line);
                wrongLines++;
                continue;
            }
            items[posDepth] = items[posDepth].simplified();
            int value = items[posDepth].remove(QChar('"')).toInt(&ok);
            if (!ok || value == NODATA || value < 0 || value > waterTableMaximumDepth)
            {
                errorList.append(line);
                wrongLines++;
                continue;
            }

            well->insertData(date, value);
        }
    }
    myFile.close();

    if (wrongLines > 0)
    {
        errorStr = "Invalid data or value or data out of range:\n" + errorList.join("\n");
    }

    return true;
}

