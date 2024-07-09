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
    bool isLatLon = false;
    QString line = in.readLine();
    QList<QString> headerItems = line.split(",");
    if (headerItems.size() != nrRequiredFields)
    {
        errorStr = "Wrong data! Required ID, utmX, utmY or ID, lat, lon.";
        return false;
    }
    if (headerItems[1].toUpper() == "LAT")
    {
        isLatLon = true;
    }

    errorStr = "";
    QList<QString> idList;
    QList<QString> errorList;
    int validLines = 0;

    while (! in.atEnd())
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

        items[0] = items[0].simplified();
        QString id = items[0].remove(QChar('"'));
        if (idList.contains(id))
        {
            // id already saved
            errorList.append(line + "(REPEATED)");
            wrongLines++;
            continue;
        }
        idList.append(id);

        bool isOk;
        items[1] = items[1].simplified();
        double value1 = items[1].remove(QChar('"')).toDouble(&isOk);
        if (! isOk)
        {
            errorList.append(line);
            wrongLines++;
            continue;
        }

        items[2] = items[2].simplified();
        double value2 = items[2].remove(QChar('"')).toDouble(&isOk);
        if (! isOk)
        {
            errorList.append(line);
            wrongLines++;
            continue;
        }

        double utmX, utmY, lat, lon;
        if (isLatLon)
        {
            lat = value1;
            lon = value2;
            gis::getUtmFromLatLon(gisSettings, lat, lon, &utmX, &utmY);
        }
        else
        {
            utmX = value1;
            utmY = value2;
            gis::getLatLonFromUtm(gisSettings, utmX, utmY, &lat, &lon);
        }

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
        errorStr = "ID repeated or with invalid coordinates:\n" + errorList.join("\n");
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
            for (int i = 0; i < int(wellList.size()); i++)
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


bool loadCsvDepthsSingleWell(const QString &csvFileName, Well* well, int waterTableMaximumDepth, QString &errorStr, int &wrongLines)
{
    QFile myFile(csvFileName);
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

