#include "importData.h"
#include "commonConstants.h"
#include <QFile>
#include <QTextStream>


bool loadCsvRegistry(QString csvRegistry, std::vector<Well> &wellList, QString &errorStr, int &wrongLines)
{
    errorStr = "";
    wellList.clear();

    QFile myFile(csvRegistry);
    QList<QString> idList;
    QList<QString> errorList;
    int posId = 0;
    int posUtmx = 1;
    int posUtmy = 2;
    int nFields = 3;
    bool ok;

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
            QStringList items = line.split(",");
            items.removeAll({});
            if (items.size()<nFields)
            {
                errorList.append(items[posId]);
                wrongLines++;
                continue;
            }
            QString id = items[posId];
            if (idList.contains(id))
            {
                // id already saved
                errorList.append(id);
                wrongLines++;
                continue;
            }
            idList.append(id);
            double utmX = items[posUtmx].toDouble(&ok);
            if (!ok)
            {
                errorList.append(id);
                wrongLines++;
                continue;
            }
            double utmY = items[posUtmy].toDouble(&ok);
            if (!ok)
            {
                errorList.append(id);
                wrongLines++;
                continue;
            }
            Well newWell;
            newWell.setId(id);
            newWell.setUtmX(utmX);
            newWell.setUtmY(utmY);
            wellList.push_back(newWell);
        }
    }
    myFile.close();

    if (wrongLines > 0)
    {
        errorStr = "ID repeated or with invalid coordinates: " + errorList.join(",");
    }

    return true;
}


bool loadCsvDepths(QString csvDepths, std::vector<Well> &wellList, int waterTableMaximumDepth, QString &errorStr, int &wrongLines)
{
    QFile myFile(csvDepths);
    QList<QString> errorList;
    int posId = 0;
    int posDate = 1;
    int posDepth = 2;

    int nFields = 3;
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
            QStringList items = line.split(",");
            items.removeAll({});
            if (items.size() < nFields)
            {
                errorList.append(line);
                wrongLines++;
                continue;
            }
            QString id = items[posId];
            bool found = false;
            int index = NODATA;
            for (int i = 0; i < wellList.size(); i++)
            {
                if (wellList[i].getId() == id)
                {
                    found = true;
                    index = 1;
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

            QDate date = QDate::fromString(items[posDate],"yyyy-MM-dd");
            if (! date.isValid())
            {
                errorList.append(line);
                wrongLines++;
                continue;
            }

            int value = items[posDepth].toInt(&ok);
            if (!ok || value == NODATA || value < 0 || value > waterTableMaximumDepth)
            {
                errorList.append(line);
                wrongLines++;
                continue;
            }

            wellList[index].insertData(date, value);
        }
    }
    myFile.close();

    if (wrongLines > 0)
    {
        errorStr = "ID not existing or with invalid data or value:\n" + errorList.join("\n");
    }

    return true;
}

