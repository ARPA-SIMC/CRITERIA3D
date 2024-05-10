#include "importData.h"
#include <QFile>
#include <QTextStream>

bool loadCsvRegistry(QString csvRegistry, QList<Well> *wellList, QString *errorStr)
{
    QFile myFile(csvRegistry);
    QList<QString> idList;
    QList<QString> errorList;
    int posId = 0;
    int posUtmx = 1;
    int posUtmy = 2;

    int nFields = 3;
    bool ok;

    if ( !myFile.open(QFile::ReadOnly | QFile::Text) )
    {
        *errorStr = "csvFileName file does not exist";
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
                *errorStr = "missing field required";
                return false;
            }
            QString id = items[posId];
            if (idList.contains(id))
            {
                // id already saved
                errorList.append(id);
                continue;
            }
            idList.append(id);
            double utmX = items[posUtmx].toDouble(&ok);
            if (!ok)
            {
                errorList.append(id);
                continue;
            }
            double utmY = items[posUtmy].toDouble(&ok);
            if (!ok)
            {
                errorList.append(id);
                continue;
            }
            Well newWell;
            newWell.setId(id);
            newWell.setUtmX(utmX);
            newWell.setUtmY(utmY);
            wellList->append(newWell);
        }
    }
    myFile.close();

    if (!errorList.isEmpty())
    {
        *errorStr = "ID repeated or with invalid coordinates: " + errorList.join(",");
    }
    return true;
}

