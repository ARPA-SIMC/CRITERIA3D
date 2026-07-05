#include "commonConstants.h"
#include "criteriaOutputVariable.h"

#include <QFile>
#include <QTextStream>


bool CriteriaOutputVariable::parserOutputVariable(const QString &fileName, QString &errorStr)
{
    QFile fileCsv(fileName);
    if ( !fileCsv.open(QFile::ReadOnly | QFile::Text) ) {
        errorStr = "File not exists";
        return false;
    }
    else
    {
        QTextStream in(&fileCsv);
        //skip header
        QString line = in.readLine();
        QList<QString> header = line.split(",");
        // whitespace removed from the start and the end.
        QMutableListIterator<QString> it(header);
        while (it.hasNext()) {
            it.next();
            it.value() = it.value().trimmed();
        }

        while (!in.atEnd())
        {
            line = in.readLine();
            QList<QString> items = line.split(",");
            if (items.size() < CSVREQUIREDINFO)
            {
                errorStr = "invalid output format CSV, missing reference data";
                return false;
            }

            int pos = int(header.indexOf("output var name"));
            if (pos == -1)
            {
                errorStr = "missing output var name";
                return false;
            }
            outputVarNameList.push_back(items[pos]);

            pos = int(header.indexOf("var name"));
            if (pos == -1)
            {
                errorStr = "missing var name";
                return false;
            }
            varNameList.push_back(items[pos].toUpper());

            pos = int(header.indexOf("reference day"));
            if (pos == -1)
            {
                errorStr = "missing reference day";
                return false;
            }
            bool ok;
            referenceDay.push_back(items[pos].toInt(&ok, 10));
            if (!ok)
            {
                return false;
            }

            pos = int(header.indexOf("computation"));
            if (pos == -1)
            {
                errorStr = "missing computation";
                return false;
            }
            computationList.push_back(items[pos]);

            pos = int(header.indexOf("nr days"));
            if (pos == -1)
            {
                errorStr = "missing nr days";
                return false;
            }
            nrDays.push_back(items[pos]);

            pos = int(header.indexOf("climate computation"));
            if (pos == -1)
            {
                errorStr = "missing climate computation";
                return false;
            }
            climateComputation.push_back(items[pos]);

            pos = int(header.indexOf("parameter 1"));
            if (pos == -1)
            {
                errorStr = "missing parameter 1";
                return false;
            }
            if (items[pos].isEmpty())
            {
                param1.push_back(NODATA);
            }
            else
            {
                param1.push_back(items[pos].toInt(&ok, 10));
                if (!ok)
                {
                    return false;
                }
            }

            pos = int(header.indexOf("parameter 2"));
            if (pos == -1)
            {
                errorStr = "missing parameter 2";
                return false;
            }
            if (items[pos].isEmpty())
            {
                param2.push_back(NODATA);
            }
            else
            {
                param2.push_back(items[pos].toInt(&ok, 10));
                if (!ok)
                {
                    return false;
                }
            }
        }
    }
    return true;
}
