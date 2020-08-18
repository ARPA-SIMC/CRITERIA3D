#include "commonConstants.h"
#include "criteriaAggregationVariable.h"
#include <QFile>
#include <QTextStream>


CriteriaAggregationVariable::CriteriaAggregationVariable()
{
}

bool CriteriaAggregationVariable::parserAggregationVariable(QString fileName, QString &error)
{
    QFile fileCsv(fileName);
    if ( !fileCsv.open(QFile::ReadOnly | QFile::Text) ) {
        error = "File not exists";
        return false;
    }
    else
    {
        QTextStream in(&fileCsv);
        //skip header
        QString line = in.readLine();
        QStringList header = line.split(",");
        // whitespace removed from the start and the end.
        QMutableListIterator<QString> it(header);
        while (it.hasNext()) {
            it.next();
            it.value() = it.value().trimmed();
        }

        while (!in.atEnd())
        {
            line = in.readLine();
            QStringList items = line.split(",");
            if (items.size() < REQUIREDAGGREGATIONINFO)
            {
                error = "invalid output format CSV, missing reference data";
                return false;
            }

            int pos = header.indexOf("output variable (csv)");
            if (pos == -1)
            {
                error = "missing output variable";
                return false;
            }
            outputVarName.push_back(items[pos]);
            if (outputVarName.isEmpty())
            {
                error = "missing output variable";
                return false;
            }

            pos = header.indexOf("input field (shape)");
            if (pos == -1)
            {
                error = "missing input field (shape)";
                return false;
            }
            inputField.push_back(items[pos].toUpper());
            if (inputField.isEmpty())
            {
                error = "missing input field";
                return false;
            }


            pos = header.indexOf("computation");
            if (pos == -1)
            {
                error = "missing computation";
                return false;
            }
            computation.push_back(items[pos]);
            if (computation.isEmpty())
            {
                error = "missing computation";
                return false;
            }
        }
    }
    if (outputVarName.isEmpty())
    {
        return false;
    }
    return true;
}
