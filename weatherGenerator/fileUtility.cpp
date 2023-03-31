#include <QDebug>
#include <QFile>
#include <QTextStream>

#include "commonConstants.h"
#include "crit3dDate.h"
#include "weatherGenerator.h"
#include "fileUtility.h"


bool readMeteoDataCsv (QString &fileName, char mySeparator, double noData, TinputObsData* inputData)
{
    clearInputData(inputData);

    QFile file(fileName);
    if (!file.open(QIODevice::ReadOnly)) {
        qDebug() << "\nERROR!\n" << fileName << file.errorString();
        return false;
    }

    QStringList listDate;
    QStringList listTMin;
    QStringList listTMax;
    QStringList listPrecip;

    int indexLine = 0;
    int indexDate = 0;
    Crit3DDate tempDate;

    QString noDataString = QString::number(noData);
    QString strDate;

    // header
    file.readLine();

    while (!file.atEnd()) {
        QByteArray line = file.readLine();

        //check format
        if (line.split(mySeparator).count() < 5)
        {
            qDebug() << "ERROR!" << "\nfile =" << fileName << "\nline =" << indexLine+2;;
            qDebug() << "missing data / invalid format / invalid separator";
            qDebug() << "required separator =" << mySeparator <<"\n";
            return false;
        }

        //DATE
        strDate = line.split(mySeparator)[0];
        //check presence of quotation
        if (strDate.left(1) == "\"")
            strDate = strDate.mid(1, strDate.length()-2);
        listDate.append(strDate);

        // save the first date into the struct and check it is a valid date
        if (indexLine == 0)
        {
            inputData->inputFirstDate.year = listDate[indexLine].mid(0,4).toInt();
            if (inputData->inputFirstDate.year == 0)
            {
                qDebug() << "Invalid date format ";
                return false;
            }
            inputData->inputFirstDate.month = listDate[indexLine].mid(5,2).toInt();
            if (inputData->inputFirstDate.month == 0 || inputData->inputFirstDate.month > 12 )
            {
                qDebug() << "Invalid date format ";
                return false;
            }
            inputData->inputFirstDate.day = listDate[indexLine].mid(8,2).toInt();
            if (inputData->inputFirstDate.day == 0 || inputData->inputFirstDate.day > 31)
            {
                qDebug() << "Invalid date format ";
                return false;
            }
        }
        else
        {
            tempDate.year = listDate[indexLine].mid(0,4).toInt();
            tempDate.month = listDate[indexLine].mid(5,2).toInt();
            tempDate.day = listDate[indexLine].mid(8,2).toInt();

            indexDate = difference(inputData->inputFirstDate , tempDate );

            // check LACK of data
            if (indexDate != indexLine)
            {
                // insert nodata row
                listDate.removeLast();
                for (int i = indexLine; i < indexDate ; i++)
                {
                    listDate.append(noDataString);
                    listTMin.append(noDataString);
                    listTMax.append(noDataString);
                    listPrecip.append(noDataString);
                    indexLine++;
                }
                listDate.append(line.split(mySeparator)[0]);
            }
        }

        if (line.split(mySeparator)[1] == "" || line.split(mySeparator)[1] == " " || line.split(mySeparator)[1] == noDataString )
            listTMin.append(QString::number(NODATA));
        else
            listTMin.append(line.split(mySeparator)[1]);

        if (line.split(mySeparator)[2] == "" || line.split(mySeparator)[2] == " " || line.split(mySeparator)[2] == noDataString)
            listTMax.append(QString::number(NODATA));
        else
            listTMax.append(line.split(mySeparator)[2]);

        if (line.split(mySeparator)[4] == "" || line.split(mySeparator)[4] == " " || line.split(mySeparator)[4] == noDataString)
            listPrecip.append(QString::number(NODATA));
        else
            listPrecip.append(line.split(mySeparator)[4]);

        indexLine++;
    }

    file.close();

    // save and check the last date
    inputData->inputLastDate = tempDate;
    if (inputData->inputLastDate.year == 0)
    {
        qDebug() << "Invalid date format ";
        return false;
    }
    if (inputData->inputLastDate.month == 0 || inputData->inputLastDate.month > 12 )
    {
        qDebug() << "Invalid date format ";
        return false;
    }
    if (inputData->inputLastDate.day == 0 || inputData->inputLastDate.day > 31)
    {
        qDebug() << "Invalid date format ";
        return false;
    }

    if (listDate.length() != listTMin.length() || (listDate.length()!= listTMax.length() ) || (listDate.length() != listPrecip.length()) )
    {
        qDebug() << "list data - different size";
        return false;
    }

    inputData->dataLenght = listDate.length();
    inputData->inputTMin.resize(inputData->dataLenght);
    inputData->inputTMax.resize(inputData->dataLenght);
    inputData->inputPrecip.resize(inputData->dataLenght);

    for (int i = 0; i < inputData->dataLenght; i++)
    {
        inputData->inputTMin[i] = listTMin[i].toFloat();
        inputData->inputTMax[i] = listTMax[i].toFloat();
        inputData->inputPrecip[i] = listPrecip[i].toFloat();

        // check tmin <= tmax
        if ((inputData->inputTMin[i] != noData) && (inputData->inputTMax[i] != noData)
             && (inputData->inputTMin[i] > inputData->inputTMax[i]))
        {
            //qDebug() << "Warning: TMIN > TMAX: " << listDate[i];
            // switch
            inputData->inputTMin[i] = listTMax[i].toFloat();
            inputData->inputTMax[i] = listTMin[i].toFloat();
        }
    }

    listTMax.clear();
    listTMin.clear();
    listPrecip.clear();

    return true;
}


// write output of weather generator: a daily meteo data series
bool writeMeteoDataCsv(QString &fileName, char separator, std::vector<ToutputDailyMeteo> &dailyData)
{
    QFile file(fileName);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        qDebug() << file.errorString();
        return false;
    }

    QTextStream stream( &file );
    stream << "date" << separator << "tmin" << separator << "tmax" << separator << "prec\n";

    for (unsigned int i=0; i < dailyData.size(); i++)
    {
        QString month = QString::number(dailyData[i].date.month).rightJustified(2, '0');
        QString day = QString::number(dailyData[i].date.day).rightJustified(2, '0');
        QString year = QString::number(dailyData[i].date.year);
        QString myDate = year + "-" + month + "-" + day;

        QString tMin = QString::number(double(dailyData[i].minTemp), 'f', 1);
        QString tMax = QString::number(double(dailyData[i].maxTemp), 'f', 1);
        QString prec = QString::number(double(dailyData[i].prec), 'f', 1);

        stream << myDate << separator << tMin << separator << tMax << separator << prec << "\n";
    }

    return true;
}

