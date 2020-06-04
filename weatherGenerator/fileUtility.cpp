#include <QDebug>
#include <QFile>
#include <QTextStream>

#include "commonConstants.h"
#include "crit3dDate.h"
#include "weatherGenerator.h"
#include "fileUtility.h"


bool readMeteoDataCsv (QString namefile, char separator, double noData, TinputObsData* inputData)
{
    clearInputData(inputData);

    QFile file(namefile);
    if (!file.open(QIODevice::ReadOnly)) {
        qDebug() << "\nERROR!\n" << namefile << file.errorString();
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
        if (line.split(separator).count() < 5)
        {
            qDebug() << "ERROR!" << "\nfile =" << namefile << "\nline =" << indexLine+2;;
            qDebug() << "missing data / invalid format / invalid separator";
            qDebug() << "required separator =" << separator <<"\n";
            return false;
        }

        //DATE
        strDate = line.split(separator)[0];
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
                listDate.append(line.split(separator)[0]);
            }
        }

        if (line.split(separator)[1] == "" || line.split(separator)[1] == " " || line.split(separator)[1] == noDataString )
            listTMin.append(QString::number(NODATA));
        else
            listTMin.append(line.split(separator)[1]);

        if (line.split(separator)[2] == "" || line.split(separator)[2] == " " || line.split(separator)[2] == noDataString)
            listTMax.append(QString::number(NODATA));
        else
            listTMax.append(line.split(separator)[2]);

        if (line.split(separator)[4] == "" || line.split(separator)[4] == " " || line.split(separator)[4] == noDataString)
            listPrecip.append(QString::number(NODATA));
        else
            listPrecip.append(line.split(separator)[4]);

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
bool writeMeteoDataCsv (QString namefile, char separator, ToutputDailyMeteo* mydailyData, long dataLenght)
{

    QFile file(namefile);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        qDebug() << file.errorString();
        return false;
    }

    QTextStream stream( &file );

    QString myDate, tMin, tMax, prec;
    QString month, day;

    stream << "date" << separator << "tmin" << separator << "tmax" << separator << "tavg"
           << separator << "prec" << separator << "etp" << separator << "watertable\n";

    for (int i=0; i < dataLenght; i++)
    {
        if (mydailyData[i].date.month < 10)
            month = "0" + QString::number(mydailyData[i].date.month);
        else
            month = QString::number(mydailyData[i].date.month);

        if (mydailyData[i].date.day < 10)
            day = "0" + QString::number(mydailyData[i].date.day);
        else
            day = QString::number(mydailyData[i].date.day);

        myDate = QString::number(mydailyData[i].date.year) + "-" + month + "-" + day;
        tMin = QString::number(mydailyData[i].minTemp, 'f', 1);
        tMax = QString::number(mydailyData[i].maxTemp, 'f', 1);
        prec = QString::number(mydailyData[i].prec, 'f', 1);

        stream << myDate << separator << tMin << separator << tMax << separator
               << separator << prec << separator << separator << "\n";

    }

    return true;
}
