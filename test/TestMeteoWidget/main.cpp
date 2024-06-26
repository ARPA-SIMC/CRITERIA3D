#include <QApplication>

#include "meteoWidget.h"
#include "meteoPoint.h"
#include "utilities.h"

#include "dbMeteoPointsHandler.h"
#include <iostream>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    QFileDialog openDBDialog;

    QString dbFileName = openDBDialog.getOpenFileName(nullptr, ("Open mp database"), "", ("SQLite files (*.db)"));
    if (dbFileName.isEmpty())
    {
        return 0;
    }

    Crit3DMeteoPointsDbHandler mpHandler(dbFileName);
    mpHandler.loadVariableProperties();

    QString errorString;
    QList<Crit3DMeteoPoint> meteoPointList;
    gis::Crit3DGisSettings gisSettings;
    if (! mpHandler.getPropertiesFromDb(meteoPointList, gisSettings, errorString))
    {
        QMessageBox::warning(nullptr, "", errorString);
        return -1;
    }

    if (meteoPointList.isEmpty())
    {
        QMessageBox::warning(nullptr, "", "Missing data.");
        return 0;
    }

    // load daily data
    QDateTime firstDaily = mpHandler.getFirstDate(daily);
    QDateTime lastDaily = mpHandler.getLastDate(daily);
    if (! firstDaily.isNull() && ! lastDaily.isNull())
    {
        mpHandler.loadDailyData(getCrit3DDate(firstDaily.date()), getCrit3DDate(lastDaily.date()), &meteoPointList[0]);
        if (meteoPointList.size() > 1)
        {
            mpHandler.loadDailyData(getCrit3DDate(firstDaily.date()), getCrit3DDate(lastDaily.date()), &meteoPointList[1]);
        }
    }

    // load hourly data
    QDateTime firstHourly = mpHandler.getFirstDate(hourly);
    QDateTime lastHourly = mpHandler.getLastDate(hourly);
    if (! firstHourly.isNull() && ! lastHourly.isNull())
    {
        mpHandler.loadHourlyData(getCrit3DDate(firstHourly.date()), getCrit3DDate(lastHourly.date()), &meteoPointList[0]);
        if (meteoPointList.size() > 1)
        {
            mpHandler.loadHourlyData(getCrit3DDate(firstHourly.date()), getCrit3DDate(lastHourly.date()), &meteoPointList[1]);
        }
    }

    if (lastDaily.isNull() && lastHourly.isNull())
    {
        QMessageBox::warning(nullptr, "", "Missing data.");
        return 0;
    }

    // set last date
    QDate lastDate;
    if (! lastDaily.isNull())
    {
        lastDate = lastDaily.date();
        if (! lastHourly.isNull())
        {
            lastDate = std::max(lastDate, lastHourly.date());
        }
    }
    else
    {
        lastDate = lastHourly.date();
    }

    bool isGrid = false;
    Crit3DMeteoSettings meteoSettings;
    Crit3DMeteoWidget widget(isGrid, "", &meteoSettings);

    widget.show();
    widget.setCurrentDate(lastDate);
    widget.drawMeteoPoint(meteoPointList[0], false);

    // add second point
    if (meteoPointList.size() > 1)
    {
        widget.drawMeteoPoint(meteoPointList[1], true);
    }

    return a.exec();
}
 
