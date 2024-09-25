#include "aggregation.h"
#include "dialogSeriesOnZones.h"

#include <iostream>


DialogSeriesOnZones::DialogSeriesOnZones(QSettings *settings, QList<QString> aggregations, QDate currentDate, bool isHourly)
    : settings(settings), aggregations(aggregations), isHourly_(isHourly)
{
    setWindowTitle("Spatial average series on zones");

    QVBoxLayout mainLayout;
    QHBoxLayout varLayout;
    QHBoxLayout dateLayout;
    QHBoxLayout spatialElabLayout;
    QHBoxLayout layoutOk;

    Q_FOREACH (QString group, settings->childGroups())
    {
        if (! group.endsWith("_VarToElab1"))
            continue;

        meteoVariable var;
        std::string item;
        std::string variableStr = group.left(group.size()-11).toStdString();   // remove "_VarToElab1"

        if (isHourly_)
        {
            var = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, variableStr);
            item = MapHourlyMeteoVarToString.at(var);
        }
        else
        {
            var = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, variableStr);
            item = MapDailyMeteoVarToString.at(var);
        }

        if (var != noMeteoVar)
        {
            variableList.addItem(QString::fromStdString(item));
        }
    }

    QLabel variableLabel("Variable: ");
    varLayout.addWidget(&variableLabel);
    varLayout.addWidget(&variableList);

    genericStartLabel.setText("Start Date:");
    genericPeriodStart.setDate(currentDate);
    genericStartLabel.setBuddy(&genericPeriodStart);
    genericEndLabel.setText("End Date:");
    genericPeriodEnd.setDate(currentDate);
    genericEndLabel.setBuddy(&genericPeriodEnd);

    dateLayout.addWidget(&genericStartLabel);
    dateLayout.addWidget(&genericPeriodStart);
    dateLayout.addWidget(&genericEndLabel);
    dateLayout.addWidget(&genericPeriodEnd);

    QLabel spatialElabLabel("Spatial Elaboration: ");
    for (int i = 0; i < aggregations.size(); i++)
    {
        spatialElab.addItem(aggregations[i]);
    }

    spatialElabLayout.addWidget(&spatialElabLabel);
    spatialElabLayout.addWidget(&spatialElab);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    layoutOk.addWidget(&buttonBox);

    mainLayout.addLayout(&varLayout);
    mainLayout.addLayout(&dateLayout);
    mainLayout.addLayout(&spatialElabLayout);

    mainLayout.addLayout(&layoutOk);

    setLayout(&mainLayout);

    show();
    exec();

}


void DialogSeriesOnZones::done(bool res)
{
    if(res)  // ok was pressed
    {
        if (! checkValidData())
        {
            QDialog::done(QDialog::Rejected);
            return;
        }
        else  // validate the data
        {
            QDialog::done(QDialog::Accepted);
            return;
        }

    }
    else    // cancel, close or exc was pressed
    {
        QDialog::done(QDialog::Rejected);
        return;
    }
}


bool DialogSeriesOnZones::checkValidData()
{
    startDate_ = genericPeriodStart.date();
    endDate_ = genericPeriodEnd.date();

    if (startDate_ > endDate_)
    {
        QMessageBox::information(nullptr, "Invalid date", "first date greater than last date");
        return false;
    }

    QString var = variableList.currentText();
    if (isHourly_)
    {
        variable_ = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, var.toStdString());
    }
    else
    {
        variable_ = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, var.toStdString());
    }

    //spatialElaboration = getAggregationMethod(spatialElab.currentText().toStdString());
    spatialElaboration_ = spatialElab.currentText();

    return true;
}



