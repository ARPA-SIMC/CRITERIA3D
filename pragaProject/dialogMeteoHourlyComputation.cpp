#include "dialogMeteoHourlyComputation.h"
#include "climate.h"
#include "utilities.h"
#include "pragaProject.h"

extern PragaProject myProject;

DialogMeteoHourlyComputation::DialogMeteoHourlyComputation(QSettings *settings, bool isMeteoGridLoaded, bool isMeteoPointLoaded)
    : settings(settings)
{
    setWindowTitle("Hourly computation");

    QVBoxLayout mainLayout;
    QHBoxLayout targetLayout;
    QHBoxLayout varLayout;
    QHBoxLayout timeRangeLayout;
    QHBoxLayout elaborationLayout;
    QHBoxLayout layoutOk;

    QGroupBox *targetGroupBox = new QGroupBox("Target");
    pointsButton.setText("meteo points");
    gridButton.setText("meteo grid");

    if (isMeteoPointLoaded)
    {
        pointsButton.setEnabled(true);
        if (! isMeteoGridLoaded)
        {
            pointsButton.setChecked(true);
        }
    }
    else
    {
        pointsButton.setEnabled(false);
    }
    if (isMeteoGridLoaded)
    {
        gridButton.setEnabled(true);
        if (! isMeteoPointLoaded)
        {
            gridButton.setChecked(true);
        }
    }
    else
    {
        gridButton.setEnabled(false);
    }

    if (isMeteoPointLoaded && isMeteoGridLoaded)
    {
        if (myProject.lastElabTargetIsGrid)
        {
            gridButton.setChecked(true);
            pointsButton.setChecked(false);
        }
        else
        {
            pointsButton.setChecked(true);
            gridButton.setChecked(false);
        }
    }

    if (pointsButton.isChecked())
    {
        isMeteoGrid = false;
        myProject.clima->setDb(myProject.meteoPointsDbHandler->getDb());
    }
    else if (gridButton.isChecked())
    {
        isMeteoGrid = true;
        myProject.clima->setDb(myProject.meteoGridDbHandler->db());
    }

    targetLayout.addWidget(&pointsButton);
    targetLayout.addWidget(&gridButton);
    targetGroupBox->setLayout(&targetLayout);

    // VARIABLES

    Q_FOREACH (QString group, settings->childGroups())
    {
        if (! group.endsWith("_VarToElab1"))
            continue;

        std::string varStr = group.left(group.size()-11).toStdString(); // remove '_VarToElab1'

        // check hourly variables
        if (MapHourlyMeteoVar.find(varStr) != MapHourlyMeteoVar.end())
        {
            variableList.addItem(QString::fromStdString(varStr));
        }
    }

    QLabel variableLabel("Variable: ");
    varLayout.addWidget(&variableLabel);
    varLayout.addWidget(&variableList);

    // TIME RANGE

    QLabel timeStartLabel("Start Date:");
    QLabel timeEndLabel("End Date:");

    timeRangeStart.setDate(myProject.getCurrentDate());
    timeRangeStart.setDisplayFormat("yyyy-MM-dd  ");
    timeRangeStart.setCalendarPopup(true);
    hourStart.setRange(0,23);
    hourStart.setValue(0);

    timeRangeEnd.setDate(myProject.getCurrentDate());
    timeRangeEnd.setDisplayFormat("yyyy-MM-dd  ");
    timeRangeEnd.setCalendarPopup(true);
    hourEnd.setRange(0,23);
    hourEnd.setValue(23);

    timeRangeLayout.addWidget(&timeStartLabel);
    timeRangeLayout.addWidget(&timeRangeStart);
    timeRangeLayout.addWidget(&hourStart);
    timeRangeLayout.addWidget(&timeEndLabel);
    timeRangeLayout.addWidget(&timeRangeEnd);
    timeRangeLayout.addWidget(&hourEnd);

    // ELABORATION

    elaborationLayout.addWidget(new QLabel("Elaboration: "));
    QString currentVar = variableList.currentText();
    listElaboration(currentVar);
    elaborationLayout.addWidget(&elaborationList);

    connect(&pointsButton, &QRadioButton::clicked, [=](){ targetChange(); });
    connect(&gridButton, &QRadioButton::clicked, [=](){ targetChange(); });
    connect(&variableList, &QComboBox::currentTextChanged, [=](const QString &newVar){ this->listElaboration(newVar); });

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    layoutOk.addWidget(&buttonBox);

    mainLayout.addWidget(targetGroupBox);
    mainLayout.addLayout(&varLayout);
    mainLayout.addLayout(&timeRangeLayout);
    mainLayout.addLayout(&elaborationLayout);
    mainLayout.addLayout(&layoutOk);

    setLayout(&mainLayout);

    show();
    exec();
}


void DialogMeteoHourlyComputation::done(bool res)
{
    // cancel or close was pressed
    if (! res)
    {
        QDialog::done(QDialog::Rejected);
        return;
    }

    // store elaboration values
    if (myProject.clima == nullptr)
    {
        QMessageBox::information(nullptr, "Error!", "clima is null...");
        return;
    }

    // check time range
    QDateTime dateStart, dateEnd;
    dateStart.setDate(timeRangeStart.date());
    dateStart.setTime(QTime(hourStart.value(), 0));
    dateEnd.setDate(timeRangeEnd.date());
    dateEnd.setTime(QTime(hourEnd.value(), 0));
    if (dateEnd < dateStart)
    {
        myProject.logWarning("Wrong time period!");
        return;
    }

    myProject.clima->setDailyCumulated(false);

    QString value = variableList.currentText();
    meteoVariable var = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, value.toStdString());

    myProject.clima->setVariable(var);

    myProject.clima->setGenericPeriodDateStart(timeRangeStart.date());
    myProject.clima->setGenericPeriodDateEnd(timeRangeEnd.date());

    myProject.clima->setYearStart(timeRangeStart.date().year());
    myProject.clima->setYearEnd(timeRangeEnd.date().year());

    myProject.clima->setHourStart(hourStart.value());
    myProject.clima->setHourEnd(hourEnd.value());

    if (elaborationList.currentText() == "No elaboration available")
    {
        myProject.clima->setElab1("noMeteoComp");
    }
    else
    {
        myProject.clima->setElab1(elaborationList.currentText());
    }

    // no parameters
    myProject.clima->setParam1IsClimate(false);
    myProject.clima->setParam1(NODATA);

    QDialog::done(QDialog::Accepted);
}


void DialogMeteoHourlyComputation::listElaboration(const QString variable)
{
    elaborationList.blockSignals(true);
    meteoVariable key = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, variable.toStdString());
    std::string keyString = getKeyStringMeteoMap(MapHourlyMeteoVar, key);

    QString group = QString::fromStdString(keyString) + "_VarToElab1";
    settings->beginGroup(group);
    int size = settings->beginReadArray(QString::fromStdString(keyString));

    QString prevElabStr = elaborationList.currentText();
    bool existsPrevElab = false;
    elaborationList.clear();

    for (int i = 0; i < size; ++i)
    {
        settings->setArrayIndex(i);
        QString elabStr = settings->value("elab").toString();
        if (prevElabStr == elabStr)
        {
            existsPrevElab = true;
        }
        elaborationList.addItem(elabStr);
    }

    if (elaborationList.count() == 0)
    {
        elaborationList.addItem("No elaboration available");
    }
    elaborationList.setEnabled(true);

    settings->endArray();
    settings->endGroup();

    this->adjustSize();

    elaborationList.blockSignals(false);

    if (existsPrevElab)
    {
        elaborationList.setCurrentText(prevElabStr);
    }
}


void DialogMeteoHourlyComputation::targetChange()
{
    if (pointsButton.isChecked())
    {
        isMeteoGrid = false;
        myProject.clima->setDb(myProject.meteoPointsDbHandler->getDb());
    }
    else if (gridButton.isChecked())
    {
        isMeteoGrid = true;
        myProject.clima->setDb(myProject.meteoGridDbHandler->db());
    }
}
