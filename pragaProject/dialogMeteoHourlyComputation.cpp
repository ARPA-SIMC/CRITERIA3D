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
    QHBoxLayout varLayout;
    QHBoxLayout targetLayout;
    QHBoxLayout dateLayout;
    QHBoxLayout periodLayout;
    QHBoxLayout displayLayout;
    QHBoxLayout genericPeriodLayout;
    QHBoxLayout layoutOk;

    QHBoxLayout elaborationLayout;

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

    meteoVariable var;

    Q_FOREACH (QString group, settings->childGroups())
    {
        if (! group.endsWith("_VarToElab1"))
            continue;

        std::string item;
        std::string variable = group.left(group.size()-11).toStdString(); // remove "_VarToElab1"
        try {
            var = MapHourlyMeteoVar.at(variable);
            item = MapHourlyMeteoVarToString.at(var);
        }
        catch (const std::out_of_range& ) {
            // check daily variable
            continue;
        }

        variableList.addItem(QString::fromStdString(item));
    }

    QLabel variableLabel("Variable: ");
    varLayout.addWidget(&variableLabel);
    varLayout.addWidget(&variableList);

    int currentYear = myProject.getCurrentDate().year();

    firstYearEdit.setText(QString::number(currentYear));
    firstYearEdit.setFixedWidth(110);
    firstYearEdit.setValidator(new QIntValidator(1800, 3000));

    lastYearEdit.setText(QString::number(currentYear));
    lastYearEdit.setFixedWidth(110);
    lastYearEdit.setValidator(new QIntValidator(1800, 3000));

    QLabel firstYearLabel("Start Year:");
    firstYearLabel.setBuddy(&firstYearEdit);

    dateLayout.addWidget(&firstYearLabel);
    dateLayout.addWidget(&firstYearEdit);

    QLabel lastYearLabel("End Year:");
    lastYearLabel.setBuddy(&lastYearEdit);

    dateLayout.addWidget(&lastYearLabel);
    dateLayout.addWidget(&lastYearEdit);

    QLabel periodTypeLabel("Period Type: ");
    periodTypeList.addItem("Generic");

    periodLayout.addWidget(&periodTypeLabel);
    periodLayout.addWidget(&periodTypeList);

    genericStartLabel.setText("Start Date:");
    genericPeriodStart.setDisplayFormat("dd/MM");
    genericStartLabel.setBuddy(&genericPeriodStart);
    genericEndLabel.setText("End Date:");
    genericPeriodEnd.setDisplayFormat("dd/MM");
    genericEndLabel.setBuddy(&genericPeriodEnd);
    nrYearLabel.setText("Delta Years:");
    nrYear.setValidator(new QIntValidator(-500, 500));
    nrYear.setText("0");
    nrYearLabel.setBuddy(&nrYear);

    genericStartLabel.setVisible(true);
    genericEndLabel.setVisible(true);
    genericPeriodStart.setVisible(true);
    genericPeriodEnd.setVisible(true);
    nrYearLabel.setVisible(true);
    nrYear.setVisible(true);

    genericPeriodLayout.addWidget(&genericStartLabel);
    genericPeriodLayout.addWidget(&genericPeriodStart);
    genericPeriodLayout.addWidget(&genericEndLabel);
    genericPeriodLayout.addWidget(&genericPeriodEnd);
    genericPeriodLayout.addWidget(&nrYearLabel);
    genericPeriodLayout.addWidget(&nrYear);

    displayPeriod("Generic");

    elaborationLayout.addWidget(new QLabel("Elaboration: "));
    QString currentVar = variableList.currentText();
    listElaboration(currentVar);
    elaborationLayout.addWidget(&elaborationList);

    elab1Parameter.setPlaceholderText("Parameter");
    elab1Parameter.setFixedWidth(90);
    elab1Parameter.setValidator(new QDoubleValidator(-9999., 9999., 2));

    QString elaborationField = elaborationList.currentText();
    if (MapElabWithParam.find(elaborationField.toStdString()) == MapElabWithParam.end())
    {
        elab1Parameter.clear();
        elab1Parameter.setReadOnly(true);
        adjustSize();
    }

    elaborationLayout.addWidget(&elab1Parameter);

    connect(&pointsButton, &QRadioButton::clicked, [=](){ targetChange(); });
    connect(&gridButton, &QRadioButton::clicked, [=](){ targetChange(); });

    connect(&periodTypeList, &QComboBox::currentTextChanged, [=](const QString &newVar){ this->displayPeriod(newVar); });
    connect(&variableList, &QComboBox::currentTextChanged, [=](const QString &newVar){ this->listElaboration(newVar); });

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    layoutOk.addWidget(&buttonBox);

    mainLayout.addWidget(targetGroupBox);
    mainLayout.addLayout(&varLayout);
    mainLayout.addLayout(&dateLayout);
    mainLayout.addLayout(&periodLayout);
    mainLayout.addLayout(&displayLayout);
    mainLayout.addLayout(&genericPeriodLayout);
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

    myProject.clima->setDailyCumulated(false);

    QString periodSelected = periodTypeList.currentText();
    QString value = variableList.currentText();
    meteoVariable var = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, value.toStdString());

    myProject.clima->setVariable(var);
    myProject.clima->setYearStart(firstYearEdit.text().toInt());
    myProject.clima->setYearEnd(lastYearEdit.text().toInt());
    myProject.clima->setPeriodStr(periodSelected);

    if (periodSelected == "Generic")
    {
        myProject.clima->setGenericPeriodDateStart(genericPeriodStart.date());
        myProject.clima->setGenericPeriodDateEnd(genericPeriodEnd.date());
        myProject.clima->setNYears(nrYear.text().toInt());
    }

    if (elaborationList.currentText() == "No elaboration available")
    {
        myProject.clima->setElab1("noMeteoComp");
    }
    else
    {
        myProject.clima->setElab1(elaborationList.currentText());
    }

    myProject.clima->setParam1IsClimate(false);
    if (! elab1Parameter.text().isEmpty())
    {
        myProject.clima->setParam1(elab1Parameter.text().toFloat());
    }
    else
    {
        myProject.clima->setParam1(NODATA);
    }

    QDialog::done(QDialog::Accepted);
}


void DialogMeteoHourlyComputation::displayPeriod(const QString value)
{
    if (value == "Generic")
    {
        elaborationList.setEnabled(true);

        genericStartLabel.setVisible(true);
        genericEndLabel.setVisible(true);
        genericPeriodStart.setVisible(true);
        genericPeriodEnd.setVisible(true);

        nrYearLabel.setVisible(true);
        nrYear.setVisible(true);
        nrYear.setText("0");
        nrYear.setEnabled(true);

        QDate defaultStart(firstYearEdit.text().toInt(),1,1);
        QDate defaultEnd(lastYearEdit.text().toInt(),1,1);
        genericPeriodStart.setDate(defaultStart);
        genericPeriodStart.setDisplayFormat("dd/MM");
        genericPeriodEnd.setDisplayFormat("dd/MM");
        genericPeriodEnd.setDate(defaultEnd);
        nrYear.setText("0");
        nrYear.setEnabled(true);
    }
}


void DialogMeteoHourlyComputation::listElaboration(const QString value)
{
    elaborationList.blockSignals(true);
    meteoVariable key = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, value.toStdString());
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


bool DialogMeteoHourlyComputation::checkValidData()
{
    if (firstYearEdit.text().size() != 4)
    {
        QMessageBox::information(nullptr, "Missing year", "Insert first year");
        return false;
    }

    if (lastYearEdit.text().size() != 4)
    {
        QMessageBox::information(nullptr, "Missing year", "Insert last year");
        return false;
    }

    if (firstYearEdit.text().toInt() > lastYearEdit.text().toInt())
    {
        QMessageBox::information(nullptr, "Invalid year", "first year greater than last year");
        return false;
    }

    if (MapElabWithParam.find(elaborationList.currentText().toStdString()) != MapElabWithParam.end())
    {
        if (elab1Parameter.text().isEmpty())
        {
            myProject.logWarning("Missing parameter!");
            return false;
        }
    }

    if (periodTypeList.currentText() == "Generic" && nrYear.text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing Parameter", "insert Nr Years");
        return false;
    }

    return true;
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
