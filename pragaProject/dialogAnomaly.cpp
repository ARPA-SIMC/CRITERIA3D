#include "dialogAnomaly.h"
#include "climate.h"
#include "dbClimate.h"
#include "utilities.h"
#include "pragaProject.h"

extern PragaProject myProject;

DialogAnomaly::DialogAnomaly()
{ }

void DialogAnomaly::build(QSettings *AnomalySettings)
{

    this->AnomalySettings = AnomalySettings;

    readReference.setText("Read reference climate from db");
    climateDbClimaList.setVisible(readReference.isChecked());
    varLayout.addWidget(&readReference);
    varLayout.addWidget(&climateDbClimaList);

    currentDayLabel.setText("Day/Month:");


    if (myProject.referenceClima->genericPeriodDateStart() == QDate(1800,1,1))
    {
        currentDay.setDate(myProject.getCurrentDate());
    }
    else
    {
        currentDay.setDate(myProject.referenceClima->genericPeriodDateStart());
    }

    currentDay.setDisplayFormat("dd/MM");
    currentDayLabel.setBuddy(&currentDay);
    currentDayLabel.setVisible(true);
    currentDay.setVisible(true);

    int currentYear = myProject.getCurrentDate().year();

    firstDateLabel.setText("Start Year:");
    if (myProject.referenceClima->yearStart() == NODATA)
    {
        firstYearEdit.setText(QString::number(currentYear));
    }
    else
    {
        firstYearEdit.setText(QString::number(myProject.referenceClima->yearStart()));
    }

    firstYearEdit.setFixedWidth(110);
    firstYearEdit.setValidator(new QIntValidator(1800, 3000));
    firstDateLabel.setBuddy(&firstYearEdit);

    lastDateLabel.setText("End Year:");
    if (myProject.referenceClima->yearEnd() == NODATA)
    {
        lastYearEdit.setText(QString::number(currentYear));
    }
    else
    {
        lastYearEdit.setText(QString::number(myProject.referenceClima->yearEnd()));
    }

    lastYearEdit.setFixedWidth(110);
    lastYearEdit.setValidator(new QIntValidator(1800, 3000));
    lastDateLabel.setBuddy(&lastYearEdit);

    dateLayout.addWidget(&currentDayLabel);
    dateLayout.addWidget(&currentDay);

    dateLayout.addWidget(&firstDateLabel);
    dateLayout.addWidget(&firstYearEdit);

    dateLayout.addWidget(&lastDateLabel);
    dateLayout.addWidget(&lastYearEdit);

    periodTypeList.addItem("Daily");
    periodTypeList.addItem("Decadal");
    periodTypeList.addItem("Monthly");
    periodTypeList.addItem("Seasonal");
    periodTypeList.addItem("Annual");
    periodTypeList.addItem("Generic");

    periodTypeLabel.setText("Period Type: ");
    periodLayout.addWidget(&periodTypeLabel);
    periodLayout.addWidget(&periodTypeList);

    QString periodSelected = periodTypeList.currentText();
    int dayOfYear = currentDay.date().dayOfYear();
    periodDisplay.setText("Day Of Year: " + QString::number(dayOfYear));
    periodDisplay.setReadOnly(true);

    displayLayout.addWidget(&periodDisplay);

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

    genericStartLabel.setVisible(false);
    genericEndLabel.setVisible(false);
    genericPeriodStart.setVisible(false);
    genericPeriodEnd.setVisible(false);
    nrYearLabel.setVisible(false);
    nrYear.setVisible(false);

    genericPeriodLayout.addWidget(&genericStartLabel);
    genericPeriodLayout.addWidget(&genericPeriodStart);
    genericPeriodLayout.addWidget(&genericEndLabel);
    genericPeriodLayout.addWidget(&genericPeriodEnd);
    genericPeriodLayout.addWidget(&nrYearLabel);
    genericPeriodLayout.addWidget(&nrYear);

    elab.setText("Elaboration: ");
    elaborationLayout.addWidget(&elab);

    meteoVariable key = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, variableElab.text().toStdString());
    std::string keyString = getKeyStringMeteoMap(MapDailyMeteoVar, key);
    QString group = QString::fromStdString(keyString)+"_VarToElab1";
    AnomalySettings->beginGroup(group);
    int size = AnomalySettings->beginReadArray(QString::fromStdString(keyString));
    for (int i = 0; i < size; ++i) {
        AnomalySettings->setArrayIndex(i);
        QString elab = AnomalySettings->value("elab").toString();
        elaborationList.addItem( elab );
    }
    AnomalySettings->endArray();
    AnomalySettings->endGroup();
    elaborationLayout.addWidget(&elaborationList);

    elab1Parameter.setPlaceholderText("Parameter");
    elab1Parameter.setFixedWidth(90);
    elab1Parameter.setValidator(new QDoubleValidator(-9999.0, 9999.0, 2));
    readParam.setText("Read param from db Climate");
    readParam.setChecked(false);
    climateDbElabList.setVisible(false);


    QString elab1Field = elaborationList.currentText();
    if ( MapElabWithParam.find(elab1Field.toStdString()) == MapElabWithParam.end())
    {
        elab1Parameter.clear();
        elab1Parameter.setReadOnly(true);
        readParam.setCheckable(false);
        climateDbElabList.setVisible(false);
        adjustSize();
    }
    else
    {
        readParam.setCheckable(true);
        if (!readParam.isChecked())
        {
            elab1Parameter.setReadOnly(false);
        }
    }


    elaborationLayout.addWidget(&elab1Parameter);
    readParamLayout.addWidget(&readParam);
    readParamLayout.addWidget(&climateDbElabList);

    secondElab.setText("Secondary Elaboration: ");
    secondElabLayout.addWidget(&secondElab);

    if (firstYearEdit.text().toInt() == lastYearEdit.text().toInt())
    {
        secondElabList.addItem("No elaboration available");
    }
    else
    {
        group = elab1Field +"_Elab1Elab2";
        AnomalySettings->beginGroup(group);
        secondElabList.addItem("None");
        size = AnomalySettings->beginReadArray(elab1Field);
        for (int i = 0; i < size; ++i) {
            AnomalySettings->setArrayIndex(i);
            QString elab2 = AnomalySettings->value("elab2").toString();
            secondElabList.addItem( elab2 );
        }
        AnomalySettings->endArray();
        AnomalySettings->endGroup();
    }
    secondElabLayout.addWidget(&secondElabList);

    elab2Parameter.setPlaceholderText("Parameter");
    elab2Parameter.setFixedWidth(90);
    elab2Parameter.setValidator(new QDoubleValidator(-9999.0, 9999.0, 2));

    QString elab2Field = secondElabList.currentText();
    if ( MapElabWithParam.find(elab2Field.toStdString()) == MapElabWithParam.end())
    {
        elab2Parameter.clear();
        elab2Parameter.setReadOnly(true);
    }
    else
    {
        elab2Parameter.setReadOnly(false);
    }

    secondElabLayout.addWidget(&elab2Parameter);

    connect(&firstYearEdit, &QLineEdit::editingFinished, [=](){ this->AnomalyCheckYears(); });
    connect(&lastYearEdit, &QLineEdit::editingFinished, [=](){ this->AnomalyCheckYears(); });
    connect(&currentDay, &QDateEdit::dateChanged, [=](){ this->AnomalyDisplayPeriod(periodTypeList.currentText()); });
    connect(&variableElab, &QLineEdit::textChanged, [=](const QString &newVar){ this->AnomalyListElaboration(newVar); });

    connect(&periodTypeList, &QComboBox::currentTextChanged, [=](const QString &newVar){ this->AnomalyDisplayPeriod(newVar); });
    connect(&elaborationList, &QComboBox::currentTextChanged, [=](const QString &newElab){ this->AnomalyListSecondElab(newElab); });
    connect(&secondElabList, &QComboBox::currentTextChanged, [=](const QString &newSecElab){ this->AnomalyActiveSecondParameter(newSecElab); });
    connect(&readReference, &QCheckBox::stateChanged, [=](int state){ this->AnomalyReadReferenceState(state); });
    connect(&readParam, &QCheckBox::stateChanged, [=](int state){ this->AnomalyReadParameter(state); });


    mainLayout.addLayout(&varLayout);
    mainLayout.addLayout(&dateLayout);
    mainLayout.addLayout(&periodLayout);
    mainLayout.addLayout(&displayLayout);
    mainLayout.addLayout(&genericPeriodLayout);
    mainLayout.addLayout(&elaborationLayout);
    mainLayout.addLayout(&readParamLayout);
    mainLayout.addLayout(&secondElabLayout);

    setLayout(&mainLayout);

    // show stored values

    if (myProject.referenceClima->elab1() != "")
    {
        elaborationList.setCurrentText(myProject.referenceClima->elab1());
        if ( (myProject.referenceClima->param1() != NODATA) && (!myProject.referenceClima->param1IsClimate()) )
        {
            elab1Parameter.setReadOnly(false);
            elab1Parameter.setText(QString::number(myProject.referenceClima->param1()));
        }
        else if (myProject.referenceClima->param1IsClimate())
        {
            elab1Parameter.clear();
            elab1Parameter.setReadOnly(true);
            readParam.setChecked(true);
            climateDbElabList.setVisible(true);
            adjustSize();
            climateDbElabList.setCurrentText(myProject.referenceClima->param1ClimateField());
        }
        else
        {
            readParam.setChecked(false);
            climateDbElabList.setVisible(false);
            adjustSize();
        }
    }
    if (myProject.referenceClima->elab2() != "")
    {
        secondElabList.setCurrentText(myProject.referenceClima->elab2());
        if (myProject.referenceClima->param2() != NODATA)
        {
            elab2Parameter.setText(QString::number(myProject.referenceClima->param2()));
        }
    }
    if (myProject.referenceClima->periodStr() != "")
    {
            periodTypeList.setCurrentText(myProject.referenceClima->periodStr());
            if (myProject.referenceClima->periodStr() == "Generic")
            {
                genericPeriodStart.setDate(myProject.referenceClima->genericPeriodDateStart());
                genericPeriodEnd.setDate(myProject.referenceClima->genericPeriodDateEnd());
                nrYear.setText(QString::number(myProject.referenceClima->nYears()));
            }
    }

}

void DialogAnomaly::AnomalySetVariableElab(const QString &value)
{
    variableElab.setText(value);
}

void DialogAnomaly::AnomalyListElaboration(const QString value)
{

    meteoVariable key = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, value.toStdString());
    std::string keyString = getKeyStringMeteoMap(MapDailyMeteoVar, key);
    QString group = QString::fromStdString(keyString)+"_VarToElab1";
    AnomalySettings->beginGroup(group);
    int size = AnomalySettings->beginReadArray(QString::fromStdString(keyString));
    elaborationList.clear();

    for (int i = 0; i < size; ++i)
    {
        AnomalySettings->setArrayIndex(i);
        QString elab = AnomalySettings->value("elab").toString();
        elaborationList.addItem( elab );

    }
    AnomalySettings->endArray();
    AnomalySettings->endGroup();

    if (myProject.referenceClima->variable() == key && myProject.referenceClima->elab1() != "")
    {
        elaborationList.setCurrentText(myProject.referenceClima->elab1());
        if (myProject.referenceClima->param1() != NODATA)
        {
            elab1Parameter.setText(QString::number(myProject.referenceClima->param1()));
        }
    }

    readParam.setChecked(false);
    climateDbElabList.setVisible(false);
    adjustSize();

    AnomalyListSecondElab(elaborationList.currentText());
}


void DialogAnomaly::AnomalyCheckYears()
{
    if (firstYearEdit.text().toInt() == lastYearEdit.text().toInt())
    {
        secondElabList.clear();
        secondElabList.addItem("No elaboration available");
    }
    else
    {
        AnomalyListSecondElab(elaborationList.currentText());
    }
}


void DialogAnomaly::AnomalyDisplayPeriod(const QString value)
{

    if (value == "Daily")
    {
        periodDisplay.setVisible(true);
        currentDayLabel.setVisible(true);
        currentDay.setVisible(true);
        genericStartLabel.setVisible(false);
        genericEndLabel.setVisible(false);
        genericPeriodStart.setVisible(false);
        genericPeriodEnd.setVisible(false);
        nrYearLabel.setVisible(false);
        nrYear.setVisible(false);
        int dayOfYear = currentDay.date().dayOfYear();
        periodDisplay.setText("Day Of Year: " + QString::number(dayOfYear));
    }
    else if (value == "Decadal")
    {
        periodDisplay.setVisible(true);
        currentDayLabel.setVisible(true);
        currentDay.setVisible(true);
        genericStartLabel.setVisible(false);
        genericEndLabel.setVisible(false);
        genericPeriodStart.setVisible(false);
        genericPeriodEnd.setVisible(false);
        nrYearLabel.setVisible(false);
        nrYear.setVisible(false);
        int decade = decadeFromDate(currentDay.date());
        periodDisplay.setText("Decade: " + QString::number(decade));
    }
    else if (value == "Monthly")
    {
        periodDisplay.setVisible(true);
        currentDayLabel.setVisible(true);
        currentDay.setVisible(true);
        genericStartLabel.setVisible(false);
        genericEndLabel.setVisible(false);
        genericPeriodStart.setVisible(false);
        genericPeriodEnd.setVisible(false);
        nrYearLabel.setVisible(false);
        nrYear.setVisible(false);
        periodDisplay.setText("Month: " + QString::number(currentDay.date().month()));
    }
    else if (value == "Seasonal")
    {
        periodDisplay.setVisible(true);
        currentDayLabel.setVisible(true);
        currentDay.setVisible(true);
        genericStartLabel.setVisible(false);
        genericEndLabel.setVisible(false);
        genericPeriodStart.setVisible(false);
        genericPeriodEnd.setVisible(false);
        nrYearLabel.setVisible(false);
        nrYear.setVisible(false);
        QString season = getStringSeasonFromDate(currentDay.date());
        periodDisplay.setText("Season: " + season);
    }
    else if (value == "Annual")
    {
        periodDisplay.setVisible(false);
        currentDayLabel.setVisible(true);
        currentDay.setVisible(true);
        genericStartLabel.setVisible(false);
        genericEndLabel.setVisible(false);
        genericPeriodStart.setVisible(false);
        genericPeriodEnd.setVisible(false);
        nrYearLabel.setVisible(false);
        nrYear.setVisible(false);
    }
    else if (value == "Generic")
    {
        periodDisplay.setVisible(false);
        currentDayLabel.setVisible(false);
        currentDay.setVisible(false);

        genericStartLabel.setVisible(true);
        genericEndLabel.setVisible(true);
        genericPeriodStart.setVisible(true);
        genericPeriodEnd.setVisible(true);

        nrYearLabel.setVisible(true);
        nrYear.setVisible(true);
        nrYear.setText("0");
        nrYear.setEnabled(true);

        if (elaborationList.currentText().toStdString() == "huglin" || elaborationList.currentText().toStdString() == "fregoni")
        {
            QDate fixStart(firstYearEdit.text().toInt(),4,1);
            QDate fixEnd(lastYearEdit.text().toInt(),9,30);
            genericPeriodStart.setDate(fixStart);
            genericPeriodStart.setDisplayFormat("dd/MM");
            genericPeriodEnd.setDisplayFormat("dd/MM");
            genericPeriodEnd.setDate(fixEnd);
            nrYear.setText("0");
            nrYear.setEnabled(false);
        }
        else if (elaborationList.currentText().toStdString() == "winkler")
        {
            QDate fixStart(firstYearEdit.text().toInt(),4,1);
            QDate fixEnd(lastYearEdit.text().toInt(),10,31);
            genericPeriodStart.setDate(fixStart);
            genericPeriodStart.setDisplayFormat("dd/MM");
            genericPeriodEnd.setDisplayFormat("dd/MM");
            genericPeriodEnd.setDate(fixEnd);
            nrYear.setText("0");
            nrYear.setEnabled(false);
        }
        else
        {
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

}

void DialogAnomaly::AnomalyListSecondElab(const QString value)
{

    if ( MapElabWithParam.find(value.toStdString()) == MapElabWithParam.end())
    {
        elab1Parameter.clear();
        elab1Parameter.setReadOnly(true);
        readParam.setChecked(false);
        climateDbElabList.setVisible(false);
        adjustSize();
        readParam.setCheckable(false);
    }
    else
    {
        elab1Parameter.setReadOnly(false);
        readParam.setCheckable(true);
    }

    if (elaborationList.currentText().toStdString() == "huglin" || elaborationList.currentText().toStdString() == "winkler" || elaborationList.currentText().toStdString() == "fregoni")
    {
        periodTypeList.setCurrentText("Generic");
        periodTypeList.setEnabled(false);
        AnomalyDisplayPeriod(periodTypeList.currentText());
    }
    else
    {
        periodTypeList.setEnabled(true);
        nrYear.setEnabled(true);
    }

    QString group = value + "_Elab1Elab2";
    AnomalySettings->beginGroup(group);
    int size = AnomalySettings->beginReadArray(value);

    if (size == 0 || firstYearEdit.text().toInt() == lastYearEdit.text().toInt())
    {
        secondElabList.clear();
        secondElabList.addItem("No elaboration available");
        AnomalySettings->endArray();
        AnomalySettings->endGroup();
        return;
    }
    secondElabList.clear();
    secondElabList.addItem("None");
    for (int i = 0; i < size; ++i) {
        AnomalySettings->setArrayIndex(i);
        QString elab2 = AnomalySettings->value("elab2").toString();
        secondElabList.addItem( elab2 );
    }
    AnomalySettings->endArray();
    AnomalySettings->endGroup();

    if (myProject.referenceClima->elab1() == value && myProject.referenceClima->elab2() != "")
    {
        secondElabList.setCurrentText(myProject.referenceClima->elab2());
        if (myProject.referenceClima->param2() != NODATA)
        {
            elab2Parameter.setText(QString::number(myProject.referenceClima->param2()));
        }
    }

}

void DialogAnomaly::AnomalyActiveSecondParameter(const QString value)
{

        if ( MapElabWithParam.find(value.toStdString()) == MapElabWithParam.end())
        {
            elab2Parameter.clear();
            elab2Parameter.setReadOnly(true);
        }
        else
        {
            elab2Parameter.setReadOnly(false);
        }
}

void DialogAnomaly::AnomalyReadParameter(int state)
{

    climateDbElabList.clear();
    climateDbElab.clear();

    if (state!= 0)
    {
        climateDbElabList.setVisible(true);
        adjustSize();
        AnomalyFillClimateDbList(&climateDbElabList);
        elab1Parameter.clear();
        elab1Parameter.setReadOnly(true);
    }
    else
    {
        climateDbElabList.setVisible(false);
        adjustSize();
        elab1Parameter.setReadOnly(false);
    }

}

void DialogAnomaly::AnomalyFillClimateDbList(QComboBox* dbList)
{
    QList<QString> climateTables;
    QString myError  = myProject.errorString;
    if (! showClimateTables(myProject.clima->db(), &myError, &climateTables) )
    {
        dbList->addItem("No saved elaborations found");
    }
    else
    {
        for (int i=0; i < climateTables.size(); i++)
        {
            selectVarElab(myProject.clima->db(), &myError, climateTables[i], variableElab.text(), &climateDbElab);
        }
        if (climateDbElab.isEmpty())
        {
            dbList->addItem("No saved elaborations found");
        }
        else
        {
            for (int i=0; i < climateDbElab.size(); i++)
            {
                dbList->addItem(climateDbElab[i]);
            }
        }
    }
    return;
}

void DialogAnomaly::AnomalyReadReferenceState(int state)
{

    if (state!= 0)
    {
        readReference.setChecked(true);
        climateDbClimaList.clear();
        climateDbElab.clear();
        climateDbClimaList.setVisible(true);
        AnomalySetAllEnable(false);
        AnomalyFillClimateDbList(&climateDbClimaList);
        adjustSize();

    }
    else
    {
        readReference.setChecked(false);
        AnomalySetAllEnable(true);
        climateDbElabList.setVisible(readParam.isChecked());
        climateDbClimaList.setVisible(false);
        adjustSize();
    }
}

void DialogAnomaly::AnomalySetAllEnable(bool set)
{

    currentDay.setEnabled(set);
    firstYearEdit.setEnabled(set);
    lastYearEdit.setEnabled(set);
    genericPeriodStart.setEnabled(set);
    genericPeriodEnd.setEnabled(set);
    nrYear.setEnabled(set);
    readParam.setEnabled(set);
    periodTypeList.setEnabled(set);
    elaborationList.setEnabled(set);
    secondElabList.setEnabled(set);
    periodDisplay.setVisible(set);
    elab1Parameter.setEnabled(set);
    elab2Parameter.setEnabled(set);
    climateDbElabList.setVisible(set);

}

bool DialogAnomaly::AnomalyGetReadReferenceState()
{
    return readReference.isChecked();
}

QString DialogAnomaly::AnomalyGetPeriodTypeList() const
{
    return periodTypeList.currentText();
}

void DialogAnomaly::AnomalySetPeriodTypeList(QString period)
{
    int index = periodTypeList.findText(period, Qt::MatchFixedString);
    periodTypeList.setCurrentIndex(index);
}

int DialogAnomaly::AnomalyGetYearStart() const
{
    return firstYearEdit.text().toInt();
}

void DialogAnomaly::AnomalySetYearStart(QString year)
{
    firstYearEdit.setText(year);
}

int DialogAnomaly::AnomalyGetYearLast() const
{
    return lastYearEdit.text().toInt();
}

void DialogAnomaly::AnomalySetYearLast(QString year)
{
    lastYearEdit.setText(year);
}

QDate DialogAnomaly::AnomalyGetGenericPeriodStart() const
{
    return genericPeriodStart.date();
}

void DialogAnomaly::AnomalySetGenericPeriodStart(QDate genericStart)
{
    genericPeriodStart.setDate(genericStart);
}

QDate DialogAnomaly::AnomalyGetGenericPeriodEnd() const
{
    return genericPeriodEnd.date();
}

void DialogAnomaly::AnomalySetGenericPeriodEnd(QDate genericEnd)
{
    genericPeriodEnd.setDate(genericEnd);
}

QDate DialogAnomaly::AnomalyGetCurrentDay() const
{
    return currentDay.date();
}

void DialogAnomaly::AnomalySetCurrentDay(QDate date)
{
    currentDay.setDate(date);
}

int DialogAnomaly::AnomalyGetNyears() const
{
    return nrYear.text().toInt();
}

void DialogAnomaly::AnomalySetNyears(QString nYears)
{
    nrYear.setText(nYears);
}

QString DialogAnomaly::AnomalyGetElaboration() const
{
    return elaborationList.currentText();
}

bool DialogAnomaly::AnomalySetElaboration(QString elab)
{
    int index = elaborationList.findText(elab, Qt::MatchFixedString);
    if (index == -1)
    {
        return false;
    }
    elaborationList.setCurrentIndex(index);
    return true;
}

QString DialogAnomaly::AnomalyGetSecondElaboration() const
{
    return secondElabList.currentText();
}

bool DialogAnomaly::AnomalySetSecondElaboration(QString elab)
{
    int index = secondElabList.findText(elab, Qt::MatchFixedString);
    if (index == -1)
    {
        return false;
    }
    secondElabList.setCurrentIndex(index);
    return true;
}

QString DialogAnomaly::AnomalyGetParam1() const
{
    return elab1Parameter.text();
}

void DialogAnomaly::AnomalySetParam1(QString param)
{
    elab1Parameter.setText(param);
}

void DialogAnomaly::AnomalySetParam1ReadOnly(bool visible)
{
    elab1Parameter.setReadOnly(visible);
}

QString DialogAnomaly::AnomalyGetParam2() const
{
    return elab2Parameter.text();
}

void DialogAnomaly::AnomalySetParam2(QString param)
{
    elab2Parameter.setText(param);
}

bool DialogAnomaly::AnomalyReadParamIsChecked() const
{
    return readParam.isChecked();
}

void DialogAnomaly::AnomalySetReadParamIsChecked(bool set)
{
    readParam.setChecked(set);
}

QString DialogAnomaly::AnomalyGetClimateDbElab() const
{
    return climateDbElabList.currentText();
}

void DialogAnomaly::AnomalySetClimateDbElab(QString elab)
{
    int index = climateDbElabList.findText(elab, Qt::MatchFixedString);
    climateDbElabList.setCurrentIndex(index);
}

QString DialogAnomaly::AnomalyGetClimateDb() const
{
    return climateDbClimaList.currentText();
}

bool DialogAnomaly::AnomalySetClimateDb(QString clima)
{
    int index = climateDbClimaList.findText(clima, Qt::MatchFixedString);
    if (index == -1)
    {
        return false;
    }
    climateDbClimaList.setCurrentIndex(index);
    return true;
}

bool DialogAnomaly::AnomalyCheckValidData()
{
    if (readReference.isChecked())
    {
        return true;
    }
    if (firstYearEdit.text().size() != 4)
    {
        QMessageBox::information(nullptr, "Anomaly missing year", "Insert first year");
        return false;
    }
    if (lastYearEdit.text().size() != 4)
    {
        QMessageBox::information(nullptr, "Anomaly missing year", "Insert last year");
        return false;
    }

    if (firstYearEdit.text().toInt() > lastYearEdit.text().toInt())
    {
        QMessageBox::information(nullptr, "Anomaly invalid year", "first year greater than last year");
        return false;
    }
    if (elaborationList.currentText().toStdString() == "huglin" || elaborationList.currentText().toStdString() == "winkler" || elaborationList.currentText().toStdString() == "fregoni")
    {
        if (secondElabList.currentText().toStdString() == "None")
        {
            QMessageBox::information(nullptr, "Anomaly second Elaboration missing", elaborationList.currentText() + " requires second elaboration");
            return false;
        }

    }
    if ( MapElabWithParam.find(elaborationList.currentText().toStdString()) != MapElabWithParam.end())
    {
        if ( (!readParam.isChecked() && elab1Parameter.text().isEmpty()) || (readParam.isChecked() && climateDbElabList.currentText() == "No saved elaborations found" ))
        {
            QMessageBox::information(nullptr, "Anomaly missing Parameter", "insert parameter");
            return false;
        }
    }
    if ( MapElabWithParam.find(secondElabList.currentText().toStdString()) != MapElabWithParam.end())
    {
        if (elab2Parameter.text().isEmpty())
        {
            QMessageBox::information(nullptr, "Anomaly missing Parameter", "insert second elaboration parameter");
            return false;
        }
    }
    if (periodTypeList.currentText() == "Generic")
    {
        if (nrYear.text().isEmpty())
        {
            QMessageBox::information(nullptr, "Anomaly missing Parameter", "insert Nr Years");
            return false;
        }
    }
    return true;

}
