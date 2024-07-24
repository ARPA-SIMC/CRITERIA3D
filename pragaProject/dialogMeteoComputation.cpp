#include "dialogMeteoComputation.h"
#include "climate.h"
#include "dbClimate.h"
#include "utilities.h"
#include "pragaProject.h"
#include "dialogXMLComputation.h"

extern PragaProject myProject;

DialogMeteoComputation::DialogMeteoComputation(QSettings *settings, bool isMeteoGridLoaded, bool isMeteoPointLoaded, bool isAnomaly, bool saveClima)
    : settings(settings), isMeteoGridLoaded(isMeteoGridLoaded), isMeteoPointLoaded(isMeteoPointLoaded), isAnomaly(isAnomaly), saveClima(saveClima)
{

    if (saveClima)
    {
        setWindowTitle("Climate Elaboration");
    }
    else if (!isAnomaly)
    {
        setWindowTitle("Elaboration");
    }
    else
    {
        setWindowTitle("Anomaly");
    }

    QVBoxLayout mainLayout;
    QHBoxLayout varLayout;
    QHBoxLayout targetLayout;
    QHBoxLayout dateLayout;
    QHBoxLayout periodLayout;
    QHBoxLayout displayLayout;
    QHBoxLayout genericPeriodLayout;
    QHBoxLayout layoutXML;
    QHBoxLayout layoutOk;

    QHBoxLayout elaborationLayout;
    QHBoxLayout readParamLayout;
    QHBoxLayout secondElabLayout;

    QVBoxLayout anomalyMainLayout;
    QHBoxLayout buttonLayout;
    QVBoxLayout saveClimaMainLayout;
    QFrame anomalyLine;

    anomalyLine.setFrameShape(QFrame::HLine);
    anomalyLine.setFrameShadow(QFrame::Sunken);
    QLabel anomalyLabel("<font color='red'>Reference Data:</font>");
    copyData.setText("Copy data above");
    copyData.setMaximumWidth(this->width()/3);


    QGroupBox *targetGroupBox = new QGroupBox("Target");
    pointsButton.setText("meteo points");
    gridButton.setText("meteo grid");

    if (isMeteoPointLoaded)
    {
        pointsButton.setEnabled(true);
        if (!isMeteoGridLoaded)
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
        if (!isMeteoPointLoaded)
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
        if (myProject.lastElabTargetisGrid)
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
        if (!group.endsWith("_VarToElab1"))
            continue;
        std::string item;
        std::string variable = group.left(group.size()-11).toStdString(); // remove "_VarToElab1"
        try {
          var = MapDailyMeteoVar.at(variable);
          item = MapDailyMeteoVarToString.at(var);
        }
        catch (const std::out_of_range& ) {
           myProject.logError("variable " + QString::fromStdString(variable) + " missing in MapDailyMeteoVar");
           continue;
        }
        variableList.addItem(QString::fromStdString(item));
    }

    QLabel variableLabel("Variable: ");
    varLayout.addWidget(&variableLabel);
    varLayout.addWidget(&variableList);

    if (myProject.clima->genericPeriodDateStart() == QDate(1800,1,1))
    {
        currentDay.setDate(myProject.getCurrentDate());
    }
    else
    {
        currentDay.setDate(myProject.clima->genericPeriodDateStart());
    }

    currentDay.setDisplayFormat("dd/MM");
    currentDayLabel.setBuddy(&currentDay);

    if(saveClima)
    {
        currentDayLabel.setVisible(false);
        currentDay.setVisible(false);
    }
    else
    {
        currentDayLabel.setText("Day/Month:");
        currentDayLabel.setVisible(true);
        currentDay.setVisible(true);
    }


    int currentYear = myProject.getCurrentDate().year();

    QLabel firstDateLabel("Start Year:");
    if (myProject.clima->yearStart() == NODATA)
    {
        firstYearEdit.setText(QString::number(currentYear));
    }
    else
    {
        firstYearEdit.setText(QString::number(myProject.clima->yearStart()));
    }

    firstYearEdit.setFixedWidth(110);
    firstYearEdit.setValidator(new QIntValidator(1800, 3000));
    firstDateLabel.setBuddy(&firstYearEdit);

    QLabel lastDateLabel("End Year:");
    if (myProject.clima->yearEnd() == NODATA)
    {
        lastYearEdit.setText(QString::number(currentYear));
    }
    else
    {
        lastYearEdit.setText(QString::number(myProject.clima->yearEnd()));
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

    QLabel periodTypeLabel("Period Type: ");
    periodLayout.addWidget(&periodTypeLabel);
    periodLayout.addWidget(&periodTypeList);

    dailyCumulated.setText("Daily cumulated");
    dailyCumulated.setChecked(false);
    QString periodSelected = periodTypeList.currentText();
    periodLayout.addWidget(&dailyCumulated);

    int dayOfYear = currentDay.date().dayOfYear();
    periodDisplay.setText("Day Of Year: " + QString::number(dayOfYear));
    periodDisplay.setReadOnly(true);

    if (saveClima)
    {
        periodDisplay.setVisible(false);
    }

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

    elaborationLayout.addWidget(new QLabel("Elaboration: "));
    QString value = variableList.currentText();
    meteoVariable key = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, value.toStdString());
    std::string keyString = getKeyStringMeteoMap(MapDailyMeteoVar, key);
    QString group = QString::fromStdString(keyString)+"_VarToElab1";
    settings->beginGroup(group);
    int size = settings->beginReadArray(QString::fromStdString(keyString));
    for (int i = 0; i < size; ++i) {
        settings->setArrayIndex(i);
        QString elab = settings->value("elab").toString();
        elaborationList.addItem( elab );
    }
    elaborationList.addItem("No elaboration available");
    if (periodSelected == "Daily")
    {
        elaborationList.setCurrentText("No elaboration available");
        elaborationList.setEnabled(false);
        dailyCumulated.setVisible(true);
    }
    else
    {
        elaborationList.setEnabled(true);
        dailyCumulated.setVisible(false);
    }
    settings->endArray();
    settings->endGroup();
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
    secondElabLayout.addWidget(new QLabel("Secondary Elaboration: "));

    if (firstYearEdit.text().toInt() == lastYearEdit.text().toInt())
    {
        secondElabList.addItem("No elaboration available");
    }
    else
    {
        group = elab1Field +"_Elab1Elab2";
        settings->beginGroup(group);
        secondElabList.addItem("None");
        size = settings->beginReadArray(elab1Field);
        for (int i = 0; i < size; ++i) {
            settings->setArrayIndex(i);
            QString elab2 = settings->value("elab2").toString();
            secondElabList.addItem( elab2 );
        }
        settings->endArray();
        settings->endGroup();
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


    if (isAnomaly)
    {
        if (pointsButton.isChecked())
        {
            isMeteoGrid = false;
            myProject.referenceClima->setDb(myProject.meteoPointsDbHandler->getDb());
        }
        else if (gridButton.isChecked())
        {
            isMeteoGrid = true;
            myProject.referenceClima->setDb(myProject.meteoGridDbHandler->db());
        }

        anomaly.AnomalySetVariableElab(variableList.currentText());
        anomaly.build(settings);

        anomalyLabel.setAlignment(Qt::AlignCenter);
        anomalyMainLayout.addWidget(&anomalyLine);
        anomalyMainLayout.setSpacing(5);
        anomalyMainLayout.addWidget(&anomalyLabel);
        anomalyMainLayout.addWidget(&copyData);
        anomalyMainLayout.setSpacing(15);
        anomalyMainLayout.addWidget(&anomaly);
    }
    else if(saveClima)
    {
        saveClimaLayout.setList(myProject.clima->getListElab()->listClimateElab());
        saveClimaMainLayout.addWidget(&saveClimaLayout);
    }


    connect(&pointsButton, &QRadioButton::clicked, [=](){ targetChange(); });
    connect(&gridButton, &QRadioButton::clicked, [=](){ targetChange(); });

    connect(&firstYearEdit, &QLineEdit::editingFinished, [=](){ this->checkYears(); });
    connect(&lastYearEdit, &QLineEdit::editingFinished, [=](){ this->checkYears(); });
    connect(&currentDay, &QDateEdit::dateChanged, [=](){ this->displayPeriod(periodTypeList.currentText()); });
    connect(&periodTypeList, &QComboBox::currentTextChanged, [=](const QString &newVar){ this->displayPeriod(newVar); });

    connect(&variableList, &QComboBox::currentTextChanged, [=](const QString &newVar){ this->listElaboration(newVar); });
    connect(&elaborationList, &QComboBox::currentTextChanged, [=](const QString &newElab){ this->listSecondElab(newElab); });
    connect(&secondElabList, &QComboBox::currentTextChanged, [=](const QString &newSecElab){ this->activeSecondParameter(newSecElab); });
    connect(&readParam, &QCheckBox::stateChanged, [=](int state){ this->readParameter(state); });
    connect(&copyData, &QPushButton::clicked, [=](){ this->copyDataToAnomaly(); });


    loadXML.setText("Load from XML");
    appendXML.setText("Append to XML");
    layoutXML.addWidget(&loadXML);
    layoutXML.addWidget(&appendXML);

    connect(&loadXML, &QPushButton::clicked, [=](){ this->copyDataFromXML(); });
    connect(&appendXML, &QPushButton::clicked, [=](){ this->saveDataToXML(); });

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
    mainLayout.addLayout(&readParamLayout);
    mainLayout.addLayout(&secondElabLayout);

    if (isAnomaly)
    {
        mainLayout.addLayout(&anomalyMainLayout);
    }
    else if(saveClima)
    {
        add.setText("Add");
        del.setText("Delete");
        delAll.setText("Delete all");
        buttonLayout.addWidget(&add);
        buttonLayout.addWidget(&del);
        buttonLayout.addWidget(&delAll);

        connect(&add, &QPushButton::clicked, [=](){ this->copyDataToSaveLayout(); });
        connect(&del, &QPushButton::clicked, [=](){ saveClimaLayout.deleteRaw(); });
        connect(&delAll, &QPushButton::clicked, [=](){ saveClimaLayout.deleteAll(); });
        mainLayout.addLayout(&buttonLayout);
        mainLayout.addLayout(&saveClimaMainLayout);
    }

    mainLayout.addLayout(&layoutXML);
    mainLayout.addLayout(&layoutOk);

    setLayout(&mainLayout);

    // show stored values
    if (myProject.clima->variable() != noMeteoVar)
    {
        std::string storedVar = MapDailyMeteoVarToString.at(myProject.clima->variable());
        variableList.setCurrentText(QString::fromStdString(storedVar));
    }
    if (myProject.clima->elab1() != "")
    {
        elaborationList.setCurrentText(myProject.clima->elab1());
        if ( (myProject.clima->param1() != NODATA) && (!myProject.clima->param1IsClimate()) )
        {
            elab1Parameter.setReadOnly(false);
            elab1Parameter.setText(QString::number(myProject.clima->param1()));
        }
        else if (myProject.clima->param1IsClimate())
        {
            elab1Parameter.clear();
            elab1Parameter.setReadOnly(true);
            readParam.setChecked(true);
            climateDbElabList.setVisible(true);
            adjustSize();
            climateDbElabList.setCurrentText(myProject.clima->param1ClimateField());
        }
        else
        {
            readParam.setChecked(false);
            climateDbElabList.setVisible(false);
            adjustSize();
        }
    }
    if (myProject.clima->elab2() != "")
    {
        secondElabList.setCurrentText(myProject.clima->elab2());
        if (myProject.clima->param2() != NODATA)
        {
            elab2Parameter.setText(QString::number(myProject.clima->param2()));
        }
    }
    if (myProject.clima->periodStr() != "")
    {
            periodTypeList.setCurrentText(myProject.clima->periodStr());
            if (myProject.clima->periodStr() == "Generic")
            {
                genericPeriodStart.setDate(myProject.clima->genericPeriodDateStart());
                genericPeriodEnd.setDate(myProject.clima->genericPeriodDateEnd());
                nrYear.setText(QString::number(myProject.clima->nYears()));
            }
    }

    show();
    exec();
}


void DialogMeteoComputation::done(bool res)
{
    if(res)  // ok was pressed
    {
        if ( (!saveClima && !checkValidData()) || (saveClima && saveClimaLayout.getList().empty()) )
        {
            return;
        }
        else if (isAnomaly && !anomaly.AnomalyCheckValidData())
        {
            return;
        }
        else  // validate the data
        {
            // store elaboration values

            if (myProject.clima == nullptr)
            {
                QMessageBox::information(nullptr, "Error!", "clima is null...");
                return;
            }
            else
            {
                if (isAnomaly && myProject.referenceClima == nullptr)
                {
                    QMessageBox::information(nullptr, "Error!", "reference clima is null...");
                    return;
                }
            }
            myProject.clima->setDailyCumulated(dailyCumulated.isChecked());

            QString periodSelected = periodTypeList.currentText();
            QString value = variableList.currentText();
            meteoVariable var = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, value.toStdString());


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
            else
            {
                myProject.clima->setNYears(0);
                QDate start;
                QDate end;
                getPeriodDates(periodSelected, firstYearEdit.text().toInt(), currentDay.date(), &start, &end);
                myProject.clima->setNYears(start.year() - firstYearEdit.text().toInt());
                myProject.clima->setGenericPeriodDateStart(start);
                myProject.clima->setGenericPeriodDateEnd(end);
            }

            if (elaborationList.currentText() == "No elaboration available")
            {
                myProject.clima->setElab1("noMeteoComp");
            }
            else
            {
                myProject.clima->setElab1(elaborationList.currentText());
            }


            if (!readParam.isChecked())
            {
                myProject.clima->setParam1IsClimate(false);
                if (! elab1Parameter.text().isEmpty())
                {
                    myProject.clima->setParam1(elab1Parameter.text().toFloat());
                }
                else
                {
                    myProject.clima->setParam1(NODATA);
                }
            }
            else
            {
                int climateIndex;
                myProject.clima->setParam1IsClimate(true);
                myProject.clima->setParam1ClimateField(climateDbElabList.currentText());
                if (periodSelected == "Generic")
                {
                    climateIndex = getClimateIndexFromElab(genericPeriodStart.date(), climateDbElabList.currentText());

                }
                else
                {
                    climateIndex = getClimateIndexFromElab(currentDay.date(), climateDbElabList.currentText());
                }
                myProject.clima->setParam1ClimateIndex(climateIndex);

            }
            if (secondElabList.currentText() == "None" || secondElabList.currentText() == "No elaboration available")
            {
                myProject.clima->setElab2("");
                myProject.clima->setParam2(NODATA);
            }
            else
            {
                myProject.clima->setElab2(secondElabList.currentText());
                if (elab2Parameter.text() != "")
                {
                    myProject.clima->setParam2(elab2Parameter.text().toFloat());
                }
                else
                {
                    myProject.clima->setParam2(NODATA);
                }
            }

            // store reference data
            if (isAnomaly)
            {
                if (anomaly.AnomalyGetReadReferenceState())
                {
                    myProject.referenceClima->resetParam();
                    myProject.referenceClima->setIsClimateAnomalyFromDb(true);
                    myProject.referenceClima->setVariable(myProject.clima->variable());
                    myProject.referenceClima->setClimateElab(anomaly.AnomalyGetClimateDb());
                    int climateIndex;
                    if (periodSelected == "Generic")
                    {
                        climateIndex = getClimateIndexFromElab(genericPeriodStart.date(), anomaly.AnomalyGetClimateDb());
                    }
                    else
                    {
                        climateIndex = getClimateIndexFromElab(currentDay.date(), anomaly.AnomalyGetClimateDb());
                    }
                    myProject.referenceClima->setParam1ClimateIndex(climateIndex);
                }
                else
                {
                    myProject.referenceClima->setIsClimateAnomalyFromDb(false);
                    myProject.referenceClima->setVariable(myProject.clima->variable());
                    QString AnomalyPeriodSelected = anomaly.AnomalyGetPeriodTypeList();

                    myProject.referenceClima->setYearStart(anomaly.AnomalyGetYearStart());
                    myProject.referenceClima->setYearEnd(anomaly.AnomalyGetYearLast());
                    myProject.referenceClima->setPeriodStr(AnomalyPeriodSelected);

                    if (AnomalyPeriodSelected == "Generic")
                    {
                        myProject.referenceClima->setGenericPeriodDateStart(anomaly.AnomalyGetGenericPeriodStart());
                        myProject.referenceClima->setGenericPeriodDateEnd(anomaly.AnomalyGetGenericPeriodEnd());
                        myProject.referenceClima->setNYears(anomaly.AnomalyGetNyears());
                    }
                    else
                    {
                        myProject.referenceClima->setNYears(0);
                        QDate start;
                        QDate end;
                        getPeriodDates(AnomalyPeriodSelected, anomaly.AnomalyGetYearStart(), anomaly.AnomalyGetCurrentDay(), &start, &end);
                        myProject.referenceClima->setNYears(start.year() - anomaly.AnomalyGetYearStart());
                        myProject.referenceClima->setGenericPeriodDateStart(start);
                        myProject.referenceClima->setGenericPeriodDateEnd(end);
                    }
                    myProject.referenceClima->setElab1(anomaly.AnomalyGetElaboration());

                    if (!anomaly.AnomalyReadParamIsChecked())
                    {
                        myProject.referenceClima->setParam1IsClimate(false);
                        if (anomaly.AnomalyGetParam1() != "")
                        {
                            myProject.referenceClima->setParam1(anomaly.AnomalyGetParam1().toFloat());
                        }
                        else
                        {
                            myProject.referenceClima->setParam1(NODATA);
                        }
                    }
                    else
                    {
                        myProject.referenceClima->setParam1IsClimate(true);
                        myProject.referenceClima->setParam1ClimateField(anomaly.AnomalyGetClimateDbElab());
                        int climateIndex;
                        if (AnomalyPeriodSelected == "Generic")
                        {
                            climateIndex = getClimateIndexFromElab(anomaly.AnomalyGetGenericPeriodStart(), anomaly.AnomalyGetClimateDbElab());

                        }
                        else
                        {
                            climateIndex = getClimateIndexFromElab(anomaly.AnomalyGetCurrentDay(), anomaly.AnomalyGetClimateDbElab());
                        }
                        myProject.referenceClima->setParam1ClimateIndex(climateIndex);

                    }
                    if (anomaly.AnomalyGetSecondElaboration() == "None" || anomaly.AnomalyGetSecondElaboration() == "No elaboration available")
                    {
                        myProject.referenceClima->setElab2("");
                        myProject.referenceClima->setParam2(NODATA);
                    }
                    else
                    {
                        myProject.referenceClima->setElab2(anomaly.AnomalyGetSecondElaboration());
                        if (anomaly.AnomalyGetParam2() != "")
                        {
                            myProject.referenceClima->setParam2(anomaly.AnomalyGetParam2().toFloat());
                        }
                        else
                        {
                            myProject.referenceClima->setParam2(NODATA);
                        }
                    }
                }
            }

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


void DialogMeteoComputation::checkYears()
{
    if (firstYearEdit.text().toInt() == lastYearEdit.text().toInt())
    {
        secondElabList.clear();
        secondElabList.addItem("No elaboration available");
    }
    else
    {
        listSecondElab(elaborationList.currentText());
    }
}


void DialogMeteoComputation::displayPeriod(const QString value)
{
    if (value == "Daily")
    {
        elaborationList.setCurrentText("No elaboration available");
        elaborationList.setEnabled(false);
        dailyCumulated.setVisible(true);
        if (saveClima)
        {
            periodDisplay.setVisible(false);
            currentDayLabel.setVisible(false);
            currentDay.setVisible(false);
            genericStartLabel.setVisible(false);
            genericEndLabel.setVisible(false);
            genericPeriodStart.setVisible(false);
            genericPeriodEnd.setVisible(false);
            nrYearLabel.setVisible(false);
            nrYear.setVisible(false);
            return;
        }
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
        elaborationList.setEnabled(true);
        dailyCumulated.setVisible(false);
        if (saveClima)
        {
            periodDisplay.setVisible(false);
            currentDayLabel.setVisible(false);
            currentDay.setVisible(false);
            genericStartLabel.setVisible(false);
            genericEndLabel.setVisible(false);
            genericPeriodStart.setVisible(false);
            genericPeriodEnd.setVisible(false);
            nrYearLabel.setVisible(false);
            nrYear.setVisible(false);
            return;
        }
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
        elaborationList.setEnabled(true);
        dailyCumulated.setVisible(false);
        if (saveClima)
        {
            periodDisplay.setVisible(false);
            currentDayLabel.setVisible(false);
            currentDay.setVisible(false);
            genericStartLabel.setVisible(false);
            genericEndLabel.setVisible(false);
            genericPeriodStart.setVisible(false);
            genericPeriodEnd.setVisible(false);
            nrYearLabel.setVisible(false);
            nrYear.setVisible(false);
            return;
        }
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
        elaborationList.setEnabled(true);
        dailyCumulated.setVisible(false);
        if (saveClima)
        {
            periodDisplay.setVisible(false);
            currentDayLabel.setVisible(false);
            currentDay.setVisible(false);
            genericStartLabel.setVisible(false);
            genericEndLabel.setVisible(false);
            genericPeriodStart.setVisible(false);
            genericPeriodEnd.setVisible(false);
            nrYearLabel.setVisible(false);
            nrYear.setVisible(false);
            return;
        }
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
        elaborationList.setEnabled(true);
        dailyCumulated.setVisible(false);
        if (saveClima)
        {
            periodDisplay.setVisible(false);
            currentDayLabel.setVisible(false);
            currentDay.setVisible(false);
            genericStartLabel.setVisible(false);
            genericEndLabel.setVisible(false);
            genericPeriodStart.setVisible(false);
            genericPeriodEnd.setVisible(false);
            nrYearLabel.setVisible(false);
            nrYear.setVisible(false);
            return;
        }
        periodDisplay.setVisible(false);
        currentDayLabel.setVisible(false);
        currentDay.setVisible(false);
        genericStartLabel.setVisible(false);
        genericEndLabel.setVisible(false);
        genericPeriodStart.setVisible(false);
        genericPeriodEnd.setVisible(false);
        nrYearLabel.setVisible(false);
        nrYear.setVisible(false);
    }
    else if (value == "Generic")
    {
        elaborationList.setEnabled(true);
        dailyCumulated.setVisible(false);
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


void DialogMeteoComputation::listElaboration(const QString value)
{

    elaborationList.blockSignals(true);
    meteoVariable key = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, value.toStdString());
    std::string keyString = getKeyStringMeteoMap(MapDailyMeteoVar, key);
    QString group = QString::fromStdString(keyString)+"_VarToElab1";
    settings->beginGroup(group);
    int size = settings->beginReadArray(QString::fromStdString(keyString));
    QString prevElab = elaborationList.currentText();
    bool existsPrevElab = false;
    elaborationList.clear();

    for (int i = 0; i < size; ++i)
    {
        settings->setArrayIndex(i);
        QString elab = settings->value("elab").toString();
        if (prevElab == elab)
        {
            existsPrevElab = true;
        }
        elaborationList.addItem( elab );

    }
    elaborationList.addItem("No elaboration available");
    if (periodTypeList.currentText() == "Daily")
    {
        elaborationList.setCurrentText("No elaboration available");
        elaborationList.setEnabled(false);
    }
    else
    {
        elaborationList.setEnabled(true);
    }

    settings->endArray();
    settings->endGroup();

    readParam.setChecked(false);
    climateDbElabList.setVisible(false);
    adjustSize();

    elaborationList.blockSignals(false);
    if (existsPrevElab)
    {
        elaborationList.setCurrentText(prevElab);
    }

    if(isAnomaly)
    {
        anomaly.AnomalySetVariableElab(variableList.currentText());
    }
}


void DialogMeteoComputation::listSecondElab(const QString value)
{

    QString elabValue;
    if (value == "No elaboration available")
    {
        elabValue = "noMeteoComp";
    }
    else
    {
        elabValue = value;
    }
    QString prevSecondElab = secondElabList.currentText();
    bool existsPrevSecondElab = false;

    if ( MapElabWithParam.find(elabValue.toStdString()) == MapElabWithParam.end())
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
        displayPeriod(periodTypeList.currentText());
    }
    else
    {
        periodTypeList.setEnabled(true);
        nrYear.setEnabled(true);
    }

    QString group = elabValue + "_Elab1Elab2";
    settings->beginGroup(group);
    int size = settings->beginReadArray(elabValue);

    if (size == 0 || firstYearEdit.text().toInt() == lastYearEdit.text().toInt())
    {
        secondElabList.clear();
        secondElabList.addItem("No elaboration available");
        settings->endArray();
        settings->endGroup();
        return;
    }
    secondElabList.clear();
    secondElabList.addItem("None");
    for (int i = 0; i < size; ++i)
    {
        settings->setArrayIndex(i);
        QString elab2 = settings->value("elab2").toString();
        if (prevSecondElab == elab2)
        {
            existsPrevSecondElab = true;
        }
        secondElabList.addItem( elab2 );
    }
    if (existsPrevSecondElab)
    {
        secondElabList.setCurrentText(prevSecondElab);
    }
    settings->endArray();
    settings->endGroup();
}


void DialogMeteoComputation::activeSecondParameter(const QString value)
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


void DialogMeteoComputation::readParameter(int state)
{

    climateDbElabList.clear();
    climateDbElab.clear();

    QString myError = myProject.errorString;

    if (state!= 0)
    {
        climateDbElabList.setVisible(true);
        adjustSize();
        QList<QString> climateTables;
        if (! getClimateTables(myProject.clima->db(), &myError, &climateTables) )
        {
            climateDbElabList.addItem("No saved elaborations found");
        }
        else
        {
            for (int i=0; i < climateTables.size(); i++)
            {
                selectVarElab(myProject.clima->db(), &myError, climateTables[i], variableList.currentText(), &climateDbElab);
            }
            if (climateDbElab.isEmpty())
            {
                climateDbElabList.addItem("No saved elaborations found");
            }
            else
            {
                for (int i=0; i < climateDbElab.size(); i++)
                {
                    climateDbElabList.addItem(climateDbElab[i]);
                }
            }
        }

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


void DialogMeteoComputation::copyDataToAnomaly()
{

    if (!anomaly.AnomalyGetReadReferenceState())
    {
        if (firstYearEdit.text().size() == 4 && lastYearEdit.text().size() == 4 && firstYearEdit.text().toInt() <= lastYearEdit.text().toInt())
        {
            anomaly.AnomalySetYearStart(firstYearEdit.text());
            anomaly.AnomalySetYearLast(lastYearEdit.text());
        }

        anomaly.AnomalySetPeriodTypeList(periodTypeList.currentText());
        if (periodTypeList.currentText() == "Generic")
        {
            anomaly.AnomalySetGenericPeriodStart(genericPeriodStart.date());
            anomaly.AnomalySetGenericPeriodEnd(genericPeriodEnd.date());
            anomaly.AnomalySetNyears(nrYear.text());
        }
        else
        {
            anomaly.AnomalySetCurrentDay(currentDay.date());
        }
        anomaly.AnomalySetElaboration(elaborationList.currentText());
        anomaly.AnomalySetReadParamIsChecked(readParam.isChecked());
        if (readParam.isChecked() == false)
        {
            anomaly.AnomalySetParam1ReadOnly(false);
            anomaly.AnomalySetParam1(elab1Parameter.text());
        }
        else
        {
            anomaly.AnomalySetClimateDbElab(climateDbElabList.currentText());
            anomaly.AnomalySetParam1ReadOnly(true);
        }

        anomaly.AnomalySetSecondElaboration(secondElabList.currentText());
        anomaly.AnomalySetParam2(elab2Parameter.text());
    }

}

void DialogMeteoComputation::copyDataToSaveLayout()
{

    if (!checkValidData())
    {
        return;
    }
    saveClimaLayout.setFirstYear(firstYearEdit.text());
    saveClimaLayout.setLastYear(lastYearEdit.text());
    QString variable = variableList.currentText();
    if (periodTypeList.currentText() == "Daily" && dailyCumulated.isChecked())
    {
        variable = variable+"CUMULATED";
    }
    saveClimaLayout.setVariable(variable);
    saveClimaLayout.setPeriod(periodTypeList.currentText());
    if (periodTypeList.currentText() == "Generic")
    {
        if (genericPeriodStart.date().day() < 10)
        {
            saveClimaLayout.setGenericPeriodStartDay("0"+QString::number(genericPeriodStart.date().day()));
        }
        else
        {
            saveClimaLayout.setGenericPeriodStartDay(QString::number(genericPeriodStart.date().day()));
        }
        if (genericPeriodStart.date().month() < 10)
        {
            saveClimaLayout.setGenericPeriodStartMonth("0"+QString::number(genericPeriodStart.date().month()));
        }
        else
        {
            saveClimaLayout.setGenericPeriodStartMonth(QString::number(genericPeriodStart.date().month()));
        }

        if (genericPeriodEnd.date().day() < 10)
        {
            saveClimaLayout.setGenericPeriodEndDay("0"+QString::number(genericPeriodEnd.date().day()));
        }
        else
        {
            saveClimaLayout.setGenericPeriodEndDay(QString::number(genericPeriodEnd.date().day()));
        }
        if (genericPeriodEnd.date().month() < 10)
        {
            saveClimaLayout.setGenericPeriodEndMonth("0"+QString::number(genericPeriodEnd.date().month()));
        }
        else
        {
            saveClimaLayout.setGenericPeriodEndMonth(QString::number(genericPeriodEnd.date().month()));
        }

        if (nrYear.text().toInt() < 10)
        {
            saveClimaLayout.setGenericNYear("0"+nrYear.text());
        }
        else
        {
            saveClimaLayout.setGenericNYear(nrYear.text());
        }

    }
    saveClimaLayout.setSecondElab(secondElabList.currentText());
    saveClimaLayout.setElab2Param(elab2Parameter.text());
    if (elaborationList.currentText() == "No elaboration available")
    {
        saveClimaLayout.setElab("noMeteoComp");
    }
    else
    {
        saveClimaLayout.setElab(elaborationList.currentText());
    }
    saveClimaLayout.setElab1Param(elab1Parameter.text());

    if (readParam.isChecked())
    {
        saveClimaLayout.setElab1ParamFromdB(climateDbElabList.currentText());
    }
    else
    {
        saveClimaLayout.setElab1ParamFromdB("");
    }

    saveClimaLayout.addElab();

}

bool DialogMeteoComputation::checkValidData()
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
    if (elaborationList.currentText().toStdString() == "huglin" || elaborationList.currentText().toStdString() == "winkler" || elaborationList.currentText().toStdString() == "fregoni")
    {
        if (secondElabList.currentText().toStdString() == "None")
        {
            QMessageBox::information(nullptr, "Second Elaboration missing", elaborationList.currentText() + " requires second elaboration");
            return false;
        }

    }
    if ( MapElabWithParam.find(elaborationList.currentText().toStdString()) != MapElabWithParam.end())
    {
        if ( (!readParam.isChecked() && elab1Parameter.text().isEmpty()) || (readParam.isChecked() && climateDbElabList.currentText() == "No saved elaborations found" ))
        {
            QMessageBox::information(nullptr, "Missing Parameter", "insert parameter");
            return false;
        }
    }
    if ( MapElabWithParam.find(secondElabList.currentText().toStdString()) != MapElabWithParam.end())
    {
        if (elab2Parameter.text().isEmpty())
        {
            QMessageBox::information(nullptr, "Missing Parameter", "insert second elaboration parameter");
            return false;
        }
    }
    if (periodTypeList.currentText() == "Generic")
    {
        if (nrYear.text().isEmpty())
        {
            QMessageBox::information(nullptr, "Missing Parameter", "insert Nr Years");
            return false;
        }
    }
    return true;
}


QList<QString> DialogMeteoComputation::getElabSaveList()
{
    return saveClimaLayout.getList();
}


void DialogMeteoComputation::copyDataFromXML()
{
    Crit3DElabList listXMLElab;
    Crit3DAnomalyList listXMLAnomaly;
    Crit3DDroughtList listXMLDrought;
    Crit3DPhenologyList listXMLPhenology;

    QString *myError = new QString();
    QString xmlName = QFileDialog::getOpenFileName(this, tr("Open XML"), "", tr("xml files (*.xml)"));
    if (xmlName != "")
    {
        if (!parseXMLElaboration(&listXMLElab, &listXMLAnomaly, &listXMLDrought, &listXMLPhenology, xmlName, myError))
        {
            QMessageBox::information(nullptr, "XML error", "Check XML");
            return;
        }
    }
    if (!isAnomaly)
    {
        if (isMeteoGrid && listXMLElab.isMeteoGrid() == false)
        {
            QMessageBox::information(nullptr, "No elaborations", "There are not Meteo Grid elaborations");
            return;
        }
        else if (!isMeteoGrid && listXMLElab.isMeteoGrid() == true)
        {
            QMessageBox::information(nullptr, "No elaborations", "There are not Meteo Points elaborations");
            return;
        }
        else
        {
            if (listXMLElab.listAll().isEmpty())
            {
                QMessageBox::information(nullptr, "Empty List", "There are not elaborations");
                return;
            }
            else
            {
                DialogXMLComputation xmlDialog(isAnomaly, listXMLElab.listAll());
                if (xmlDialog.result() == QDialog::Accepted)
                {
                    unsigned int index = xmlDialog.getIndex();
                    firstYearEdit.setText(QString::number(listXMLElab.listYearStart()[index]));
                    lastYearEdit.setText(QString::number(listXMLElab.listYearEnd()[index]));
                    std::string xmlVariable = MapDailyMeteoVarToString.at(listXMLElab.listVariable()[index]);
                    variableList.setCurrentText(QString::fromStdString(xmlVariable));
                    int elabIndex = elaborationList.findText(listXMLElab.listElab1()[index], Qt::MatchFixedString);
                    elaborationList.setCurrentIndex(elabIndex);

                    if ( (listXMLElab.listParam1()[index] != NODATA) && (!listXMLElab.listParam1IsClimate()[index]) )
                    {
                        elab1Parameter.setReadOnly(false);
                        elab1Parameter.setText(QString::number(listXMLElab.listParam1()[index]));
                    }
                    else if (listXMLElab.listParam1IsClimate()[index])
                    {
                        elab1Parameter.clear();
                        elab1Parameter.setReadOnly(true);
                        readParam.setChecked(true);
                        climateDbElabList.setVisible(true);
                        adjustSize();

                        int climateIndex = climateDbElabList.findText(listXMLElab.listParam1ClimateField()[index], Qt::MatchFixedString);
                        if (climateIndex == -1)
                        {
                            QMessageBox::information(nullptr, "climate field not found", "Check param1 climate field");
                        }
                        climateDbElabList.setCurrentIndex(climateIndex);
                    }
                    else
                    {
                        readParam.setChecked(false);
                        climateDbElabList.setVisible(false);
                        adjustSize();
                    }
                    if (listXMLElab.listElab2()[index] != "")
                    {
                        int elabIndex = secondElabList.findText(listXMLElab.listElab2()[index], Qt::MatchFixedString);
                        secondElabList.setCurrentIndex(elabIndex);
                        if ( listXMLElab.listParam2()[index] != NODATA)
                        {
                            elab2Parameter.setText(QString::number(listXMLElab.listParam2()[index]));
                        }
                    }
                    int periodIndex = periodTypeList.findText(listXMLElab.listPeriodStr()[index], Qt::MatchFixedString);
                    periodTypeList.setCurrentIndex(periodIndex);
                    if (listXMLElab.listPeriodStr()[index] == "Generic")
                    {
                        genericPeriodStart.setDate(listXMLElab.listDateStart()[index]);
                        genericPeriodEnd.setDate(listXMLElab.listDateEnd()[index]);
                        nrYear.setText(QString::number(listXMLElab.listNYears()[index]));
                    }
                    else
                    {
                        currentDay.setDate(listXMLElab.listDateStart()[index]);
                    }
                }
            }
        }
    }
    else
    {
        if (isMeteoGrid && listXMLAnomaly.isMeteoGrid() == false)
        {
            QMessageBox::information(nullptr, "No anomalies", "There are not Meteo Grid anomalies");
            return;
        }
        else if (!isMeteoGrid && listXMLAnomaly.isMeteoGrid() == true)
        {
            QMessageBox::information(nullptr, "No anomalies", "There are not Meteo Points anomalies");
            return;
        }
        else
        {
            if (listXMLAnomaly.listAll().isEmpty())
            {
                QMessageBox::information(nullptr, "Empty List", "There are not anomalies");
                return;
            }
            else
            {
                DialogXMLComputation xmlDialog(isAnomaly, listXMLAnomaly.listAll());
                if (xmlDialog.result() == QDialog::Accepted)
                {
                    unsigned int index = xmlDialog.getIndex();
                    firstYearEdit.setText(QString::number(listXMLAnomaly.listYearStart()[index]));
                    lastYearEdit.setText(QString::number(listXMLAnomaly.listYearEnd()[index]));
                    std::string xmlVariable = MapDailyMeteoVarToString.at(listXMLAnomaly.listVariable()[index]);
                    variableList.setCurrentText(QString::fromStdString(xmlVariable));

                    int elabIndex = elaborationList.findText(listXMLAnomaly.listElab1()[index], Qt::MatchFixedString);
                    elaborationList.setCurrentIndex(elabIndex);

                    if ( (listXMLAnomaly.listParam1()[index] != NODATA) && (!listXMLAnomaly.listParam1IsClimate()[index]) )
                    {
                        elab1Parameter.setReadOnly(false);
                        elab1Parameter.setText(QString::number(listXMLAnomaly.listParam1()[index]));
                    }
                    else if (listXMLAnomaly.listParam1IsClimate()[index])
                    {
                        elab1Parameter.clear();
                        elab1Parameter.setReadOnly(true);
                        readParam.setChecked(true);
                        climateDbElabList.setVisible(true);
                        adjustSize();

                        int climateIndex = climateDbElabList.findText(listXMLAnomaly.listParam1ClimateField()[index], Qt::MatchFixedString);
                        if (climateIndex == -1)
                        {
                            QMessageBox::information(nullptr, "climate field not found", "Check param1 climate field");
                        }
                        climateDbElabList.setCurrentIndex(climateIndex);

                    }
                    else
                    {
                        readParam.setChecked(false);
                        climateDbElabList.setVisible(false);
                        adjustSize();
                    }
                    if (listXMLAnomaly.listElab2()[index] != "")
                    {
                        int elabIndex = secondElabList.findText(listXMLAnomaly.listElab2()[index], Qt::MatchFixedString);
                        secondElabList.setCurrentIndex(elabIndex);

                        if ( listXMLAnomaly.listParam2()[index] != NODATA)
                        {
                            elab2Parameter.setText(QString::number(listXMLAnomaly.listParam2()[index]));
                        }
                    }
                    int periodIndex = periodTypeList.findText(listXMLAnomaly.listPeriodStr()[index], Qt::MatchFixedString);
                    periodTypeList.setCurrentIndex(periodIndex);
                    if (listXMLAnomaly.listPeriodStr()[index] == "Generic")
                    {
                        genericPeriodStart.setDate(listXMLAnomaly.listDateStart()[index]);
                        genericPeriodEnd.setDate(listXMLAnomaly.listDateEnd()[index]);
                        nrYear.setText(QString::number(listXMLAnomaly.listNYears()[index]));
                    }
                    else
                    {
                        currentDay.setDate(listXMLAnomaly.listDateStart()[index]);
                    }
                    // reference clima
                    if (listXMLAnomaly.isAnomalyFromDb()[index])
                    {
                        anomaly.AnomalyReadReferenceState(1);
                        if (!anomaly.AnomalySetClimateDb(listXMLAnomaly.listAnomalyClimateField()[index]))
                        {
                            QMessageBox::information(nullptr, "climate field not found", "Check anomaly climate");
                        }
                    }
                    else
                    {
                        anomaly.AnomalyReadReferenceState(0);
                        anomaly.AnomalySetYearStart(QString::number(listXMLAnomaly.listRefYearStart()[index]));
                        anomaly.AnomalySetYearLast(QString::number(listXMLAnomaly.listRefYearEnd()[index]));
                        if (!anomaly.AnomalySetElaboration(listXMLAnomaly.listRefElab1()[index]))
                        {
                            QMessageBox::information(nullptr, "ref primary elaboration not found", "Check ref primary elaboration");
                        }
                        if ( (listXMLAnomaly.listRefParam1()[index] != NODATA) && (!listXMLAnomaly.listRefParam1IsClimate()[index]) )
                        {
                            anomaly.AnomalySetParam1ReadOnly(false);
                            anomaly.AnomalySetParam1(QString::number(listXMLAnomaly.listRefParam1()[index]));
                        }
                        else if (listXMLAnomaly.listRefParam1IsClimate()[index])
                        {
                            anomaly.AnomalyReadParameter(1);

                            int climateIndex = climateDbElabList.findText(listXMLAnomaly.listRefParam1ClimateField()[index], Qt::MatchFixedString);
                            if (climateIndex == -1)
                            {
                                QMessageBox::information(nullptr, "climate field not found", "Check reference param1 climate field");
                            }
                            climateDbElabList.setCurrentIndex(climateIndex);
                        }
                        else
                        {
                            anomaly.AnomalyReadParameter(0);
                        }
                        if (listXMLAnomaly.listRefElab2()[index] != "")
                        {
                            if (!anomaly.AnomalySetSecondElaboration(listXMLAnomaly.listRefElab2()[index]))
                            {
                                QMessageBox::information(nullptr, "ref secondary elaboration not found", "Check ref secondary elaboration");
                            }
                            if ( listXMLAnomaly.listRefParam2()[index] != NODATA)
                            {
                                anomaly.AnomalySetParam2(QString::number(listXMLAnomaly.listRefParam2()[index]));
                            }
                        }
                        anomaly.AnomalySetPeriodTypeList(listXMLAnomaly.listRefPeriodStr()[index]);
                        if (listXMLAnomaly.listRefPeriodStr()[index] == "Generic")
                        {
                            anomaly.AnomalySetGenericPeriodStart(listXMLAnomaly.listRefDateStart()[index]);
                            anomaly.AnomalySetGenericPeriodEnd(listXMLAnomaly.listRefDateEnd()[index]);
                            anomaly.AnomalySetNyears(QString::number(listXMLAnomaly.listRefNYears()[index]));
                        }
                        else
                        {
                            anomaly.AnomalySetCurrentDay(listXMLAnomaly.listRefDateStart()[index]);
                        }
                    }
                }
            }
        }
    }
}

void DialogMeteoComputation::saveDataToXML()
{
    if (!checkValidData())
    {
        return;
    }
    if (isAnomaly && !anomaly.AnomalyCheckValidData())
    {
        return;
    }
    QString xmlName = QFileDialog::getOpenFileName(this, tr("Open XML"), "", tr("xml files (*.xml)"));
    QString *myError = new QString();
    if (xmlName == "")
    {
        return;
    }
    if (!checkDataType(xmlName, isMeteoGrid, myError))
    {
        QMessageBox::information(nullptr, "Error", *myError);
        return;
    }
    if(!isAnomaly)
    {
        Crit3DElabList *listXMLElab = new Crit3DElabList();
        listXMLElab->setIsMeteoGrid(isMeteoGrid);
        listXMLElab->insertYearStart(firstYearEdit.text().toInt());
        listXMLElab->insertYearEnd(lastYearEdit.text().toInt());
        QString value = variableList.currentText();
        meteoVariable var = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, value.toStdString());
        listXMLElab->insertVariable(var);
        listXMLElab->insertElab1(elaborationList.currentText());
        if (!elab1Parameter.isReadOnly())
        {
            listXMLElab->insertParam1(elab1Parameter.text().toFloat());
            listXMLElab->insertParam1IsClimate(false);
        }
        else if (readParam.isChecked())
        {
            listXMLElab->insertParam1IsClimate(true);
            listXMLElab->insertParam1(NODATA);
            listXMLElab->insertParam1ClimateField(climateDbElabList.currentText());
        }
        else
        {
            listXMLElab->insertParam1IsClimate(false);
            listXMLElab->insertParam1(NODATA);
        }
        if (secondElabList.currentText() == "None" || secondElabList.currentText() == "No elaboration available" || secondElabList.currentText().isEmpty())
        {
            listXMLElab->insertElab2("");
            listXMLElab->insertParam2(NODATA);
        }
        else
        {
            listXMLElab->insertElab2(secondElabList.currentText());
            if (!elab2Parameter.isReadOnly())
            {
                listXMLElab->insertParam2(elab2Parameter.text().toFloat());
            }
            else
            {
                listXMLElab->insertParam2(NODATA);
            }
        }
        listXMLElab->insertPeriodStr(periodTypeList.currentText());
        if (periodTypeList.currentText() == "Generic")
        {
            listXMLElab->insertDateStart(genericPeriodStart.date());
            listXMLElab->insertDateEnd(genericPeriodEnd.date());
            listXMLElab->insertNYears(nrYear.text().toInt());
        }
        else
        {
            QDate start;
            QDate end;
            getPeriodDates(periodTypeList.currentText(), firstYearEdit.text().toInt(), currentDay.date(), &start, &end);

            listXMLElab->insertNYears(start.year() - firstYearEdit.text().toInt());
            listXMLElab->insertDateStart(start);
            listXMLElab->insertDateEnd(end);

        }
        if (appendXMLElaboration(listXMLElab, xmlName, myError))
        {
            QMessageBox::information(nullptr, "Done", "elaboration has been appended");
        }
        else
        {
            QMessageBox::information(nullptr, "Error", "append XML error");
        }
        delete listXMLElab;

    }
    else
    {
        Crit3DAnomalyList *listXMLAnomaly = new Crit3DAnomalyList();
        listXMLAnomaly->setIsMeteoGrid(isMeteoGrid);
        listXMLAnomaly->insertYearStart(firstYearEdit.text().toInt());
        listXMLAnomaly->insertYearEnd(lastYearEdit.text().toInt());
        QString value = variableList.currentText();
        meteoVariable var = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, value.toStdString());
        listXMLAnomaly->insertVariable(var);
        listXMLAnomaly->insertElab1(elaborationList.currentText());
        if (!elab1Parameter.isReadOnly())
        {
            listXMLAnomaly->insertParam1(elab1Parameter.text().toFloat());
            listXMLAnomaly->insertParam1IsClimate(false);
        }
        else if (readParam.isChecked())
        {
            listXMLAnomaly->insertParam1IsClimate(true);
            listXMLAnomaly->insertParam1(NODATA);
            listXMLAnomaly->insertParam1ClimateField(climateDbElabList.currentText());
        }
        else
        {
            listXMLAnomaly->insertParam1IsClimate(false);
            listXMLAnomaly->insertParam1(NODATA);
        }
        if (secondElabList.currentText() == "None" || secondElabList.currentText() == "No elaboration available" || secondElabList.currentText().isEmpty())
        {
            listXMLAnomaly->insertElab2("");
            listXMLAnomaly->insertParam2(NODATA);
        }
        else
        {
            listXMLAnomaly->insertElab2(secondElabList.currentText());
            if (!elab2Parameter.isReadOnly())
            {
                listXMLAnomaly->insertParam2(elab2Parameter.text().toFloat());
            }
            else
            {
                listXMLAnomaly->insertParam2(NODATA);
            }
        }
        listXMLAnomaly->insertPeriodStr(periodTypeList.currentText());
        if (periodTypeList.currentText() == "Generic")
        {
            listXMLAnomaly->insertDateStart(genericPeriodStart.date());
            listXMLAnomaly->insertDateEnd(genericPeriodEnd.date());
            listXMLAnomaly->insertNYears(nrYear.text().toInt());
        }
        else
        {
            QDate start;
            QDate end;
            getPeriodDates(periodTypeList.currentText(), firstYearEdit.text().toInt(), currentDay.date(), &start, &end);

            listXMLAnomaly->insertNYears(start.year() - firstYearEdit.text().toInt());
            listXMLAnomaly->insertDateStart(start);
            listXMLAnomaly->insertDateEnd(end);

        }
        // reference
        if (anomaly.AnomalyGetReadReferenceState())
        {
            listXMLAnomaly->insertIsAnomalyFromDb(true);
            listXMLAnomaly->insertAnomalyClimateField(anomaly.AnomalyGetClimateDb());
        }
        else
        {
            listXMLAnomaly->insertIsAnomalyFromDb(false);
            listXMLAnomaly->insertAnomalyClimateField("");
            listXMLAnomaly->insertRefYearStart(anomaly.AnomalyGetYearStart());
            listXMLAnomaly->insertRefYearEnd(anomaly.AnomalyGetYearLast());
            listXMLAnomaly->insertRefElab1(anomaly.AnomalyGetElaboration());
            QString AnomalyPeriodSelected = anomaly.AnomalyGetPeriodTypeList();
            listXMLAnomaly->insertRefPeriodStr(AnomalyPeriodSelected);
            if (AnomalyPeriodSelected == "Generic")
            {
                listXMLAnomaly->insertRefDateStart(anomaly.AnomalyGetGenericPeriodStart());
                listXMLAnomaly->insertRefDateEnd(anomaly.AnomalyGetGenericPeriodEnd());
                listXMLAnomaly->insertRefNYears(anomaly.AnomalyGetNyears());
            }
            else
            {
                QDate start;
                QDate end;
                getPeriodDates(AnomalyPeriodSelected, anomaly.AnomalyGetYearStart(), anomaly.AnomalyGetCurrentDay(), &start, &end);

                listXMLAnomaly->insertRefNYears(start.year() - anomaly.AnomalyGetYearStart());
                listXMLAnomaly->insertRefDateStart(start);
                listXMLAnomaly->insertRefDateEnd(end);

            }
            if (!anomaly.AnomalyReadParamIsChecked())
            {
                listXMLAnomaly->insertRefParam1IsClimate(false);
                if (anomaly.AnomalyGetParam1() != "")
                {
                    listXMLAnomaly->insertRefParam1(anomaly.AnomalyGetParam1().toFloat());
                }
                else
                {
                    listXMLAnomaly->insertRefParam1(NODATA);
                }
            }
            else
            {
                listXMLAnomaly->insertRefParam1IsClimate(true);
                listXMLAnomaly->insertRefParam1ClimateField(anomaly.AnomalyGetClimateDbElab());
            }
            if (anomaly.AnomalyGetSecondElaboration() == "None" || anomaly.AnomalyGetSecondElaboration() == "No elaboration available" || anomaly.AnomalyGetSecondElaboration().isEmpty())
            {
                listXMLAnomaly->insertRefElab2("");
                listXMLAnomaly->insertRefParam2(NODATA);
            }
            else
            {
                listXMLAnomaly->insertRefElab2(anomaly.AnomalyGetSecondElaboration());
                if (anomaly.AnomalyGetParam2() != "")
                {
                    listXMLAnomaly->insertRefParam2(anomaly.AnomalyGetParam2().toFloat());
                }
                else
                {
                    listXMLAnomaly->insertRefParam2(NODATA);
                }
            }

        }
        if (appendXMLAnomaly(listXMLAnomaly, xmlName, myError))
        {
            QMessageBox::information(nullptr, "Done", "anomaly has been appended");
        }
        else
        {
            QMessageBox::information(nullptr, "Error", "append XML error");
        }
        delete listXMLAnomaly;
    }



}

void DialogMeteoComputation::targetChange()
{
    if (pointsButton.isChecked())
    {
        isMeteoGrid = false;
        myProject.clima->setDb(myProject.meteoPointsDbHandler->getDb());
        if (isAnomaly)
        {
            myProject.referenceClima->setDb(myProject.meteoPointsDbHandler->getDb());
        }
    }
    else if (gridButton.isChecked())
    {
        isMeteoGrid = true;
        myProject.clima->setDb(myProject.meteoGridDbHandler->db());
        if (isAnomaly)
        {
            myProject.referenceClima->setDb(myProject.meteoGridDbHandler->db());
        }
    }
}

bool DialogMeteoComputation::getIsMeteoGrid() const
{
    return isMeteoGrid;
}

