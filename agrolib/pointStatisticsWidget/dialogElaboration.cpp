#include "dialogElaboration.h"
#include "climate.h"
#include "utilities.h"


DialogElaboration::DialogElaboration(QSettings *settings, Crit3DClimate *clima, QDate firstDate, QDate lastDate)
    : settings(settings), clima(clima), firstDate(firstDate), lastDate(lastDate)
{

    setWindowTitle("Elaboration");

    QVBoxLayout mainLayout;
    QHBoxLayout varLayout;
    QHBoxLayout dateLayout;
    QHBoxLayout periodLayout;
    QHBoxLayout displayLayout;
    QHBoxLayout genericPeriodLayout;
    QHBoxLayout layoutOk;
    QHBoxLayout elaborationLayout;
    meteoVariable var;

    std::string lastVariableUsed = "";
    Q_FOREACH (QString group, settings->childGroups())
    {
        if (!group.endsWith("_VarToElab1"))
            continue;
        std::string item;
        std::string variable = group.left(group.size()-11).toStdString(); // remove "_VarToElab1"
        try {
          var = MapDailyMeteoVar.at(variable);
          item = MapDailyMeteoVarToString.at(var);
          if (clima->variable() == var)
          {
            lastVariableUsed = item;
          }
        }
        catch (const std::out_of_range& ) {
           continue;
        }
        variableList.addItem(QString::fromStdString(item));
    }
    if (lastVariableUsed != "")
    {
        variableList.setCurrentText(QString::fromStdString(lastVariableUsed));
    }

    QLabel variableLabel("Variable: ");
    varLayout.addWidget(&variableLabel);
    varLayout.addWidget(&variableList);

    currentDay.setDate(firstDate);
    currentDay.setDisplayFormat("dd/MM");
    currentDayLabel.setBuddy(&currentDay);
    currentDayLabel.setText("Day/Month:");
    currentDayLabel.setVisible(true);
    currentDay.setVisible(true);

    QLabel firstDateLabel("Start Year:");
    if (clima->yearStart() != NODATA)
    {
        firstYearEdit.setText(QString::number(clima->yearStart()));
    }
    else
    {
        firstYearEdit.setText(QString::number(firstDate.year()));
    }

    firstYearEdit.setFixedWidth(110);
    firstYearEdit.setValidator(new QIntValidator(1800, 3000));
    firstDateLabel.setBuddy(&firstYearEdit);

    QLabel lastDateLabel("End Year:");
    if (clima->yearEnd() != NODATA)
    {
        lastYearEdit.setText(QString::number(clima->yearEnd()));
    }
    else
    {
        lastYearEdit.setText(QString::number(lastDate.year()));
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

    if (!clima->periodStr().isEmpty())
    {
        periodTypeList.setCurrentText(clima->periodStr());
        if (clima->periodStr() == "Generic")
        {
            genericPeriodStart.setDate(clima->genericPeriodDateStart());
            genericPeriodEnd.setDate(clima->genericPeriodDateEnd());
            nrYear.setText(QString::number(clima->nYears()));
            periodDisplay.setVisible(false);
            currentDayLabel.setVisible(false);
            currentDay.setVisible(false);
            genericStartLabel.setVisible(true);
            genericEndLabel.setVisible(true);
            genericPeriodStart.setVisible(true);
            genericPeriodEnd.setVisible(true);
            nrYearLabel.setVisible(true);
            nrYear.setVisible(true);
        }
        else
        {
            genericStartLabel.setVisible(false);
            genericEndLabel.setVisible(false);
            genericPeriodStart.setVisible(false);
            genericPeriodEnd.setVisible(false);
            nrYearLabel.setVisible(false);
            nrYear.setVisible(false);
        }
    }

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
    if (!clima->elab1().isEmpty())
    {
        elaborationList.setCurrentText(clima->elab1());
    }
    settings->endArray();
    settings->endGroup();
    elaborationLayout.addWidget(&elaborationList);

    elab1Parameter.setPlaceholderText("Parameter");
    elab1Parameter.setFixedWidth(90);
    elab1Parameter.setValidator(new QDoubleValidator(-9999.0, 9999.0, 2));

    QString elab1Field = elaborationList.currentText();
    if ( MapElabWithParam.find(elab1Field.toStdString()) == MapElabWithParam.end())
    {
        elab1Parameter.clear();
        elab1Parameter.setReadOnly(true);
        adjustSize();
    }
    else
    {
        elab1Parameter.setReadOnly(false);
        if (clima->param1() != NODATA)
        {
            elab1Parameter.setText(QString::number(clima->param1()));
        }
    }

    elaborationLayout.addWidget(&elab1Parameter);

    connect(&currentDay, &QDateEdit::dateChanged, [=](){ this->displayPeriod(periodTypeList.currentText()); });
    connect(&periodTypeList, &QComboBox::currentTextChanged, [=](const QString &newVar){ this->displayPeriod(newVar); });
    connect(&variableList, &QComboBox::currentTextChanged, [=](const QString &newVar){ this->listElaboration(newVar); });
    connect(&elaborationList, &QComboBox::currentTextChanged, [=](const QString &newElab){ this->changeElab(newElab); });

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    layoutOk.addWidget(&buttonBox);

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


void DialogElaboration::done(bool res)
{

    if(res)  // ok was pressed
    {
        if ( !checkValidData() )
        {
            return;
        }
        else  // validate the data
        {
            // store elaboration values

            QString periodSelected = periodTypeList.currentText();
            QString value = variableList.currentText();
            meteoVariable var = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, value.toStdString());


            clima->setVariable(var);
            clima->setYearStart(firstYearEdit.text().toInt());
            clima->setYearEnd(lastYearEdit.text().toInt());
            clima->setPeriodStr(periodSelected);
            if (periodSelected == "Generic")
            {
                clima->setGenericPeriodDateStart(genericPeriodStart.date());
                clima->setGenericPeriodDateEnd(genericPeriodEnd.date());
                clima->setNYears(nrYear.text().toInt());
            }
            else
            {
                clima->setNYears(0);
                QDate start;
                QDate end;
                getPeriodDates(periodSelected, firstYearEdit.text().toInt(), currentDay.date(), &start, &end);
                clima->setNYears(start.year() - firstYearEdit.text().toInt());
                clima->setGenericPeriodDateStart(start);
                clima->setGenericPeriodDateEnd(end);
            }

            clima->setElab1(elaborationList.currentText());

            clima->setParam1IsClimate(false);
            if (! elab1Parameter.text().isEmpty())
            {
                clima->setParam1(elab1Parameter.text().toFloat());
            }
            else
            {
                clima->setParam1(NODATA);
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

void DialogElaboration::displayPeriod(const QString value)
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

void DialogElaboration::listElaboration(const QString value)
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
    settings->endArray();
    settings->endGroup();
    adjustSize();

    elaborationList.blockSignals(false);
    if (existsPrevElab)
    {
        elaborationList.setCurrentText(prevElab);
    }
}

bool DialogElaboration::checkValidData()
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
    if ( MapElabWithParam.find(elaborationList.currentText().toStdString()) != MapElabWithParam.end())
    {
        if (elab1Parameter.text().isEmpty())
        {
            QMessageBox::information(nullptr, "Missing Parameter", "insert parameter");
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

void DialogElaboration::changeElab(const QString value)
{

    if ( MapElabWithParam.find(value.toStdString()) == MapElabWithParam.end())
    {
        elab1Parameter.clear();
        elab1Parameter.setReadOnly(true);
        adjustSize();
    }
    else
    {
        elab1Parameter.setReadOnly(false);
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

}

