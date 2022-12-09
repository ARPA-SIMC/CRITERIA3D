#include "dialogPointDeleteData.h"

DialogPointDeleteData::DialogPointDeleteData(QDate currentdate)
{
    setWindowTitle("Delete Data");
    QVBoxLayout mainLayout;
    QHBoxLayout titleLayout;
    QHBoxLayout timeVarLayout;
    QHBoxLayout allVarLayout;
    QHBoxLayout dateLayout;
    QHBoxLayout buttonLayout;

    QLabel dailyTitle;
    dailyTitle.setText("Daily variables:");
    QLabel hourlyTitle;
    hourlyTitle.setText("Hourly variables:");
    titleLayout.addWidget(&dailyTitle);
    titleLayout.addWidget(&hourlyTitle);
    dailyVar.setSelectionMode(QAbstractItemView::MultiSelection);

    std::map<meteoVariable, std::string>::const_iterator it;
    for(it = MapDailyMeteoVarToString.begin(); it != MapDailyMeteoVarToString.end(); ++it)
    {
        dailyVar.addItem(QString::fromStdString(it->second));
    }

    timeVarLayout.addWidget(&dailyVar);

    hourlyVar.setSelectionMode(QAbstractItemView::MultiSelection);

    for(it = MapHourlyMeteoVarToString.begin(); it != MapHourlyMeteoVarToString.end(); ++it)
    {
        hourlyVar.addItem(QString::fromStdString(it->second));
    }

    timeVarLayout.addWidget(&hourlyVar);

    allDaily.setText("All daily variables");
    allHourly.setText("All hourly variables");

    allVarLayout.addWidget(&allDaily);
    allVarLayout.addWidget(&allHourly);

    connect(&allDaily, &QCheckBox::toggled, [=](int toggled){ this->allDailyVarClicked(toggled); });
    connect(&allHourly, &QCheckBox::toggled, [=](int toggled){ this->allHourlyVarClicked(toggled); });

    connect(&dailyVar, &QListWidget::itemClicked, [=](QListWidgetItem * item){ this->dailyItemClicked(item); });
    connect(&hourlyVar, &QListWidget::itemClicked, [=](QListWidgetItem * item){ this->hourlyItemClicked(item); });

    firstDateEdit.setDate(currentdate);
    firstDateEdit.setDisplayFormat("yyyy-MM-dd");
    QLabel *FirstDateLabel = new QLabel("   Start Date:");
    FirstDateLabel->setBuddy(&firstDateEdit);

    lastDateEdit.setDate(currentdate);
    lastDateEdit.setDisplayFormat("yyyy-MM-dd");
    QLabel *LastDateLabel = new QLabel("    End Date:");
    LastDateLabel->setBuddy(&lastDateEdit);

    dateLayout.addWidget(FirstDateLabel);
    dateLayout.addWidget(&firstDateEdit);

    dateLayout.addWidget(LastDateLabel);
    dateLayout.addWidget(&lastDateEdit);

    QDialogButtonBox buttonBox;
    QPushButton downloadButton("Delete Data");
    downloadButton.setCheckable(true);
    downloadButton.setAutoDefault(false);

    QPushButton cancelButton("Cancel");
    cancelButton.setCheckable(true);
    cancelButton.setAutoDefault(false);

    buttonBox.addButton(&downloadButton, QDialogButtonBox::AcceptRole);
    buttonBox.addButton(&cancelButton, QDialogButtonBox::RejectRole);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    buttonLayout.addWidget(&buttonBox);
    mainLayout.addLayout(&titleLayout);
    mainLayout.addLayout(&timeVarLayout);
    mainLayout.addLayout(&allVarLayout);
    mainLayout.addLayout(&dateLayout);
    mainLayout.addLayout(&buttonLayout);
    setLayout(&mainLayout);

    show();
    exec();
}

void DialogPointDeleteData::allDailyVarClicked(int toggled)
{

    bool allSelected = 1;
    for (int i=0; i < dailyVar.count(); i++)
    {
        if (toggled)
        {
            dailyVar.item(i)->setSelected(toggled);
        }
        else
        {
            if (!dailyVar.item(i)->isSelected())
            {
                allSelected = 0;
            }
        }
    }
    for (int i=0; i < dailyVar.count(); i++)
    {
        if(allSelected)
        {
            dailyVar.item(i)->setSelected(toggled);
        }
    }

}

void DialogPointDeleteData::allHourlyVarClicked(int toggled)
{

    bool allSelected = 1;
    for (int i=0; i < hourlyVar.count(); i++)
    {
        if (toggled)
        {
            hourlyVar.item(i)->setSelected(toggled);
        }
        else
        {
            if (!hourlyVar.item(i)->isSelected())
            {
                allSelected = 0;
            }
        }
    }
    for (int i=0; i < hourlyVar.count(); i++)
    {
        if(allSelected)
        {
            hourlyVar.item(i)->setSelected(toggled);
        }
    }

}

void DialogPointDeleteData::dailyItemClicked(QListWidgetItem * item)
{
    if (!item->isSelected())
    {
        if (allDaily.isChecked())
        {
            allDaily.setChecked(false);
        }
    }
}

void DialogPointDeleteData::hourlyItemClicked(QListWidgetItem * item)
{
    if (!item->isSelected())
    {
        if (allHourly.isChecked())
        {
            allHourly.setChecked(false);
        }
    }
}

void DialogPointDeleteData::done(bool res)
{

    if(res)  // ok was pressed
    {
        QDate firstDate = firstDateEdit.date();
        QDate lastDate = lastDateEdit.date();

        if ((! firstDate.isValid()) || (! lastDate.isValid()) || (firstDate > lastDate) )
        {
            QMessageBox::information(nullptr, "Invalid period", "Enter delete period");
            return;
        }
        else
        {
            QListWidgetItem* item = nullptr;
            for (int i = 0; i < dailyVar.count(); ++i)
            {
                item = dailyVar.item(i);
                if (item->isSelected())
                    varD.append(item->text());

            }
            for (int i = 0; i < hourlyVar.count(); ++i)
            {
                item = hourlyVar.item(i);
                if (item->isSelected())
                    varH.append(item->text());

            }
            if (varD.isEmpty() && varH.isEmpty())
            {
                 QMessageBox::information(nullptr, "Missing parameter", "Select variable");
                 return;
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

QList<meteoVariable> DialogPointDeleteData::getVarD() const
{
    QList<meteoVariable> dailyVarList;
    meteoVariable var;

    for (int i=0; i<varD.size(); i++)
    {
        var = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, varD[i].toStdString());
        dailyVarList.append(var);
    }
    return dailyVarList;
}

QList<meteoVariable> DialogPointDeleteData::getVarH() const
{
    QList<meteoVariable> hourlyVarList;
    meteoVariable var;

    for (int i=0; i<varH.size(); i++)
    {
        var = getKeyMeteoVarMeteoMap(MapHourlyMeteoVarToString, varH[i].toStdString());
        hourlyVarList.append(var);
    }
    return hourlyVarList;
}

QDate DialogPointDeleteData::getFirstDate()
{
    return firstDateEdit.date();
}

QDate DialogPointDeleteData::getLastDate()
{
    return lastDateEdit.date();
}

bool DialogPointDeleteData::getAllDailyVar()
{
    return allDaily.isChecked();
}

bool DialogPointDeleteData::getAllHourlyVar()
{
    return allHourly.isChecked();
}
