#include <QMessageBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QCheckBox>
#include <QDateEdit>
#include <QLabel>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QRadioButton>

#include "dialogExportDataGrid.h"


DialogExportDataGrid::DialogExportDataGrid()
{
    QVBoxLayout mainLayout;
    QHBoxLayout titleLayout;
    QHBoxLayout timeVarLayout;
    QHBoxLayout allVarLayout;
    QHBoxLayout dateLayout;
    QHBoxLayout buttonLayout;

    setWindowTitle("Export Grid Data");

    QLabel dailyTitle;
    dailyTitle.setText("Daily variables:");
    QLabel hourlyTitle;
    hourlyTitle.setText("Hourly variables:");
    titleLayout.addWidget(&dailyTitle);
    titleLayout.addWidget(&hourlyTitle);

    dailyVar.setSelectionMode(QAbstractItemView::MultiSelection);

    dailyItems.resize(8);
    dailyItems[0].setText("DAILY_TMIN");
    dailyItems[1].setText("DAILY_TMAX");
    dailyItems[2].setText("DAILY_TAVG");
    dailyItems[3].setText("DAILY_PREC");
    dailyItems[4].setText("DAILY_RHMIN");
    dailyItems[5].setText("DAILY_RHMAX");
    dailyItems[6].setText("DAILY_RHAVG");
    dailyItems[7].setText("DAILY_RAD");

    dailyVar.addItem(&dailyItems[0]);
    dailyVar.addItem(&dailyItems[1]);
    dailyVar.addItem(&dailyItems[2]);
    dailyVar.addItem(&dailyItems[3]);
    dailyVar.addItem(&dailyItems[4]);
    dailyVar.addItem(&dailyItems[5]);
    dailyVar.addItem(&dailyItems[6]);
    dailyVar.addItem(&dailyItems[7]);

    timeVarLayout.addWidget(&dailyVar);

    allDaily.setText("All daily variables");
    allVarLayout.addWidget(&allDaily);

    connect(&allDaily, &QCheckBox::toggled, [=](int toggled){ this->allDailyVarClicked(toggled); });
    connect(&dailyVar, &QListWidget::itemClicked, [=](QListWidgetItem * item){ this->dailyItemClicked(item); });

    // TODO manca la funzione per esportare gli orari
    hourlyVar.setSelectionMode(QAbstractItemView::MultiSelection);

    hourlyItems.resize(6);
    hourlyItems[0].setText("TAVG");
    hourlyItems[1].setText("PREC");
    hourlyItems[2].setText("RHAVG");
    hourlyItems[3].setText("RAD");
    hourlyItems[4].setText("W_SCAL_INT");
    hourlyItems[5].setText("W_VEC_DIR");

    hourlyVar.addItem(&hourlyItems[0]);
    hourlyVar.addItem(&hourlyItems[1]);
    hourlyVar.addItem(&hourlyItems[2]);
    hourlyVar.addItem(&hourlyItems[3]);
    hourlyVar.addItem(&hourlyItems[4]);
    hourlyVar.addItem(&hourlyItems[5]);

    timeVarLayout.addWidget(&hourlyVar);

    allHourly.setText("All hourly variables");
    allVarLayout.addWidget(&allHourly);

    hourlyVar.setVisible(false);
    allHourly.setVisible(false);

    //connect(&allHourly, &QCheckBox::toggled, [=](int toggled){ this->allHourlyVarClicked(toggled); });
    //connect(&hourlyVar, &QListWidget::itemClicked, [=](QListWidgetItem * item){ this->hourlyItemClicked(item); });

    QDate yesterday = QDate::currentDate().addDays(-1);

    firstDateEdit.setDate(yesterday);
    firstDateEdit.setDisplayFormat("yyyy-MM-dd");
    firstDateEdit.setCalendarPopup(true);
    QLabel *FirstDateLabel = new QLabel("   Start Date:");
    FirstDateLabel->setBuddy(&firstDateEdit);

    lastDateEdit.setDate(yesterday);
    lastDateEdit.setDisplayFormat("yyyy-MM-dd");
    lastDateEdit.setCalendarPopup(true);
    QLabel *LastDateLabel = new QLabel("    End Date:");
    LastDateLabel->setBuddy(&lastDateEdit);

    dateLayout.addWidget(FirstDateLabel);
    dateLayout.addWidget(&firstDateEdit);

    dateLayout.addWidget(LastDateLabel);
    dateLayout.addWidget(&lastDateEdit);

    QDialogButtonBox buttonBox;
    QPushButton downloadButton("Export");
    downloadButton.setCheckable(true);
    downloadButton.setAutoDefault(false);

    QPushButton cancelButton("Cancel");
    cancelButton.setCheckable(true);
    cancelButton.setAutoDefault(false);

    buttonBox.addButton(&downloadButton, QDialogButtonBox::AcceptRole);
    buttonBox.addButton(&cancelButton, QDialogButtonBox::RejectRole);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    // menu
    QMenuBar* menuBar = new QMenuBar();
    QMenu *menuFile = new QMenu("File");
    menuBar->addMenu(menuFile);

    QAction* loadCellList = new QAction(tr("&Load the list of cell IDs"), this);
    menuFile->addAction(loadCellList);
    connect(loadCellList, &QAction::triggered, this, &DialogExportDataGrid::on_actionLoadCellList);

    buttonLayout.addWidget(&buttonBox);
    mainLayout.addLayout(&titleLayout);
    mainLayout.addLayout(&timeVarLayout);
    mainLayout.addLayout(&allVarLayout);
    mainLayout.addLayout(&dateLayout);
    mainLayout.addLayout(&buttonLayout);
    mainLayout.setMenuBar(menuBar);
    setLayout(&mainLayout);

    show();
    exec();
}


void DialogExportDataGrid::on_actionLoadCellList()
{
    cellListFileName = QFileDialog::getOpenFileName(this, tr("Select list of cell IDs"), "", tr("All files (*.*)"));

    QFile listFile(cellListFileName);
    if (! listFile.open(QFile::ReadOnly | QFile::Text))
    {
        QMessageBox::information(nullptr, "Wrong file", "Wrong format: must be a text file.");
        cellListFileName = "";
    }
}


void DialogExportDataGrid::allDailyVarClicked(int toggled)
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


void DialogExportDataGrid::allHourlyVarClicked(int toggled)
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


void DialogExportDataGrid::dailyItemClicked(QListWidgetItem * item)
{
    if (! item->isSelected())
    {
        if (allDaily.isChecked())
        {
            allDaily.setChecked(false);
        }
    }
}

void DialogExportDataGrid::hourlyItemClicked(QListWidgetItem * item)
{
    if (! item->isSelected())
    {
        if (allHourly.isChecked())
        {
            allHourly.setChecked(false);
        }
    }
}


void DialogExportDataGrid::done(bool result)
{
    if (! result)
    {
        // cancel, close or exc was pressed
        QDialog::done(QDialog::Rejected);
        return;
    }

    QDate firstDate = firstDateEdit.date();
    QDate lastDate = lastDateEdit.date();
    if ((! firstDate.isValid()) || (! lastDate.isValid()))
    {
        QMessageBox::information(nullptr, "Missing parameter", "Select period.");
        return;
    }

     QListWidgetItem* item = nullptr;
     for (int i = 0; i < dailyVar.count(); ++i)
     {
            item = dailyVar.item(i);
            if (item->isSelected())
                dailyVariableList.append(item->text());

     }
     for (int i = 0; i < hourlyVar.count(); ++i)
     {
            item = hourlyVar.item(i);
            if (item->isSelected())
                hourlyVariableList.append(item->text());

     }

     if (dailyVariableList.isEmpty() && hourlyVariableList.isEmpty())
     {
         QMessageBox::information(nullptr, "Missing parameter", "Select at least a variable.");
         return;
     }

     QDialog::done(QDialog::Accepted);
}
