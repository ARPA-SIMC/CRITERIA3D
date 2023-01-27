#include "dialogComputeData.h"

DialogComputeData::DialogComputeData(QDate myDateFrom, QDate myDateTo, bool isGrid, bool allPoints)
    :isGrid(isGrid)
{

    if (isGrid)
    {
        this->setWindowTitle("Compute monthly data from daily");
    }
    else
    {
        this->setWindowTitle("Compute daily data from hourly");
    }
    QVBoxLayout* mainLayout = new QVBoxLayout;
    QHBoxLayout *layoutOk = new QHBoxLayout;
    QHBoxLayout *dateLayout = new QHBoxLayout;
    QVBoxLayout *variableLayout = new QVBoxLayout;

    QLabel *subTitleLabel = new QLabel();
    if (!isGrid)
    {
        if (allPoints)
        {
            subTitleLabel->setText("All points");
        }
        else
        {
            subTitleLabel->setText("Selected points");
        }
    }

    mainLayout->addWidget(subTitleLabel);
    QLabel *dateFromLabel = new QLabel(tr("From"));
    dateLayout->addWidget(dateFromLabel);
    dateLayout->addWidget(&dateFrom);
    QLabel *dateToLabel = new QLabel(tr("To"));
    dateLayout->addWidget(dateToLabel);
    dateLayout->addWidget(&dateTo);

    dateFrom.setDate(myDateFrom);
    dateTo.setDate(myDateTo);
    dateFrom.setDisplayFormat("dd.MM.yyyy");
    dateTo.setDisplayFormat("dd.MM.yyyy");

    QLabel *variableLabel = new QLabel(tr("Variable: "));
    std::map<meteoVariable, std::string>::const_iterator it;
    listVariable.setSelectionMode(QAbstractItemView::ExtendedSelection);
    if (isGrid)
    {
        for(it = MapMonthlyMeteoVarToString.begin(); it != MapMonthlyMeteoVarToString.end(); ++it)
        {
            listVariable.addItem(QString::fromStdString(it->second));
        }
    }
    else
    {
        for(it = MapDailyMeteoVarToString.begin(); it != MapDailyMeteoVarToString.end(); ++it)
        {
            listVariable.addItem(QString::fromStdString(it->second));
        }
    }
    allVar.setText("All variables");
    variableLayout->addWidget(variableLabel);
    variableLayout->addWidget(&listVariable);
    variableLayout->addWidget(&allVar);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });
    connect(&allVar, &QCheckBox::toggled, [=](int toggled){ this->allVarClicked(toggled); });
    connect(&listVariable, &QListWidget::itemClicked, [=](QListWidgetItem* item){ this->varClicked(item); });

    layoutOk->addWidget(&buttonBox);

    mainLayout->addLayout(dateLayout);
    mainLayout->addLayout(variableLayout);
    mainLayout->addLayout(layoutOk);

    setLayout(mainLayout);
    exec();
}

DialogComputeData::~DialogComputeData()
{
    close();
}


void DialogComputeData::allVarClicked(int toggled)
{

    bool allSelected = 1;
    for (int i=0; i < listVariable.count(); i++)
    {
        if (toggled)
        {
            listVariable.item(i)->setSelected(toggled);
        }
        else
        {
            if (!listVariable.item(i)->isSelected())
            {
                allSelected = 0;
            }
        }
    }
    for (int i=0; i < listVariable.count(); i++)
    {
        if(allSelected)
        {
            listVariable.item(i)->setSelected(toggled);
        }
    }

}

void DialogComputeData::varClicked(QListWidgetItem* item)
{
    Q_UNUSED(item);
    allVar.setChecked(false);
}

void DialogComputeData::done(bool res)
{
    if (res) // ok
    {
        if (listVariable.selectedItems().isEmpty())
        {
            QMessageBox::information(nullptr, "Missing variable", "Select a variable");
            return;
        }
        if (dateFrom.date() > dateTo.date())
        {
            QMessageBox::information(nullptr, "Invalid interval", "First date should be <= last date ");
            return;
        }
        QDialog::done(QDialog::Accepted);
        return;
    }
    else    // cancel, close or exc was pressed
    {
        QDialog::done(QDialog::Rejected);
        return;
    }
}

QList <meteoVariable> DialogComputeData::getVariables() const
{
    QList <meteoVariable> selMeteoVar;
    QList<QListWidgetItem*> selVar = listVariable.selectedItems();
    for(int i = 0; i < selVar.count(); ++i)
    {
        QString var = selVar[i]->text();
        if (isGrid)
        {
            selMeteoVar.append(getKeyMeteoVarMeteoMap(MapMonthlyMeteoVarToString, var.toStdString()));
        }
        else
        {
            selMeteoVar.append(getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, var.toStdString()));
        }

    }
    return selMeteoVar;
}

QDate DialogComputeData::getDateFrom() const
{
    return dateFrom.date();
}

QDate DialogComputeData::getDateTo() const
{
    return dateTo.date();
}

