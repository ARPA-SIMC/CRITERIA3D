#include "dialogPointProperties.h"

DialogPointProperties::DialogPointProperties(QList<QString> pragaProperties, QList<QString> CSVFields)
    :pragaProperties(pragaProperties), CSVFields(CSVFields)
{ 
    setWindowTitle("Point Properties");
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QHBoxLayout *headerLayout = new QHBoxLayout;
    QHBoxLayout *listLayout = new QHBoxLayout;
    QVBoxLayout *displayLayout = new QVBoxLayout;

    QHBoxLayout *layoutOk = new QHBoxLayout;
    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    layoutOk->addWidget(&buttonBox);
    propertiesList = new QListWidget;
    csvList = new QListWidget;
    joinedList = new QListWidget;

    QLabel *internal = new QLabel("Internal properties");
    QLabel *csv = new QLabel("csv properties");

    joinButton = new QPushButton(tr("Join"));
    deleteButton = new QPushButton(tr("Delete"));
    joinButton->setEnabled(false);
    deleteButton->setEnabled(false);

    propertiesList->addItems(pragaProperties);
    csvList->addItems(CSVFields);
    listLayout->addWidget(propertiesList);
    listLayout->addWidget(joinButton);
    listLayout->addWidget(csvList);

    displayLayout->addWidget(joinedList);
    displayLayout->addWidget(deleteButton);

    headerLayout->addWidget(internal, Qt::AlignCenter);
    headerLayout->addSpacing(propertiesList->width());
    headerLayout->addWidget(csv, Qt::AlignCenter);

    mainLayout->addLayout(headerLayout);
    mainLayout->addLayout(listLayout);
    mainLayout->addLayout(displayLayout);
    mainLayout->addLayout(layoutOk);
    setLayout(mainLayout);

    connect(propertiesList, &QListWidget::itemClicked, [=](QListWidgetItem* item){ this->propertiesClicked(item); });
    connect(csvList, &QListWidget::itemClicked, [=](QListWidgetItem* item){ this->csvClicked(item); });
    connect(joinedList, &QListWidget::itemClicked, [=](QListWidgetItem* item){ this->joinedClicked(item); });
    connect(joinButton, &QPushButton::clicked, [=](){ addCouple(); });
    connect(deleteButton, &QPushButton::clicked, [=](){ deleteCouple(); });
    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });


    show();
    exec();

}

void DialogPointProperties::propertiesClicked(QListWidgetItem* item)
{
    propertiesSelected = item->text();
    if (!csvSelected.isEmpty())
    {
        joinButton->setEnabled(true);
    }
}

void DialogPointProperties::csvClicked(QListWidgetItem* item)
{
    csvSelected = item->text();
    if (!propertiesSelected.isEmpty())
    {
        joinButton->setEnabled(true);
    }
}

void DialogPointProperties::joinedClicked(QListWidgetItem* item)
{
    Q_UNUSED(item)
    deleteButton->setEnabled(true);
}

void DialogPointProperties::addCouple()
{
    if (propertiesSelected == "" || csvSelected == "") return;

    joinedList->addItem(propertiesSelected+"-->"+csvSelected);
    QList<QListWidgetItem *> item = propertiesList->findItems(propertiesSelected, Qt::MatchExactly);
    if (!item.isEmpty())
    {
        item.at(0)->setHidden(true);
    }
    item = csvList->findItems(csvSelected, Qt::MatchExactly);
    if (!item.isEmpty())
    {
        item.at(0)->setHidden(true);
    }
    propertiesSelected.clear();
    csvSelected.clear();
}

void DialogPointProperties::deleteCouple()
{
    QList<QListWidgetItem *> itemList = joinedList->selectedItems();

    for (int i = 0; i<itemList.size(); i++)
    {
        QString selected = itemList[i]->text();
        QList<QString> splitItemList = selected.split("-->");
        QList<QListWidgetItem *> item = propertiesList->findItems(splitItemList[0], Qt::MatchExactly);
        if (!item.isEmpty())
        {
            item.at(0)->setHidden(false);
        }
        item = csvList->findItems(splitItemList[1], Qt::MatchExactly);
        if (!item.isEmpty())
        {
            item.at(0)->setHidden(false);
        }
        joinedList->takeItem(joinedList->row(itemList[i]));
    }
    if (joinedList->count() == 0)
    {
        deleteButton->setEnabled(false);
    }

}

QList<QString> DialogPointProperties::getJoinedList()
{
    QList<QString> joinedFields;
    for(int i = 0; i < joinedList->count(); ++i)
    {
        QString var = joinedList->item(i)->text();
        joinedFields.append(var);
    }
    return joinedFields;
}

void DialogPointProperties::done(bool res)
{
    if (res) // ok
    {
        QList<QString> unusedProperties;
        for(int i = 0; i < propertiesList->count(); ++i)
        {
            if (!propertiesList->item(i)->isHidden())
            {
                QString var = propertiesList->item(i)->text();
                unusedProperties.append(var);
            }
        }
        if (unusedProperties.contains("id_point"))
        {
            QMessageBox::information(nullptr, "Missing id_point", "Join id_point fields");
            return;
        }
        if (unusedProperties.contains("name"))
        {
            QMessageBox::information(nullptr, "Missing name", "Join name fields");
            return;
        }
        if (unusedProperties.contains("altitude"))
        {
            QMessageBox::information(nullptr, "Missing altitude", "Join altitude fields");
            return;
        }
        if (unusedProperties.contains("latitude") || unusedProperties.contains("longitude"))
        {
            if (unusedProperties.contains("utm_x") || unusedProperties.contains("utm_y"))
            {
                QMessageBox::information(nullptr, "", "Missing geographical coordinates");
                return;
            }
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




