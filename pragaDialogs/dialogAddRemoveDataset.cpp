#include "dialogAddRemoveDataset.h"

DialogAddRemoveDataset::DialogAddRemoveDataset(QList<QString> availableDataset, QList<QString> dbDataset)
: availableDataset(availableDataset), dbDataset(dbDataset)
{
    setWindowTitle("Add or remove dataset");
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QHBoxLayout *headerLayout = new QHBoxLayout;
    QHBoxLayout *datasetLayout = new QHBoxLayout;
    QVBoxLayout *arrowLayout = new QVBoxLayout();
    QHBoxLayout *layoutOk = new QHBoxLayout;
    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    layoutOk->addWidget(&buttonBox);
    listAvailableDataset = new QListWidget;
    listDbDataset = new QListWidget;

    QLabel *allHeader = new QLabel("Dataset available");
    QLabel *selectedHeader = new QLabel("Db dataset");
    addButton = new QPushButton(tr("➡"));
    deleteButton = new QPushButton(tr("⬅"));
    addButton->setEnabled(false);
    deleteButton->setEnabled(false);
    arrowLayout->addWidget(addButton);
    arrowLayout->addWidget(deleteButton);
    listAvailableDataset->addItems(availableDataset);
    listDbDataset->addItems(dbDataset);
    datasetLayout->addWidget(listAvailableDataset);
    datasetLayout->addLayout(arrowLayout);
    datasetLayout->addWidget(listDbDataset);

    headerLayout->addWidget(allHeader);
    headerLayout->addSpacing(listAvailableDataset->width());
    headerLayout->addWidget(selectedHeader);
    mainLayout->addLayout(headerLayout);
    mainLayout->addLayout(datasetLayout);
    mainLayout->addLayout(layoutOk);
    setLayout(mainLayout);

    connect(listAvailableDataset, &QListWidget::itemClicked, [=](QListWidgetItem* item){ this->datasetAllClicked(item); });
    connect(listDbDataset, &QListWidget::itemClicked, [=](QListWidgetItem* item){ this->datasetDbClicked(item); });
    connect(addButton, &QPushButton::clicked, [=](){ addDataset(); });
    connect(deleteButton, &QPushButton::clicked, [=](){ deleteDataset(); });
    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    show();
    exec();

}

void DialogAddRemoveDataset::datasetAllClicked(QListWidgetItem* item)
{
    Q_UNUSED(item);

    addButton->setEnabled(true);
    deleteButton->setEnabled(false);
    listDbDataset->clearSelection();
}

void DialogAddRemoveDataset::datasetDbClicked(QListWidgetItem* item)
{
    Q_UNUSED(item)

    addButton->setEnabled(false);
    deleteButton->setEnabled(true);
    listAvailableDataset->clearSelection();
}

void DialogAddRemoveDataset::addDataset()
{
    QListWidgetItem *item = listAvailableDataset->currentItem();
    int row = listAvailableDataset->currentRow();
    listAvailableDataset->takeItem(row);
    listDbDataset->addItem(item);
}

void DialogAddRemoveDataset::deleteDataset()
{
    QListWidgetItem *item = listDbDataset->currentItem();
    int row = listDbDataset->currentRow();
    listDbDataset->takeItem(row);
    listAvailableDataset->addItem(item);
}

QList<QString> DialogAddRemoveDataset::getDatasetDb()
{
    QList<QString> datasetSelected;
    for(int i = 0; i < listDbDataset->count(); ++i)
    {
        QString dataset = listDbDataset->item(i)->text();
        datasetSelected.append(dataset);
    }
    return datasetSelected;
}



