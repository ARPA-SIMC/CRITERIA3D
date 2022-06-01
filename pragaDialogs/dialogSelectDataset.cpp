#include "dialogSelectDataset.h"


DialogSelectDataset::DialogSelectDataset(QList<QString> activeDataset)
: activeDataset(activeDataset)
{

    this->setWindowTitle("Select dataset to update");
    QVBoxLayout* mainLayout = new QVBoxLayout;
    this->resize(300, 350);

    QHBoxLayout *layoutOk = new QHBoxLayout;
    QHBoxLayout *datasetLayout = new QHBoxLayout;

    listDataset = new QListWidget;
    listDataset->setSelectionMode(QAbstractItemView::MultiSelection);
    listDataset->addItems(activeDataset);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    datasetLayout->addWidget(listDataset);
    layoutOk->addWidget(&buttonBox);

    mainLayout->addLayout(datasetLayout);
    mainLayout->addLayout(layoutOk);

    setLayout(mainLayout);
    exec();
}

DialogSelectDataset::~DialogSelectDataset()
{
    close();
}

void DialogSelectDataset::done(bool res)
{
    if (res) // ok
    {
        if (getSelectedDatasets().isEmpty())
        {
            QMessageBox::information(nullptr, "Missing dataset", "Select a dataset");
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

QList<QString> DialogSelectDataset::getSelectedDatasets()
{
    QList<QString> datasetsSelected;
    for(int i = 0; i < listDataset->count(); ++i)
    {
        if (listDataset->item(i)->isSelected())
        {
            QString dataset = listDataset->item(i)->text();
            datasetsSelected.append(dataset);
        }
    }
    return datasetsSelected;
}




