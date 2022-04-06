#include "dialogAddMissingStation.h"

DialogAddMissingStation::DialogAddMissingStation(QList<QString> idStations, QList<QString> nameStations)
    : idStations(idStations), nameStations(nameStations)
{
    this->setWindowTitle("Select stations to add");
    QVBoxLayout* mainLayout = new QVBoxLayout;
    this->resize(300, 350);

    QHBoxLayout *layoutOk = new QHBoxLayout;
    QHBoxLayout *datasetLayout = new QHBoxLayout;

    listStations = new QListWidget;
    listStations->setSelectionMode(QAbstractItemView::MultiSelection);
    QList<QString> completeStations;
    for (int i = 0; i < idStations.size(); i++)
    {
        QString station = "Id: " + idStations[i] + " Name: " + nameStations[i];
        completeStations.append(station);
    }
    listStations->addItems(completeStations);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    datasetLayout->addWidget(listStations);
    layoutOk->addWidget(&buttonBox);

    mainLayout->addLayout(datasetLayout);
    mainLayout->addLayout(layoutOk);

    setLayout(mainLayout);
    exec();
}

DialogAddMissingStation::~DialogAddMissingStation()
{
    close();
}

void DialogAddMissingStation::done(bool res)
{
    if (res) // ok
    {
        if (getSelectedStations().isEmpty())
        {
            QMessageBox::information(nullptr, "Missing station", "Select a station");
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

QList<QString> DialogAddMissingStation::getSelectedStations()
{
    QList<QString> idSelected;
    for(int i = 0; i < listStations->count(); ++i)
    {
        if (listStations->item(i)->isSelected())
        {
            idSelected.append(idStations[i]);
        }
    }
    return idSelected;
}

