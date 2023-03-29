#include "dialogRemoveStation.h"

DialogRemoveStation::DialogRemoveStation(QList<QString> allStations)
: allStations(allStations)
{
    setWindowTitle("Remove stations");
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QHBoxLayout *headerLayout = new QHBoxLayout;
    QHBoxLayout *stationLayout = new QHBoxLayout;

    QHBoxLayout *layoutOk = new QHBoxLayout;
    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    layoutOk->addWidget(&buttonBox);
    listAllStations = new QListWidget;
    listAllStations->setSelectionMode(QAbstractItemView::ExtendedSelection);

    QLabel *allHeader = new QLabel("All stations");

    listAllStations->addItems(allStations);
    stationLayout->addWidget(listAllStations);

    headerLayout->addWidget(allHeader);
    headerLayout->addSpacing(listAllStations->width());
    mainLayout->addLayout(headerLayout);
    mainLayout->addLayout(stationLayout);
    mainLayout->addLayout(layoutOk);
    setLayout(mainLayout);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    show();
    exec();

}

void DialogRemoveStation::done(bool res)
{
    if (res)
    {
        QList<QListWidgetItem*> selStations = listAllStations->selectedItems();
        for(int i = 0; i < selStations.count(); ++i)
        {
            QString station = selStations[i]->text();
            selectedStations.append(station);
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

QList<QString> DialogRemoveStation::getSelectedStations()
{
    return selectedStations;
}


