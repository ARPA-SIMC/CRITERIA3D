#include "dialogAddStation.h"

DialogAddStation::DialogAddStation(QList<QString> _activeStationsList)
: _activeStationsList(_activeStationsList)
{
    setWindowTitle("Add stations");
    //finestra generale

    QVBoxLayout *mainLayout = new QVBoxLayout();
    QHBoxLayout *headerLayout = new QHBoxLayout;
    QHBoxLayout *stationLayout = new QHBoxLayout;
    QHBoxLayout *singleValueLayout = new QHBoxLayout; //distanza
    QHBoxLayout *nearStationsLayout = new QHBoxLayout;
    QHBoxLayout *buttonsLayout = new QHBoxLayout;
    QHBoxLayout *addButtonLayout = new QHBoxLayout();

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    buttonsLayout->addWidget(&buttonBox);

    _listActiveStationsWidget = new QListWidget;
    _listActiveStationsWidget->setSelectionMode(QAbstractItemView::SingleSelection);

    _listNearStationsWidget = new QListWidget;

    QLabel *stationHeader = new QLabel("Active stations");
    _listActiveStationsWidget->addItems(_activeStationsList);
    stationLayout->addWidget(_listActiveStationsWidget);

    QLabel singleValueLabel("Insert distance [m]:"); //TODO check unità di misura
    singleValueLayout->addWidget(&singleValueLabel);
    singleValueLabel.setBuddy(&_singleValueEdit);

    _singleValueEdit.setValidator(new QDoubleValidator(0.0, 9999.0,1));
    _singleValueEdit.setText(QString::number(getSingleValue()));

    singleValueLayout->addWidget(&_singleValueEdit);

    QLabel nearStationsLabel("Near stations");
    nearStationsLayout->addWidget(&nearStationsLabel);
    _listNearStationsWidget->addItems(_nearStationsList);
    nearStationsLayout->addWidget(_listNearStationsWidget);

    QPushButton *_add = new QPushButton("Add station");
    addButtonLayout->addWidget(_add);
    //connect(_add, &QPushButton::clicked, [=](){ this->addStation(); });

    headerLayout->addWidget(stationHeader);
    headerLayout->addSpacing(_listActiveStationsWidget->width());
    mainLayout->addLayout(headerLayout);
    mainLayout->addLayout(stationLayout);
    mainLayout->addLayout(singleValueLayout);
    mainLayout->addLayout(buttonsLayout);
    mainLayout->addLayout(nearStationsLayout);
    mainLayout->addLayout(addButtonLayout);
    setLayout(mainLayout);

    // Bottoni ok e cancel.
    //connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->searchStations(true, _allMeteoPointsPointer); });
    //connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->searchStations(false, _allMeteoPointsPointer); });

    show();
    exec();

}

double DialogAddStation::getSingleValue()
{
    double chosenDistance = _singleValueEdit.text().toFloat();
    return chosenDistance;
}

void DialogAddStation::searchStations(bool res, Crit3DMeteoPoint* _allMeteoPointsPointer, int nrMeteoPoints)
{
    if (res) //l'utente ha inserito la distanza e cliccato su ok
    {
        QList<QListWidgetItem*> _selectedStation = _listActiveStationsWidget->selectedItems();
        for (int i=0; i == _selectedStation.count(); i++)
        {
            std::string myStation = _selectedStation[i]->text().toStdString();

            double chosenDistance = DialogAddStation::getSingleValue();
            //prende la distanza

            for (int j=0; j < nrMeteoPoints; j++)
            {
                if (myStation == _allMeteoPointsPointer[j].name)
                {
                    Crit3DMeteoPoint myStationMp = _allMeteoPointsPointer[j];
                    if (myStationMp.latitude - _allMeteoPointsPointer[j].latitude <= chosenDistance)
                    {
                        _nearStationsList.append(QString::fromStdString(_allMeteoPointsPointer[j].name)); //il vettore da riempire
                        QDialog::done(QDialog::Accepted);
                    }

                }
            }
        };
    }
    else    // cancel, close or exc was pressed
    {
        QDialog::done(QDialog::Rejected);
        return;
    }
}


