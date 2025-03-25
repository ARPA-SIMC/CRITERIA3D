#include "dialogAddStation.h"
#include "gis.h"

DialogAddStation::DialogAddStation(QList<QString> activeStationsList, Crit3DMeteoPoint* allMeteoPointsPointer, QVector<Crit3DMeteoPoint> _meteoPoints)
    : _activeStationsList(activeStationsList), _allMeteoPointsPointer(allMeteoPointsPointer), _meteoPoints(_meteoPoints)
{
    setWindowTitle("Add stations");
    //finestra generale

    QVBoxLayout *mainLayout = new QVBoxLayout();
    QHBoxLayout *headerLayout = new QHBoxLayout;
    QHBoxLayout *stationLayout = new QHBoxLayout;
    QHBoxLayout *singleValueLayout = new QHBoxLayout; //distanza
    QHBoxLayout *nearStationsLayout = new QHBoxLayout;
    QHBoxLayout *searchButtonLayout = new QHBoxLayout;
    QHBoxLayout *buttonsLayout = new QHBoxLayout;


    QPushButton *_search = new QPushButton("Search stations");
    searchButtonLayout->addWidget(_search);
    connect(_search, &QPushButton::clicked, [=](){ this->searchStations(); });

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    buttonsLayout->addWidget(&buttonBox);

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

    headerLayout->addWidget(stationHeader);
    headerLayout->addSpacing(_listActiveStationsWidget->width());
    mainLayout->addLayout(headerLayout);
    mainLayout->addLayout(stationLayout);
    mainLayout->addLayout(singleValueLayout);
    mainLayout->addLayout(searchButtonLayout);
    mainLayout->addLayout(nearStationsLayout);
    mainLayout->addLayout(buttonsLayout);
    setLayout(mainLayout);

    // Bottoni ok e cancel.
    //connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->addStation(true); });
    //connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->addStation(false); });
    show();
    exec();

}

double DialogAddStation::getSingleValue()
{
    bool isNumber = false;
    double chosenDistance = _singleValueEdit.text().toFloat(&isNumber);
    if (isNumber)
    {
        if (chosenDistance > 0)
        {
            return chosenDistance;
        }
    }
    return NODATA;
}

void DialogAddStation::searchStations()
{
    std::string myStation = _listActiveStationsWidget->currentText().toStdString();
    double chosenDistance = DialogAddStation::getSingleValue();

    if (chosenDistance == NODATA)
    {
        QMessageBox::warning(this, "Warning!", "Wrong value: distance must be a positive number.");
        return;
    }

    for (int i=0; i < _nrAllMeteoPoints; i++)
    {
        if (myStation == _allMeteoPointsPointer[i].name)
        {
            Crit3DMeteoPoint myStationMp = _allMeteoPointsPointer[i];
            double X0 = myStationMp.point.utm.x;
            double Y0 = myStationMp.point.utm.y;

            for (int j=0; j < _nrAllMeteoPoints; j++)
            {
                double computedDistance = gis::computeDistance(X0, Y0, _allMeteoPointsPointer[j].point.utm.x, _allMeteoPointsPointer[j].point.utm.y);
                if (computedDistance <= chosenDistance)
                {
                    _nearStationsList.append(QString::fromStdString(_allMeteoPointsPointer[j].name));
                }
            }
        }
    }

    this->update(); //aggiorna tutta la widget
    QDialog::done(QDialog::Accepted);
}

