#include "dialogAddStation.h"
#include "commonConstants.h"
#include "gis.h"

DialogAddStation::DialogAddStation(const QList<QString> &activeStationsList, Crit3DMeteoPoint *allMeteoPointsPointer, int nrAllMeteoPoints)
    : _activeStationsList(activeStationsList), _allMeteoPointsPointer(allMeteoPointsPointer), _nrAllMeteoPoints(nrAllMeteoPoints)
{
    setWindowTitle("Add stations");

    QVBoxLayout *mainLayout = new QVBoxLayout;
    QHBoxLayout *stationLayout = new QHBoxLayout;
    QHBoxLayout *singleValueLayout = new QHBoxLayout;
    QHBoxLayout *nearStationsLayout = new QHBoxLayout;
    QHBoxLayout *searchButtonLayout = new QHBoxLayout;
    QHBoxLayout *buttonsLayout = new QHBoxLayout;

    QPushButton *_search = new QPushButton("Search stations");
    searchButtonLayout->addWidget(_search);
    connect(_search, &QPushButton::clicked, [=](){ this->searchStations(); });

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    buttonsLayout->addWidget(&buttonBox);

    QLabel *stationHeader = new QLabel("Active stations");
    stationLayout->addWidget(stationHeader);
    _listActiveStationsWidget = new QComboBox;
    _listActiveStationsWidget->addItems(_activeStationsList);
    stationLayout->addWidget(_listActiveStationsWidget);

    QLabel singleValueLabel("Insert distance [m]:");
    _singleValueEdit = new QLineEdit;
    _singleValueEdit->setValidator(new QIntValidator(0, 10000));
    _singleValueEdit->setText("1000");
    singleValueLayout->addWidget(&singleValueLabel);
    singleValueLayout->addWidget(_singleValueEdit);

    QLabel nearStationsLabel("Near stations");
    nearStationsLayout->addWidget(&nearStationsLabel);
    _listNearStationsWidget = new QListWidget;
    _listNearStationsWidget->addItems(_nearStationsList);
    nearStationsLayout->addWidget(_listNearStationsWidget);

    mainLayout->addLayout(stationLayout);
    mainLayout->addLayout(singleValueLayout);
    mainLayout->addLayout(searchButtonLayout);
    mainLayout->addLayout(nearStationsLayout);
    mainLayout->addLayout(buttonsLayout);
    setLayout(mainLayout);

    //connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->addStation(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->close(); });

    show();
    exec();
}


double DialogAddStation::getSingleValue()
{
    bool isNumber = false;
    double chosenDistance = _singleValueEdit->text().toFloat(&isNumber);
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

            _nearStationsList.clear();
            for (int j=0; j < _nrAllMeteoPoints; j++)
            {
                double computedDistance = gis::computeDistance(X0, Y0, _allMeteoPointsPointer[j].point.utm.x, _allMeteoPointsPointer[j].point.utm.y);
                if (computedDistance <= chosenDistance)
                {
                    _nearStationsList.append(QString::fromStdString(_allMeteoPointsPointer[j].name));
                }
            }

            _listNearStationsWidget->clear();
            _listNearStationsWidget->addItems(_nearStationsList);
            break;
        }
    }

    this->update(); // aggiorna la widget
}

