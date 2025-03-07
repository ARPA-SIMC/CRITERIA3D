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

    _listNearStationsWidget = new QComboBox;
    //ma forse ci vorrebbe una combobox per scegliere quale tra le attive usare

    QLabel *stationHeader = new QLabel("Active stations"); //header stazioni
    _listActiveStationsWidget->addItems(_activeStationsList);
    stationLayout->addWidget(_listActiveStationsWidget);

    QLabel singleValueLabel("Insert distance [m]:"); //check unità di misura
    singleValueLayout->addWidget(&singleValueLabel);
    singleValueLabel.setBuddy(&_singleValueEdit);

    _singleValueEdit.setValidator(new QDoubleValidator(0.0, 9999.0,1));
    _singleValueEdit.setText(QString::number(getSingleValue()));

    singleValueLayout->addWidget(&_singleValueEdit);

    QLabel nearStationsLabel("Near stations"); //header stazioni vicine
    nearStationsLayout->addWidget(&nearStationsLabel);
    _listNearStationsWidget->addItems(_nearStationsList); //qua bisognerà aggiungere le stazioni vicine
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

    // Bottoni ok e cancel. Infatti this è il puntatore al dialog che ha la funzione 'done'
    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->searchStations(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->searchStations(false); });

    show();
    exec();

}

void DialogAddStation::searchStations(bool res)
{
    if (res)
    {
        QList<QListWidgetItem*> _nearStationsList = _listActiveStationsWidget->selectedItems();

        //qua andrà inserita la funzione che cerca i meteopoints entro la distanza scelta
        //connect(this(la widget), SIGNAL(nomeFunzioneChiamante), object2(la main window), SLOT(widgetListaStazioniVicine));


        QDialog::done(QDialog::Accepted);
    }
    else    // cancel, close or exc was pressed
    {
        QDialog::done(QDialog::Rejected);
        return;
    }
}

double DialogAddStation::getSingleValue()
{
    return _singleValueEdit.text().toFloat();
}

QList<QString> DialogAddStation::getSelectedStations()
{
    return _selectedStations;
}


