#include "dialogNewPoint.h"

DialogNewPoint::DialogNewPoint(QList<QString> idList, gis::Crit3DRasterGrid DEM)
:idList(idList), DEM(DEM)
{
    setWindowTitle("New point");
    this->resize(400, 180);
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    this->setAttribute(Qt::WA_DeleteOnClose);

    QVBoxLayout *mainLayout = new QVBoxLayout();
    QHBoxLayout *idLayout = new QHBoxLayout();
    QHBoxLayout *utmxLayout = new QHBoxLayout();
    QHBoxLayout *utmyLayout = new QHBoxLayout();
    QHBoxLayout *latLonLayout = new QHBoxLayout();
    QHBoxLayout *heightLayout = new QHBoxLayout;
    QHBoxLayout *layoutOk = new QHBoxLayout;
    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    layoutOk->addWidget(&buttonBox);

    QLabel *idLabel = new QLabel("Id point");
    id.setMaximumWidth(60);
    id.setMaximumHeight(30);
    QLabel *utmxLabel = new QLabel("UTM X");
    QLabel *utmyLabel = new QLabel("UTM Y");
    utmx.setMaximumWidth(60);
    utmx.setMaximumHeight(30);
    utmy.setMaximumWidth(60);
    utmy.setMaximumHeight(30);

    QLabel *latLabel = new QLabel("Latitude");
    QLabel *lonLabel = new QLabel("Longitude");
    lat.setMaximumWidth(60);
    lat.setMaximumHeight(30);
    lon.setMaximumWidth(60);
    lon.setMaximumHeight(30);

    QLabel *heightLabel = new QLabel("Height");
    height.setMaximumWidth(60);
    height.setMaximumHeight(30);

    idLayout->addWidget(idLabel);
    idLayout->addWidget(&id);
    utmxLayout->addWidget(utmxLabel);
    utmxLayout->addWidget(&utmx);
    utmyLayout->addWidget(utmyLabel);
    utmyLayout->addWidget(&utmy);

    latLonLayout->addWidget(latLabel);
    latLonLayout->addWidget(&lat);
    latLonLayout->addWidget(lonLabel);
    latLonLayout->addWidget(&lon);

    computeUTMButton.setText("Compute from UTMxy");
    latLonLayout->addWidget(&computeUTMButton);

    heightLayout->addWidget(heightLabel);
    heightLayout->addWidget(&height);
    getFromDEMButton.setText("Get from Digital Elevation Map");
    heightLayout->addWidget(&getFromDEMButton);

    mainLayout->addLayout(idLayout);
    mainLayout->addLayout(utmxLayout);
    mainLayout->addLayout(utmyLayout);
    mainLayout->addLayout(latLonLayout);
    mainLayout->addLayout(heightLayout);
    mainLayout->addLayout(layoutOk);
    setLayout(mainLayout);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    show();
    exec();
}
