#include "dialogNewPoint.h"

DialogNewPoint::DialogNewPoint(QList<QString> idList, gis::Crit3DRasterGrid DEM)
:idList(idList), DEM(DEM)
{
    setWindowTitle("New point");
    this->resize(400, 180);
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

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
    QDoubleValidator *doubleValLat = new QDoubleValidator( -90.0, 90.0, 5, this );
    doubleValLat->setNotation(QDoubleValidator::StandardNotation);
    lat.setValidator(doubleValLat);
    QDoubleValidator *doubleValLon = new QDoubleValidator( -180.0, 180.0, 5, this );
    doubleValLon->setNotation(QDoubleValidator::StandardNotation);
    lon.setValidator(doubleValLon);

    QLabel *heightLabel = new QLabel("Height");
    height.setMaximumWidth(60);
    height.setMaximumHeight(30);
    QDoubleValidator *doubleValHeight = new QDoubleValidator( -9999.0, 9999.0, 5, this );
    doubleValHeight->setNotation(QDoubleValidator::StandardNotation);
    height.setValidator(doubleValHeight);

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

    exec();
}

DialogNewPoint::~DialogNewPoint()
{
    close();
}

void DialogNewPoint::done(bool res)
{
    if (res) // ok
    {
        if (id.text().isEmpty())
        {
            QMessageBox::information(nullptr, "Missing id point ", "Insert id point");
            return;
        }
        if (idList.contains(id.text()))
        {
            QMessageBox::information(nullptr, "id point already used", "Change id point");
            return;
        }

        if (lat.text().isEmpty())
        {
            QMessageBox::information(nullptr, "Missing latitude ", "Insert latitude");
            return;
        }
        else if (lat.text().toDouble() < -90 || lat.text().toDouble() > 90)
        {
            QMessageBox::information(nullptr, "Invalid latitude ", "Insert a value between -90 and 90");
            return;
        }
        if (lon.text().isEmpty())
        {
            QMessageBox::information(nullptr, "Missing longitude ", "Insert longitude");
            return;
        }
        else if (lon.text().toDouble() < -180 || lon.text().toDouble() > 180)
        {
            QMessageBox::information(nullptr, "Invalid longitude ", "Insert a value between -180 and 180");
            return;
        }
        if (height.text().isEmpty())
        {
            QMessageBox::information(nullptr, "Missing height ", "Insert height");
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
