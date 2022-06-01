#include "dialogNewPoint.h"

DialogNewPoint::DialogNewPoint(const QList<QString>& _idList, const gis::Crit3DGisSettings& _gisSettings, gis::Crit3DRasterGrid* _DEMptr)
:idList(_idList), gisSettings(_gisSettings), DEMpointer(_DEMptr)
{

    setWindowTitle("New point");
    this->resize(300, 180);
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

    QVBoxLayout *mainLayout = new QVBoxLayout();
    QVBoxLayout *idLayout = new QVBoxLayout();
    QVBoxLayout *utmxLayout = new QVBoxLayout();
    QVBoxLayout *utmyLayout = new QVBoxLayout();
    QVBoxLayout *latLayout = new QVBoxLayout();
    QVBoxLayout *lonLayout = new QVBoxLayout();
    QVBoxLayout *heightLayout = new QVBoxLayout;
    QHBoxLayout *layoutOk = new QHBoxLayout;
    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    layoutOk->addWidget(&buttonBox);

    QLabel *idLabel = new QLabel("Id point");
    id.setMaximumWidth(80);
    id.setMaximumHeight(30);

    QLabel *utmxLabel = new QLabel("UTM X");
    QLabel *utmyLabel = new QLabel("UTM Y");
    utmx.setMaximumWidth(80);
    utmx.setMaximumHeight(30);
    utmy.setMaximumWidth(80);
    utmy.setMaximumHeight(30);
    QDoubleValidator *doubleUTM = new QDoubleValidator( -99999999.0, 99999999.0, 5, this );
    doubleUTM->setNotation(QDoubleValidator::StandardNotation);
    utmx.setValidator(doubleUTM);
    utmy.setValidator(doubleUTM);

    QLabel *latLabel = new QLabel("Latitude");
    QLabel *lonLabel = new QLabel("Longitude");
    lat.setMaximumWidth(80);
    lat.setMaximumHeight(30);
    lon.setMaximumWidth(80);
    lon.setMaximumHeight(30);
    QDoubleValidator *doubleValLat = new QDoubleValidator( -90.0, 90.0, 5, this );
    doubleValLat->setNotation(QDoubleValidator::StandardNotation);
    lat.setValidator(doubleValLat);
    QDoubleValidator *doubleValLon = new QDoubleValidator( -180.0, 180.0, 5, this );
    doubleValLon->setNotation(QDoubleValidator::StandardNotation);
    lon.setValidator(doubleValLon);

    QLabel *heightLabel = new QLabel("Height");
    height.setMaximumWidth(80);
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
    latLayout->addWidget(latLabel);
    latLayout->addWidget(&lat);
    lonLayout->addWidget(lonLabel);
    lonLayout->addWidget(&lon);

    computeUTMButton.setText("Compute lat/lon from UTM xy");
    lonLayout->addWidget(&computeUTMButton);

    getFromDEMButton.setText("Get height from Digital Elevation Map");
    heightLayout->addWidget(&getFromDEMButton);
    heightLayout->addWidget(heightLabel);
    heightLayout->addWidget(&height);

    mainLayout->addLayout(idLayout);
    mainLayout->addLayout(latLayout);
    mainLayout->addLayout(lonLayout);
    mainLayout->addLayout(utmxLayout);
    mainLayout->addLayout(utmyLayout);
    mainLayout->addLayout(heightLayout);
    mainLayout->addLayout(layoutOk);
    setLayout(mainLayout);

    connect(&computeUTMButton, &QPushButton::clicked, [=](){ computeUTM(); });
    connect(&getFromDEMButton, &QPushButton::clicked, [=](){ getFromDEM(); });

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(QDialog::Accepted); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(QDialog::Rejected); });

    exec();
}

DialogNewPoint::~DialogNewPoint()
{
    close();
}

void DialogNewPoint::computeUTM()
{
    if (utmx.text().isEmpty() || utmy.text().isEmpty())
    {
        return;
    }
    double myLat;
    double myLon;
    gis::getLatLonFromUtm(gisSettings, utmx.text().toDouble(), utmy.text().toDouble(), &myLat, &myLon);
    lat.setText(QString::number(myLat));
    lon.setText(QString::number(myLon));
}

void DialogNewPoint::getFromDEM()
{
    if (DEMpointer == nullptr)
        return;

    if (! DEMpointer->isLoaded)
    {
        QMessageBox::information(nullptr, "DEM not loaded", "Load DEM");
        return;
    }
    float demValue;
    if (utmx.text().isEmpty() || utmy.text().isEmpty())
    {
        if (lat.text().isEmpty() || lon.text().isEmpty())
        {
            QMessageBox::information(nullptr, "missing coordinates", "Insert lat lon or utmx utmy");
            return;
        }
        gis::Crit3DGeoPoint point(lat.text().toDouble(), lon.text().toDouble());
        gis::Crit3DUtmPoint utmPoint;
        gis::getUtmFromLatLon(gisSettings.utmZone, point, &utmPoint);
        demValue = gis::getValueFromXY(*DEMpointer, utmPoint.x, utmPoint.y);
    }
    else
    {
        demValue = gis::getValueFromXY(*DEMpointer, utmx.text().toDouble(), utmy.text().toDouble());
    }

    if (demValue != DEMpointer->header->flag)
    {
        height.setText(QString::number(demValue));
    }
}

void DialogNewPoint::done(int res)
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
            QMessageBox::information(nullptr, "Change id point", "id point already used.");
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

        if (DEMpointer->isLoaded)
        {
            float demValue;
            if (utmx.text().isEmpty() || utmy.text().isEmpty())
            {
                gis::Crit3DGeoPoint point(lat.text().toDouble(), lon.text().toDouble());
                gis::Crit3DUtmPoint utmPoint;
                gis::getUtmFromLatLon(gisSettings.utmZone, point, &utmPoint);
                demValue = gis::getValueFromXY(*DEMpointer, utmPoint.x, utmPoint.y);
            }
            else
            {
                demValue = gis::getValueFromXY(*DEMpointer, utmx.text().toDouble(), utmy.text().toDouble());
            }

            if (demValue != DEMpointer->header->flag)
            {
                if ((height.text().toDouble() < demValue) || (height.text().toDouble() > demValue+2))
                {
                    QMessageBox::StandardButton reply;
                    reply = QMessageBox::question(this, "Are you sure?" ,
                                                  "DEM elevation in this point is different: " + QString::number(demValue),
                                                  QMessageBox::Yes|QMessageBox::No);
                    if (reply == QMessageBox::No)
                    {
                        return;
                    }
                }
            }
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

QString DialogNewPoint::getId()
{
    return id.text();
}

double DialogNewPoint::getLat()
{
    return lat.text().toDouble();
}

double DialogNewPoint::getLon()
{
    return lon.text().toDouble();
}

double DialogNewPoint::getHeight()
{
    return height.text().toDouble();
}
