#include "dialogSettings.h"
#include "commonConstants.h"


#define EDIT_SIZE 100


ProjectTab::ProjectTab(Project* project_)
{
    QLabel *startLocationLat = new QLabel(tr("<b>start location latitude </b> (negative for Southern Emisphere) [decimal degrees]:"));
    QDoubleValidator *doubleValLat = new QDoubleValidator( -90.0, 90.0, 5, this );
    doubleValLat->setNotation(QDoubleValidator::StandardNotation);
    startLocationLatEdit.setFixedWidth(EDIT_SIZE);
    startLocationLatEdit.setValidator(doubleValLat);
    startLocationLatEdit.setText(QString::number(project_->gisSettings.startLocation.latitude));

    QLabel *startLocationLon = new QLabel(tr("<b>start location longitude </b> [decimal degrees]:"));
    QDoubleValidator *doubleValLon = new QDoubleValidator( -180.0, 180.0, 5, this );
    doubleValLon->setNotation(QDoubleValidator::StandardNotation);
    startLocationLonEdit.setFixedWidth(EDIT_SIZE);
    startLocationLonEdit.setValidator(doubleValLon);
    startLocationLonEdit.setText(QString::number(project_->gisSettings.startLocation.longitude));

    QLabel *utmZone = new QLabel(tr("UTM zone:"));
    utmZoneEdit.setFixedWidth(EDIT_SIZE);
    utmZoneEdit.setValidator(new QIntValidator(0, 60));
    utmZoneEdit.setText(QString::number(project_->gisSettings.utmZone));

    QLabel *timeZone = new QLabel(tr("Time zone:"));
    timeZoneEdit.setFixedWidth(EDIT_SIZE);
    timeZoneEdit.setValidator(new QIntValidator(-12, 12));
    timeZoneEdit.setText(QString::number(project_->gisSettings.timeZone));

    QLabel *timeConvention = new QLabel(tr("Time Convention:"));
    QButtonGroup *group = new QButtonGroup();
    group->setExclusive(true);
    utc.setText("UTC");
    localTime.setText("Local Time");
    if (project_->gisSettings.isUTC)
    {
        utc.setChecked(true);
        localTime.setChecked(false);
    }
    else
    {
        utc.setChecked(false);
        localTime.setChecked(true);
    }
    group->addButton(&utc);
    group->addButton(&localTime);

    loadGridData.setText("Load grid data at start");
    loadGridData.setChecked(project_->loadGridDataAtStart);

    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(startLocationLat);
    mainLayout->addWidget(&startLocationLatEdit);

    mainLayout->addWidget(startLocationLon);
    mainLayout->addWidget(&startLocationLonEdit);

    mainLayout->addWidget(utmZone);
    mainLayout->addWidget(&utmZoneEdit);

    mainLayout->addWidget(timeZone);
    mainLayout->addWidget(&timeZoneEdit);

    QHBoxLayout *buttonLayout = new QHBoxLayout;
    buttonLayout->addWidget(timeConvention);
    buttonLayout->addWidget(&utc);
    buttonLayout->addWidget(&localTime);

    mainLayout->addLayout(buttonLayout);

    mainLayout->addWidget(&loadGridData);

    mainLayout->addStretch(1);
    setLayout(mainLayout);
}

QualityTab::QualityTab(Crit3DQuality *quality)
{
    QLabel *referenceClimateHeight = new QLabel(tr("reference height for quality control [m]:"));
    QDoubleValidator *doubleValHeight = new QDoubleValidator( -100.0, 100.0, 5, this );
    doubleValHeight->setNotation(QDoubleValidator::StandardNotation);
    referenceClimateHeightEdit.setFixedWidth(EDIT_SIZE);
    referenceClimateHeightEdit.setValidator(doubleValHeight);
    referenceClimateHeightEdit.setText(QString::number(quality->getReferenceHeight()));

    QLabel *deltaTSuspect = new QLabel(tr("difference in temperature in climatological control (suspect value) [degC]:"));
    QDoubleValidator *doubleValT = new QDoubleValidator( -100.0, 100.0, 5, this );
    doubleValT->setNotation(QDoubleValidator::StandardNotation);
    deltaTSuspectEdit.setFixedWidth(EDIT_SIZE);
    deltaTSuspectEdit.setValidator(doubleValT);
    deltaTSuspectEdit.setText(QString::number(quality->getDeltaTSuspect()));


    QLabel *deltaTWrong = new QLabel(tr("difference in temperature in climatological control (wrong value) [degC]:"));
    deltaTWrongEdit.setFixedWidth(EDIT_SIZE);
    deltaTWrongEdit.setValidator(doubleValT);
    deltaTWrongEdit.setText(QString::number(quality->getDeltaTWrong()));

    QLabel *humidityTolerance = new QLabel(tr("instrumental maximum allowed relative humidity [%]:"));
    humidityToleranceEdit.setFixedWidth(EDIT_SIZE);
    QDoubleValidator *doubleValPerc = new QDoubleValidator( 0.0, 100.0, 5, this );
    doubleValPerc->setNotation(QDoubleValidator::StandardNotation);
    humidityToleranceEdit.setFixedWidth(EDIT_SIZE);
    humidityToleranceEdit.setValidator(doubleValPerc);
    humidityToleranceEdit.setText(QString::number(quality->getRelHumTolerance()));

    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(referenceClimateHeight);
    mainLayout->addWidget(&referenceClimateHeightEdit);

    mainLayout->addWidget(deltaTSuspect);
    mainLayout->addWidget(&deltaTSuspectEdit);

    mainLayout->addWidget(deltaTWrong);
    mainLayout->addWidget(&deltaTWrongEdit);

    mainLayout->addWidget(humidityTolerance);
    mainLayout->addWidget(&humidityToleranceEdit);

    mainLayout->addStretch(1);
    setLayout(mainLayout);
}

MeteoTab::MeteoTab(Crit3DMeteoSettings *meteoSettings)
{
    QLabel *minimumPercentage = new QLabel(tr("minimum percentage of valid data [%]:"));
    minimumPercentageEdit.setFixedWidth(EDIT_SIZE);
    QDoubleValidator *doubleValPerc = new QDoubleValidator( 0.0, 100.0, 5, this );
    doubleValPerc->setNotation(QDoubleValidator::StandardNotation);
    minimumPercentageEdit.setFixedWidth(EDIT_SIZE);
    minimumPercentageEdit.setValidator(doubleValPerc);
    minimumPercentageEdit.setText(QString::number(meteoSettings->getMinimumPercentage()));

    QLabel *rainfallThreshold = new QLabel(tr("minimum value for valid precipitation [mm]:"));
    QDoubleValidator *doubleValThreshold = new QDoubleValidator( 0.0, 20.0, 5, this );
    doubleValThreshold->setNotation(QDoubleValidator::StandardNotation);
    rainfallThresholdEdit.setFixedWidth(EDIT_SIZE);
    rainfallThresholdEdit.setValidator(doubleValThreshold);
    rainfallThresholdEdit.setText(QString::number(meteoSettings->getRainfallThreshold()));

    QLabel *thomThreshold = new QLabel(tr("threshold for thom index [degC]:"));
    QDoubleValidator *doubleValThom = new QDoubleValidator( -100.0, 100.0, 5, this );
    doubleValThom->setNotation(QDoubleValidator::StandardNotation);
    thomThresholdEdit.setFixedWidth(EDIT_SIZE);
    thomThresholdEdit.setValidator(doubleValThom);
    thomThresholdEdit.setText(QString::number(meteoSettings->getThomThreshold()));

    QLabel *transSamaniCoefficient = new QLabel(tr("Samani coefficient for ET0 computation []:"));
    QDoubleValidator *doubleValSamani = new QDoubleValidator( -5.0, 5.0, 5, this );
    doubleValSamani->setNotation(QDoubleValidator::StandardNotation);
    transSamaniCoefficientEdit.setFixedWidth(EDIT_SIZE);
    transSamaniCoefficientEdit.setValidator(doubleValSamani);
    transSamaniCoefficientEdit.setText(QString::number(meteoSettings->getTransSamaniCoefficient()));

    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(minimumPercentage);
    mainLayout->addWidget(&minimumPercentageEdit);

    mainLayout->addWidget(rainfallThreshold);
    mainLayout->addWidget(&rainfallThresholdEdit);

    mainLayout->addWidget(thomThreshold);
    mainLayout->addWidget(&thomThresholdEdit);

    mainLayout->addWidget(transSamaniCoefficient);
    mainLayout->addWidget(&transSamaniCoefficientEdit);

    mainLayout->addStretch(1);
    setLayout(mainLayout);
}


DialogSettings::DialogSettings(Project* myProject)
{
    project_ = myProject;

    setWindowTitle(tr("Parameters"));
    setFixedSize(650,700);
    projectTab = new ProjectTab(myProject);
    qualityTab = new QualityTab(myProject->quality);
    metTab = new MeteoTab(myProject->meteoSettings);

    tabWidget = new QTabWidget;
    tabWidget->addTab(projectTab, tr("PROJECT"));
    tabWidget->addTab(qualityTab, tr("QUALITY"));
    tabWidget->addTab(metTab, tr("METEO"));

    buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(tabWidget);
    mainLayout->addWidget(buttonBox);
    setLayout(mainLayout);
}


bool DialogSettings::acceptValues()
{
    if (projectTab->startLocationLatEdit.text().isEmpty() || projectTab->startLocationLatEdit.text() == "0")
    {
        QMessageBox::information(nullptr, "Missing Parameter", "insert start location latitude");
        return false;
    }

    if (projectTab->startLocationLonEdit.text().isEmpty() || projectTab->startLocationLonEdit.text() == "0")
    {
        QMessageBox::information(nullptr, "Missing Parameter", "insert start location longitude");
        return false;
    }

    if (projectTab->utmZoneEdit.text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing Parameter", "insert UTM zone");
        return false;
    }

    if (projectTab->timeZoneEdit.text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing Parameter", "insert time zone");
        return false;
    }

    if (!projectTab->utc.isChecked() && !projectTab->localTime.isChecked())
    {
        QMessageBox::information(nullptr, "Missing time convention", "choose UTC or local time");
        return false;
    }

    // check UTM/time zone
    int utmZone = projectTab->utmZoneEdit.text().toInt();
    int timeZone = projectTab->timeZoneEdit.text().toInt();
    if (! gis::isValidUtmTimeZone(utmZone, timeZone))
    {
        QMessageBox::information(nullptr, "Wrong parameter", "Correct UTM zone or Time zone");
        return false;
    }

    ////////////////

    if (qualityTab->referenceClimateHeightEdit.text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing Parameter", "insert reference height for quality control");
        return false;
    }

    if (qualityTab->deltaTSuspectEdit.text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing Parameter", "insert difference in temperature suspect value");
        return false;
    }

    if (qualityTab->deltaTWrongEdit.text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing Parameter", "insert difference in temperature wrong value");
        return false;
    }

    if (qualityTab->humidityToleranceEdit.text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing Parameter", "instrumental maximum allowed relative humidity");
        return false;
    }

    ////////////////////

    if (metTab->minimumPercentageEdit.text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing Parameter", "insert minimum percentage of valid data");
        return false;
    }

    if (metTab->rainfallThresholdEdit.text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing Parameter", "insert minimum value for precipitation");
        return false;
    }

    if (metTab->thomThresholdEdit.text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing Parameter", "insert threshold for thom index");
        return false;
    }

    if (metTab->transSamaniCoefficientEdit.text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing Parameter", "insert Samani coefficient for ET0 computation");
        return false;
    }

    // store elaboration values

    project_->gisSettings.startLocation.latitude = projectTab->startLocationLatEdit.text().toDouble();
    project_->gisSettings.startLocation.longitude = projectTab->startLocationLonEdit.text().toDouble();
    project_->gisSettings.utmZone = projectTab->utmZoneEdit.text().toInt();
    project_->gisSettings.timeZone = projectTab->timeZoneEdit.text().toInt();
    project_->gisSettings.isUTC = projectTab->utc.isChecked();
    project_->loadGridDataAtStart = projectTab->loadGridData.isChecked();

    project_->quality->setReferenceHeight(qualityTab->referenceClimateHeightEdit.text().toFloat());
    project_->quality->setDeltaTSuspect(qualityTab->deltaTSuspectEdit.text().toFloat());
    project_->quality->setDeltaTWrong(qualityTab->deltaTWrongEdit.text().toFloat());
    project_->quality->setRelHumTolerance(qualityTab->humidityToleranceEdit.text().toFloat());

    project_->meteoSettings->setMinimumPercentage(metTab->minimumPercentageEdit.text().toFloat());
    project_->meteoSettings->setRainfallThreshold(metTab->rainfallThresholdEdit.text().toFloat());
    project_->meteoSettings->setThomThreshold(metTab->thomThresholdEdit.text().toFloat());
    project_->meteoSettings->setTransSamaniCoefficient(metTab->transSamaniCoefficientEdit.text().toFloat());

    project_->saveProjectSettings();
    project_->saveGenericParameters();

    return true;
}

void DialogSettings::accept()
{
    if (acceptValues()) QDialog::done(QDialog::Accepted);
}


QTabWidget *DialogSettings::getTabWidget() const
{
    return tabWidget;
}

void DialogSettings::setTabWidget(QTabWidget *value)
{
    tabWidget = value;
}
