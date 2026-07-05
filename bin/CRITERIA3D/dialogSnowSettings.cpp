#include <QtWidgets>
#include "dialogSnowSettings.h"
#include "QDoubleValidator"

DialogSnowSettings::DialogSnowSettings(QWidget *parent) : QDialog(parent)
{
    setWindowTitle("Snow Settings");
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QGridLayout *layoutSettings = new QGridLayout();
    QHBoxLayout *layoutOk = new QHBoxLayout();

    QLabel *rainfallThresholdLabel = new QLabel(tr("All rainfall at temperature [°C] > "));
    rainfallThresholdValue = new QLineEdit();
    rainfallThresholdValue->setFixedWidth(70);

    QDoubleValidator* doubleRainfallThresholdVal = new QDoubleValidator(0.0, 4.0, 1, rainfallThresholdValue);
    doubleRainfallThresholdVal->setNotation(QDoubleValidator::StandardNotation);
    rainfallThresholdValue->setValidator(doubleRainfallThresholdVal);

    QLabel *snowThreshold = new QLabel(tr("All snow at temperature [°C] < "));
    snowThresholdValue = new QLineEdit();
    QDoubleValidator* doubleAllSnowThresholdVal = new QDoubleValidator(-2.0, 1.0, 1, snowThresholdValue);
    doubleAllSnowThresholdVal->setNotation(QDoubleValidator::StandardNotation);
    snowThresholdValue->setValidator(doubleAllSnowThresholdVal);
    snowThresholdValue->setFixedWidth(70);

    QLabel *waterHolding = new QLabel(tr("Water holding capacity [-] "));
    waterHoldingValue = new QLineEdit();
    QDoubleValidator* doubleWaterHoldingVal = new QDoubleValidator(0.0, 1.0, 2, waterHoldingValue);
    doubleWaterHoldingVal->setNotation(QDoubleValidator::StandardNotation);
    waterHoldingValue->setValidator(doubleWaterHoldingVal);
    waterHoldingValue->setFixedWidth(70);

    QLabel *surfaceThick = new QLabel(tr("Surface layer thickness [m] "));
    surfaceThickValue = new QLineEdit();
    QDoubleValidator* doubleSurfaceThickVal = new QDoubleValidator(0.001, 0.1, 3, surfaceThickValue);
    doubleSurfaceThickVal->setNotation(QDoubleValidator::StandardNotation);
    surfaceThickValue->setValidator(doubleSurfaceThickVal);
    surfaceThickValue->setFixedWidth(70);

    QLabel *vegetationHeight = new QLabel(tr("Vegetation height [m] "));
    vegetationHeightValue = new QLineEdit();
    QDoubleValidator* doubleVegetationHeightVal = new QDoubleValidator(0.0, 10.0, 1, vegetationHeightValue);
    doubleVegetationHeightVal->setNotation(QDoubleValidator::StandardNotation);
    vegetationHeightValue->setValidator(doubleVegetationHeightVal);
    vegetationHeightValue->setFixedWidth(70);

    QLabel *soilAlbedo = new QLabel(tr("Soil albedo [-] "));
    soilAlbedoValue = new QLineEdit();
    QDoubleValidator* doubleAlbedoVal = new QDoubleValidator(0.0, 1.0, 2, soilAlbedoValue);
    doubleAlbedoVal->setNotation(QDoubleValidator::StandardNotation);
    soilAlbedoValue->setValidator(doubleAlbedoVal);
    soilAlbedoValue->setFixedWidth(70);

    QLabel *dampingDepth = new QLabel(tr("Snow surface damping depth [m] "));
    snowDampingDepthValue = new QLineEdit();
    QDoubleValidator* doubleDampingDepthVal = new QDoubleValidator(0.0, 1.0, 2, snowDampingDepthValue);
    doubleDampingDepthVal->setNotation(QDoubleValidator::StandardNotation);
    snowDampingDepthValue->setValidator(doubleAlbedoVal);
    snowDampingDepthValue->setFixedWidth(70);

    layoutSettings->addWidget(rainfallThresholdLabel, 0 , 0);
    layoutSettings->addWidget(rainfallThresholdValue, 0 , 1);
    layoutSettings->addWidget(snowThreshold, 1 , 0);
    layoutSettings->addWidget(snowThresholdValue, 1 , 1);
    layoutSettings->addWidget(waterHolding, 2 , 0);
    layoutSettings->addWidget(waterHoldingValue, 2 , 1);
    layoutSettings->addWidget(surfaceThick, 3 , 0);
    layoutSettings->addWidget(surfaceThickValue, 3 , 1);
    layoutSettings->addWidget(vegetationHeight, 4 , 0);
    layoutSettings->addWidget(vegetationHeightValue, 4 , 1);
    layoutSettings->addWidget(soilAlbedo, 5 , 0);
    layoutSettings->addWidget(soilAlbedoValue, 5 , 1);
    layoutSettings->addWidget(dampingDepth, 6 , 0);
    layoutSettings->addWidget(snowDampingDepthValue, 6 , 1);

    QDialogButtonBox *buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok
                                         | QDialogButtonBox::Cancel);

    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    layoutOk->addWidget(buttonBox);

    mainLayout->addLayout(layoutSettings);
    mainLayout->addLayout(layoutOk);

    setLayout(mainLayout);

    show();
}

void DialogSnowSettings::accept()
{
    if (checkEmptyValues() && checkWrongValues())
    {
        return QDialog::accept();
    }
    else
    {
        return;
    }
}


bool DialogSnowSettings::checkEmptyValues()
{
    if (rainfallThresholdValue->text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing max temp", "Insert maximum temperature with snow [0, 4]");
        return false;
    }
    if (snowThresholdValue->text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing min temp", "Insert minimum temperature with rain [-2, 1]");
        return false;
    }
    if (waterHoldingValue->text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing water holding", "Insert water holding capacity [0, 1]");
        return false;
    }
    if (surfaceThickValue->text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing snow thickness", "Insert snow skin thickness [0.001, 0.02]");
        return false;
    }
    if (vegetationHeightValue->text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing vegetation heigh", "Insert vegetation heigh [0, 10]");
        return false;
    }
    if (soilAlbedoValue->text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing soil albedo", "Insert soil albedo [0, 1]");
        return false;
    }
    return true;
}


bool DialogSnowSettings::checkWrongValues()
{
    bool ok;

    double _rainfallThreshold = QLocale().toDouble(rainfallThresholdValue->text(), &ok);
    if (!ok)
    {
        QMessageBox::information(nullptr, "Wrong value", "Max. temperature with snow value is not a number");
        return false;
    }
    if ((_rainfallThreshold < 0.0) || (_rainfallThreshold > 4.0))
    {
        QMessageBox::information(nullptr, "Wrong value", "Max. temperature with snow should be in [0:4]");
        return false;
    }

    double snowThreshold = QLocale().toDouble(snowThresholdValue->text(), &ok);
    if (!ok)
    {
        QMessageBox::information(nullptr, "Wrong value", "Min. temperature with rain value is not a number");
        return false;
    }
    if ((snowThreshold < -2.0) || (snowThreshold > 1.0))
    {
        QMessageBox::information(nullptr, "Wrong value", "Min. temperature with rain should be in [-2:1]");
        return false;
    }

    double waterHolding = QLocale().toDouble(waterHoldingValue->text(), &ok);
    if (!ok)
    {
        QMessageBox::information(nullptr, "Wrong value", "Water holding capacity value is not a number");
        return false;
    }
    if ((waterHolding < 0) || (waterHolding > 1))
    {
        QMessageBox::information(nullptr, "Wrong value", "Water holding capacity should be in [0:1]");
        return false;
    }

    double surfaceThick = QLocale().toDouble(surfaceThickValue->text(), &ok);
    if (!ok)
    {
        QMessageBox::information(nullptr, "Wrong value", "Snow skin thickness value is not a number");
        return false;
    }
    if ((surfaceThick < 0.001) || (surfaceThick > 0.02))
    {
        QMessageBox::information(nullptr, "Wrong value", "Snow skin thickness should be in [0.001 : 0.02]");
        return false;
    }

    double vegetationHeight = QLocale().toDouble(vegetationHeightValue->text(), &ok);
    if (!ok)
    {
        QMessageBox::information(nullptr, "Wrong value", "Vegetation height value is not a number");
        return false;
    }
    if ((vegetationHeight < 0) || (vegetationHeight > 10))
    {
        QMessageBox::information(nullptr, "Wrong value", "Vegetation height should be in [0:10]");
        return false;
    }

    double soilAlbedo = QLocale().toDouble(soilAlbedoValue->text(), &ok);
    if (!ok)
    {
        QMessageBox::information(nullptr, "Wrong value", "Soil albedo value is not a number");
        return false;
    }
    if ((soilAlbedo < 0) || (soilAlbedo > 1))
    {
        QMessageBox::information(nullptr, "Wrong value", "Soil albedo should be in [0:1]");
        return false;
    }

    return true;
}


double DialogSnowSettings::getRainfallThresholdValue() const
{
    return QLocale().toDouble(rainfallThresholdValue->text());
}

void DialogSnowSettings::setRainfallThresholdValue(double value)
{
    rainfallThresholdValue->setText(QLocale().toString(value));
}

double DialogSnowSettings::getSnowThresholdValue() const
{
    return QLocale().toDouble(snowThresholdValue->text());
}

void DialogSnowSettings::setSnowThresholdValue(double value)
{
    snowThresholdValue->setText(QLocale().toString(value));
}

double DialogSnowSettings::getWaterHoldingValue() const
{
    return QLocale().toDouble(waterHoldingValue->text());
}

void DialogSnowSettings::setWaterHoldingValue(double value)
{
    waterHoldingValue->setText(QLocale().toString(value));
}

double DialogSnowSettings::getSurfaceThickValue() const
{
    return QLocale().toDouble(surfaceThickValue->text());
}

void DialogSnowSettings::setSurfaceThickValue(double value)
{
    surfaceThickValue->setText(QLocale().toString(value));
}

double DialogSnowSettings::getVegetationHeightValue() const
{
    return QLocale().toDouble(vegetationHeightValue->text());
}

void DialogSnowSettings::setVegetationHeightValue(double value)
{
    vegetationHeightValue->setText(QLocale().toString(value));
}

double DialogSnowSettings::getSoilAlbedoValue() const
{
    return QLocale().toDouble(soilAlbedoValue->text());
}

void DialogSnowSettings::setSoilAlbedoValue(double value)
{
    soilAlbedoValue->setText(QLocale().toString(value));
}

double DialogSnowSettings::getSnowDampingDepthValue() const
{
    return QLocale().toDouble(snowDampingDepthValue->text());
}

void DialogSnowSettings::setSnowDampingDepthValue(double value)
{
    snowDampingDepthValue->setText(QLocale().toString(value));
}
