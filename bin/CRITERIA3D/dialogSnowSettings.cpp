#include <QtWidgets>
#include "dialogSnowSettings.h"
#include "QDoubleValidator"

DialogSnowSettings::DialogSnowSettings(QWidget *parent) : QDialog(parent)
{
    setWindowTitle("Snow Settings");
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QGridLayout *layoutSettings = new QGridLayout();
    QHBoxLayout *layoutOk = new QHBoxLayout();

    QLabel *rainfallThreshold = new QLabel(tr("All rainfall at temperature [°C] > "));
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

    layoutSettings->addWidget(rainfallThreshold, 0 , 0);
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
    if (rainfallThresholdValue->text().toDouble(&ok) < 0.0 || rainfallThresholdValue->text().toDouble() > 4.0)
    {
        QMessageBox::information(nullptr, "Insert a valid value", "Max temp value with snow should be in [0:4]");
        return false;
    }
    if (!ok)
    {
        QMessageBox::information(nullptr, "Insert a valid value", "Max temp value with snow is not a number");
        return false;
    }
    if (snowThresholdValue->text().toDouble(&ok) < -2.0 || snowThresholdValue->text().toDouble() > 1.0)
    {
        QMessageBox::information(nullptr, "Insert a valid value", "Min temp value with rain should be in [-2:1]");
        return false;
    }
    if (!ok)
    {
        QMessageBox::information(nullptr, "Insert a valid value", "Min temp value with rain is not a number");
        return false;
    }
    if (waterHoldingValue->text().toDouble(&ok) < 0 || waterHoldingValue->text().toDouble() > 1)
    {
        QMessageBox::information(nullptr, "Insert a valid value", "Water holding capacity should be in [0:1]");
        return false;
    }
    if (!ok)
    {
        QMessageBox::information(nullptr, "Insert a valid value", "Water holding capacity is not a number");
        return false;
    }
    if (surfaceThickValue->text().toDouble(&ok) < 0.001 || surfaceThickValue->text().toDouble() > 0.02)
    {
        QMessageBox::information(nullptr, "Insert a valid value", "Snow skin thickness should be in [0.001:0.02]");
        return false;
    }
    if (!ok)
    {
        QMessageBox::information(nullptr, "Insert a valid value", "Snow skin thickness is not a number");
        return false;
    }
    if (vegetationHeightValue->text().toDouble(&ok) < 0 || vegetationHeightValue->text().toDouble() > 10)
    {
        QMessageBox::information(nullptr, "Insert a valid value", "Vegetation heigh should be in [0:10]");
        return false;
    }
    if (!ok)
    {
        QMessageBox::information(nullptr, "Insert a valid value", "Vegetation heigh is not a number");
        return false;
    }
    if (soilAlbedoValue->text().toDouble(&ok) < 0 || soilAlbedoValue->text().toDouble() > 1)
    {
        QMessageBox::information(nullptr, "Insert a valid value", "Soil albedo should be in [0:1]");
        return false;
    }
    if (!ok)
    {
        QMessageBox::information(nullptr, "Insert a valid value", "Soil albedo is not a number");
        return false;
    }
    return true;
}

double DialogSnowSettings::getRainfallThresholdValue() const
{
    return rainfallThresholdValue->text().toDouble();
}

void DialogSnowSettings::setRainfallThresholdValue(double value)
{
    rainfallThresholdValue->setText(QString::number(value));
}

double DialogSnowSettings::getSnowThresholdValue() const
{
    return snowThresholdValue->text().toDouble();
}

void DialogSnowSettings::setSnowThresholdValue(double value)
{
    snowThresholdValue->setText(QString::number(value));
}

double DialogSnowSettings::getWaterHoldingValue() const
{
    return waterHoldingValue->text().toDouble();
}

void DialogSnowSettings::setWaterHoldingValue(double value)
{
    waterHoldingValue->setText(QString::number(value));
}

double DialogSnowSettings::getSurfaceThickValue() const
{
    return surfaceThickValue->text().toDouble();
}

void DialogSnowSettings::setSurfaceThickValue(double value)
{
    surfaceThickValue->setText(QString::number(value));
}

double DialogSnowSettings::getVegetationHeightValue() const
{
    return vegetationHeightValue->text().toDouble();
}

void DialogSnowSettings::setVegetationHeightValue(double value)
{
    vegetationHeightValue->setText(QString::number(value));
}

double DialogSnowSettings::getSoilAlbedoValue() const
{
    return soilAlbedoValue->text().toDouble();
}

void DialogSnowSettings::setSoilAlbedoValue(double value)
{
    soilAlbedoValue->setText(QString::number(value));
}
