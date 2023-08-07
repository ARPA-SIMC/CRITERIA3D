#ifndef DIALOGSNOWSETTINGS_H
#define DIALOGSNOWSETTINGS_H

#include <QtWidgets>

// custom QDoubleValidator to allow empty values
class doubleValidator : public QDoubleValidator {

public:
    doubleValidator ( double bottom, double top, int decimals, QObject* parent );
    QValidator::State validate (QString& s, int& ) const;
};

class DialogSnowSettings : public QDialog
{
    Q_OBJECT
public:
    DialogSnowSettings(QWidget *parent = nullptr);

    double getRainfallThresholdValue() const;
    void setRainfallThresholdValue(double value);

    double getSnowThresholdValue() const;
    void setSnowThresholdValue(double value);

    double getWaterHoldingValue() const;
    void setWaterHoldingValue(double value);

    double getSurfaceThickValue() const;
    void setSurfaceThickValue(double value);

    double getVegetationHeightValue() const;
    void setVegetationHeightValue(double value);

    double getSoilAlbedoValue() const;
    void setSoilAlbedoValue(double value);

    bool checkEmptyValues();
    bool checkWrongValues();
    void accept();

private:
    QLineEdit *rainfallThresholdValue;
    QLineEdit *snowThresholdValue;
    QLineEdit *waterHoldingValue;
    QLineEdit *surfaceThickValue;
    QLineEdit *vegetationHeightValue;
    QLineEdit *soilAlbedoValue;
};

#endif // DIALOGSNOWSETTINGS_H
