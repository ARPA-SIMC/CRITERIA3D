#ifndef DIALOGWATERFLUXESSETTINGS_H
#define DIALOGWATERFLUXESSETTINGS_H

    #include <QDialog>
    #include <QLineEdit>
    #include <QRadioButton>
    #include <QCheckBox>
    #include <QSlider>

    class DialogWaterFluxesSettings : public QDialog
    {
    private:
        QLineEdit *initialWaterPotentialEdit;
        QLineEdit *initialDegreeOfSaturationEdit;
        QLineEdit *imposedComputationDepthEdit;
        QLineEdit *conductivityHVRatioEdit;

        bool _isUpdateAccuracy;

    private slots :
        void updateAccuracy() { _isUpdateAccuracy = true; }

    public:
        QPushButton *updateButton;

        QRadioButton *useInitialWaterPotential;
        QRadioButton *useInitialDegreeOfSaturation;

        QRadioButton *onlySurface;
        QRadioButton *allSoilDepth;
        QRadioButton *imposedDepth;
        QRadioButton *useWaterRetentionFitting;

        QSlider *accuracySlider;

        DialogWaterFluxesSettings();

        double getConductivityHVRatio() const
        { return conductivityHVRatioEdit->text().toDouble(); }

        void setConductivityHVRatio(double value)
        { conductivityHVRatioEdit->setText(QString::number(value)); }

        double getInitialWaterPotential() const
        { return initialWaterPotentialEdit->text().toDouble(); }

        void setInitialWaterPotential(double value)
        { initialWaterPotentialEdit->setText(QString::number(value)); }

        double getInitialDegreeOfSaturation() const
        { return initialDegreeOfSaturationEdit->text().toDouble(); }

        void setInitialDegreeOfSaturation(double value)
        { initialDegreeOfSaturationEdit->setText(QString::number(value)); }

        double getImposedComputationDepth() const
        { return imposedComputationDepthEdit->text().toDouble(); }

        void setImposedComputationDepth(double value)
        { imposedComputationDepthEdit->setText(QString::number(value)); }

        bool isUpdateAccuracy() const
        { return _isUpdateAccuracy; }
    };


#endif // DIALOGWATERFLUXESSETTINGS_H
