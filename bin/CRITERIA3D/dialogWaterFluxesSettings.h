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
        QLineEdit *imposedComputationDepthEdit;

    public:
        QCheckBox *snowProcess;
        QCheckBox *cropProcess;
        QCheckBox *waterFluxesProcess;

        QRadioButton *onlySurface;
        QRadioButton *allSoilDepth;
        QRadioButton *imposedDepth;
        QRadioButton *useWaterRetentionFitting;

        QSlider *accuracySlider;

        DialogWaterFluxesSettings();

        double getInitialWaterPotential() const
        { return initialWaterPotentialEdit->text().toDouble(); }

        void setInitialWaterPotential(double value)
        { initialWaterPotentialEdit->setText(QString::number(value)); }

        double getImposedComputationDepth() const
        { return imposedComputationDepthEdit->text().toDouble(); }

        void setImposedComputationDepth(double value)
        { imposedComputationDepthEdit->setText(QString::number(value)); }
    };


#endif // DIALOGWATERFLUXESSETTINGS_H
