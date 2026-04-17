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
        // initial conditions
        QLineEdit *initialWaterPotentialEdit;
        QLineEdit *initialDegreeOfSaturationEdit;

        // soil depth
        QRadioButton *onlySurfaceButton;
        QRadioButton *allSoilDepthButton;
        QRadioButton *imposedDepthButton;
        QLineEdit *imposedComputationDepthEdit;

        // soil properties
        QRadioButton *useWaterRetentionFitting;
        QLineEdit *conductivityHVRatioEdit;

        // boundary conditions
        QCheckBox *freeCatchmentRunoffBox;
        QCheckBox *freeLateralDrainageBox;
        QCheckBox *freeBottomDrainageBox;

        // numerical solution
        QCheckBox *useLineal;
        QRadioButton *GaussSeidel;
        QRadioButton *conjugateGradient;
        QRadioButton *pgc_sor;
        QRadioButton *pcg_amg_sor;
        QLineEdit *threadsNumberEdit;
        bool _isUpdateAccuracy;

    private slots :
        void updateAccuracy();

    public:
        QPushButton *updateButton;

        QRadioButton *useInitialWaterPotential;
        QRadioButton *useInitialDegreeOfSaturation;

        QSlider *accuracySlider;

        DialogWaterFluxesSettings();

        void setLinealAvailable(bool value)
        {
            useLineal->setVisible(value);
            conjugateGradient->setVisible(value);
            pgc_sor->setVisible(value);
            pcg_amg_sor->setVisible(value);
        }

        void setLinealUse(bool value) { useLineal->setChecked(value); }

        bool getUseLineal() const { return useLineal->isChecked(); }

        void setLinealMethod(int value)
        {
            if (value == 0)
                GaussSeidel->setChecked(true);
            else if (value == 1)
                conjugateGradient->setChecked(true);
            else if (value == 2)
                pgc_sor->setChecked(true);
            else if (value == 3)
                pcg_amg_sor->setChecked(true);
        }

        int getLinealMethod() const
        {
            if (GaussSeidel->isChecked())
                return 0;
            if (conjugateGradient->isChecked())
                return 1;
            else if (pgc_sor->isChecked())
                return 2;
            else if (pcg_amg_sor->isChecked())
                return 3;
            else
                return 1;
        }

        int getThreadsNumber() const
        { return threadsNumberEdit->text().toInt(); }

        void setOnlySurface(bool isChecked)
        {
            if (onlySurfaceButton != nullptr)
                onlySurfaceButton->setChecked(isChecked);
        }

        bool getOnlySurface() const
        {
            if (onlySurfaceButton != nullptr)
                return onlySurfaceButton->isChecked();
            return false;
        }

        void setAllSoilDepth(bool isChecked)
        {
            if (allSoilDepthButton != nullptr)
                allSoilDepthButton->setChecked(isChecked);
        }

        bool getAllSoilDepth() const
        {
            if (allSoilDepthButton != nullptr)
                return allSoilDepthButton->isChecked();
            return false;
        }

        void setImposedDepth(bool isChecked)
        {
            if (imposedDepthButton != nullptr)
                imposedDepthButton->setChecked(isChecked);
        }

        bool getImposedDepth() const
        {
            if (imposedDepthButton != nullptr)
                return imposedDepthButton->isChecked();
            return false;
        }

        void setUseWaterRetentionFitting(bool isChecked)
        {
            if (useWaterRetentionFitting != nullptr)
                useWaterRetentionFitting->setChecked(isChecked);
        }

        bool getUseWaterRetentionFitting() const
        {
            if (useWaterRetentionFitting != nullptr)
                return useWaterRetentionFitting->isChecked();
            return false;
        }

        void setFreeCatchmentRunoff(bool isChecked)
        {
            if (freeCatchmentRunoffBox != nullptr)
                freeCatchmentRunoffBox->setChecked(isChecked);
        }

        bool getFreeCatchmentRunoff() const
        {
            if (freeCatchmentRunoffBox != nullptr)
                return freeCatchmentRunoffBox->isChecked();
            return false;
        }

        void setFreeLateralDrainage(bool isChecked)
        {
            if (freeLateralDrainageBox != nullptr)
                freeLateralDrainageBox->setChecked(isChecked);
        }

        bool getFreeLateralDrainage() const
        {
            if (freeLateralDrainageBox != nullptr)
                return freeLateralDrainageBox->isChecked();
            return false;
        }

        void setFreeBottomDrainage(bool isChecked)
        {
            if (freeBottomDrainageBox != nullptr)
                freeBottomDrainageBox->setChecked(isChecked);
        }

        bool getFreeBottomDrainage() const
        {
            if (freeBottomDrainageBox != nullptr)
                return freeBottomDrainageBox->isChecked();
            return false;
        }

        void setThreadsNumber(int value)
        { threadsNumberEdit->setText(QString::number(value)); }

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
