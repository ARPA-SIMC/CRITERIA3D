#ifndef DIALOGRADIATION_H
#define DIALOGRADIATION_H

    #include <QtWidgets>

    #ifndef PROJECT_H
        #include "project.h"
    #endif

    class QDialogButtonBox;

    class DialogRadiation : public QDialog
    {
        Q_OBJECT

        public:
            explicit DialogRadiation(Project* myProject);

            QComboBox* comboAlgorithm;

            QCheckBox* checkRealSky;
            QComboBox* comboRealSky;
            QLineEdit* editTransClearSky;

            QGroupBox* groupLinke;
            QComboBox* comboLinkeMode;
            QLineEdit* editLinke;
            QLineEdit* editLinkeMap;
            QPushButton* buttonLinke;

            QComboBox* comboAlbedoMode;
            QLineEdit* editAlbedo;
            QLineEdit* editAlbedoMap;
            QPushButton* buttonAlbedo;

            QComboBox* comboTiltMode;
            QLineEdit* editTilt;
            QLineEdit* editAspect;

            QCheckBox* checkShadowing;

            void loadLinke();
            void loadAlbedo();

            void updateRealSky();
            void updateAlgorithm(const QString myString);
            void updateRealSkyAlgorithm(const QString myString);
            void updateLinkeMode(const QString myString);
            void updateAlbedoMode(const QString myString);
            void updateTiltMode(const QString myString);

            void accept();

        protected:
            Project* project_;
            gis::Crit3DRasterGrid* linkeMap;
            gis::Crit3DRasterGrid* albedoMap;

        private:
            QDialogButtonBox *buttonBox;
    };

#endif // DIALOGRADIATION_H
