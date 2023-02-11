#ifndef CRITERIA1DWIDGET_H
#define CRITERIA1DWIDGET_H

    #ifndef MAX_YEARS
        #define MAX_YEARS 10
    #endif
    #ifndef DBMETEOGRID_H
        #include "dbMeteoGrid.h"
    #endif

    #include <QWidget>
    #include <QComboBox>
    #include <QGroupBox>
    #include <QLineEdit>
    #include <QLabel>

    #include "criteria1DProject.h"
    #include "tabLAI.h"
    #include "tabRootDepth.h"
    #include "tabRootDensity.h"
    #include "tabIrrigation.h"
    #include "tabWaterContent.h"
    #include "tabCarbonNitrogen.h"


    class Criteria1DWidget : public QWidget
    {
        Q_OBJECT

        public:
            Criteria1DWidget();
            void on_actionOpenProject();
            void on_actionNewProject();
            void on_actionOpenCropDB();
            void on_actionChooseCase();
            void on_actionChooseCrop(QString idCrop);
            void on_actionOpenMeteoDB();
            void on_actionOpenSoilDB();
            void on_actionExecuteCase();
            void on_actionChooseMeteo(QString idMeteo);
            void on_actionChooseFirstYear(QString year);
            void on_actionChooseLastYear(QString year);
            void on_actionChooseSoil(QString soilCode);
            void on_actionDeleteCrop();
            void on_actionRestoreData();
            void on_actionNewCrop();
            void on_actionSave();
            void on_actionUpdate();
            void on_actionViewWeather();
            void on_actionViewSoil();
            bool saveCrop();
            void updateMeteoPointValues();
            void updateCropParam(QString idCrop);
            bool updateCrop();
            void updateTabLAI();
            void updateTabRootDepth();
            void updateTabRootDensity();
            void updateTabIrrigation();
            void updateTabWaterContent();
            void updateTabCarbonNitrogen();

            void tabChanged(int index);
            bool checkIfCropIsChanged();
            void irrigationVolumeChanged();

        private:
            Crit1DProject myProject;

            Crit3DCrop cropFromDB;

            QString meteoTableName;
            bool cropChanged;
            QList<QString> yearList;
            bool onlyOneYear;

            Crit3DMeteoGridDbHandler xmlMeteoGrid;
            Crit3DMeteoSettings meteoSettings;

            QGroupBox *infoCaseGroup;
            QGroupBox *infoCropGroup;
            QGroupBox *infoMeteoGroup;
            QGroupBox *infoSoilGroup;
            QGroupBox *laiParametersGroup;
            QGroupBox *rootParametersGroup;
            QGroupBox *irrigationParametersGroup;
            QGroupBox *waterStressParametersGroup;
            QGroupBox *waterContentGroup;
            QGroupBox *carbonNitrogenGroup;

            QComboBox caseListComboBox;
            QComboBox cropListComboBox;
            QComboBox meteoListComboBox;
            QComboBox soilListComboBox;
            QComboBox firstYearListComboBox;
            QComboBox lastYearListComboBox;
            QLineEdit* cropNameValue;
            QLineEdit* cropTypeValue;
            QLineEdit* maxKcValue;
            QLabel cropSowing;
            QLabel cropCycleMax;
            QSpinBox *cropSowingValue;
            QSpinBox* cropCycleMaxValue;
            QDoubleSpinBox* LAIminValue;
            QDoubleSpinBox* LAImaxValue;
            QLabel *LAIgrass;
            QLineEdit* LAIgrassValue;
            QLineEdit* thermalThresholdValue;
            QLineEdit* upperThermalThresholdValue;
            QLineEdit* degreeDaysEmergenceValue;
            QLineEdit* degreeDaysLAIincValue;
            QLineEdit* degreeDaysLAIdecValue;
            QLineEdit* LAIcurveAValue;
            QLineEdit* LAIcurveBValue;
            QLineEdit* rootDepthZeroValue;
            QLineEdit* rootDepthMaxValue;
            QComboBox* rootShapeComboBox;
            QDoubleSpinBox* shapeDeformationValue;
            QLabel *degreeDaysInc;
            QLineEdit* degreeDaysIncValue;
            QLineEdit* irrigationVolumeValue;
            QSpinBox* irrigationShiftValue;
            QLineEdit* degreeDaysStartValue;
            QLineEdit* degreeDaysEndValue;
            QLineEdit* psiLeafValue;
            QDoubleSpinBox* rawFractionValue;
            QDoubleSpinBox* stressToleranceValue;

            QRadioButton *volWaterContent;
            QRadioButton *degreeSat;
            QRadioButton *nitrogen_NH3;
            QRadioButton *nitrogen_NH4;
            QRadioButton *nitrogen_humus;
            QRadioButton *nitrogen_litter;
            QRadioButton *carbon_humus;
            QRadioButton *carbon_litter;

            QTabWidget* tabWidget;
            QAction* saveChanges;
            QAction* restoreData;
            QPushButton *saveButton;
            QPushButton *updateButton;
            QMenu *viewMenu;
            QAction* viewWeather;
            QAction* viewSoil;

            TabLAI* tabLAI;
            TabRootDepth* tabRootDepth;
            TabRootDensity* tabRootDensity;
            TabIrrigation* tabIrrigation;
            TabWaterContent* tabWaterContent;
            TabCarbonNitrogen* tabCarbonNitrogen;

            bool isRedraw;

            void clearCrop();
            void checkCropUpdate();
            void openComputationUnitsDB(QString dbComputationUnitsName);
            void openCropDB(QString dbCropName);
            void openMeteoDB(QString dbMeteoName);
            void openSoilDB(QString dbSoilName);
            bool setMeteoSqlite(QString &error);
    };


#endif // CRITERIA1DWIDGET_H
