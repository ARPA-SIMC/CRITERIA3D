#ifndef CROPWIDGET_H
#define CROPWIDGET_H

    #include <QWidget>
    #include <QComboBox>
    #include <QGroupBox>
    #include <QLineEdit>
    #include <QLabel>
    #include <QSqlDatabase>

    #include "tabLAI.h"
    #include "tabRootDepth.h"
    #include "tabRootDensity.h"

    class Crit3DCropWidget : public QWidget
    {
        Q_OBJECT

        public:
            Crit3DCropWidget();
            void on_actionOpenProject();
            void on_actionOpenCropDB();
            void on_actionChooseCrop(QString cropName);
            void on_actionOpenMeteoDB();
            void on_actionOpenSoilDB();
            void on_actionChooseMeteo(QString idMeteo);
            void on_actionChooseYear(QString year);
            void on_actionChooseSoil(QString soilCode);
            void on_actionDeleteCrop();
            void on_actionRestoreData();
            void on_actionNewCrop();
            void on_actionSave();
            void on_actionUpdate();
            bool saveCrop();
            void updateCropParam(QString idCrop);
            bool updateCrop();
            bool updateMeteoPoint();
            void updateTabLAI();
            void updateTabRootDepth();
            void updateTabRootDensity();
            void tabChanged(int index);
            bool checkIfCropIsChanged();

        private:
            QSqlDatabase dbCrop;
            QSqlDatabase dbMeteo;
            QSqlDatabase dbSoil;
            Crit3DCrop* myCrop;
            Crit3DCrop cropFromDB;
            soil::Crit3DSoil mySoil;
            soil::Crit3DTextureClass textureClassList[13];
            soil::Crit3DFittingOptions fittingOptions;
            double layerThickness;
            QString tableMeteo;
            Crit3DMeteoPoint *meteoPoint;
            std::vector<soil::Crit3DLayer> soilLayers;
            bool cropChanged;
            double meteoLatBackUp;

            QGroupBox *infoCropGroup;
            QGroupBox *infoMeteoGroup;
            QGroupBox *infoSoilGroup;
            QGroupBox *laiParametersGroup;
            QGroupBox *rootParametersGroup;
            QComboBox cropListComboBox;
            QComboBox meteoListComboBox;
            QComboBox soilListComboBox;
            QComboBox yearListComboBox;
            QLineEdit* cropIdValue;
            QLineEdit* cropTypeValue;
            QLineEdit* maxKcValue;
            QLabel cropSowing;
            QLabel cropCycleMax;
            QSpinBox *cropSowingValue;
            QSpinBox* cropCycleMaxValue;
            QDoubleSpinBox* latValue;
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
            QTabWidget* tabWidget;
            QAction* saveChanges;
            QAction* restoreData;
            QPushButton *saveButton;
            QPushButton *updateButton;

            TabLAI* tabLAI;
            TabRootDepth* tabRootDepth;
            TabRootDensity* tabRootDensity;

            void clearCrop();
            void checkCropUpdate();
            void openCropDB(QString newDbCropName);
            void openMeteoDB(QString dbMeteoName);
            void openSoilDB(QString dbSoilName);
    };


#endif // CROPWIDGET_H
