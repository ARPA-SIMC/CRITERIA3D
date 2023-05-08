#ifndef SOILWIDGET_H
#define SOILWIDGET_H

    #ifndef SOIL_H
        #include "soil.h"
    #endif
    #ifndef TABHORIZONS_H
        #include "tabHorizons.h"
    #endif
    #ifndef TABHYDRAULICCONDUCTIVITYCURVE_H
        #include "tabHydraulicConductivityCurve.h"
    #endif
    #ifndef TABWATERRETENTIONCURVE_H
        #include "tabWaterRetentionCurve.h"
    #endif
    #ifndef TABWATERRETENTIONDATA_H
        #include "tabWaterRetentionData.h"
    #endif

    #include <QWidget>
    #include <QComboBox>
    #include <QTextEdit>
    #include <QSqlDatabase>


    class Crit3DSoilWidget : public QWidget
    {
        Q_OBJECT

        public:
            Crit3DSoilWidget();

            void setDbSoil(QSqlDatabase dbOpened, QString soilCode);

        private:
            QComboBox soilListComboBox;
            QTabWidget* tabWidget;
            TabHorizons* horizonsTab;
            TabWaterRetentionData* wrDataTab;
            TabWaterRetentionCurve* wrCurveTab;
            TabHydraulicConductivityCurve* hydraConducCurveTab;

            QSqlDatabase dbSoil;
            soil::Crit3DSoil mySoil;
            soil::Crit3DSoil savedSoil;
            std::vector<soil::Crit3DTextureClass> textureClassList;
            std::vector<soil::Crit3DGeotechnicsClass> geotechnicsClassList;
            soil::Crit3DFittingOptions fittingOptions;
            int dbSoilType;

            QGroupBox *infoGroup;
            QLineEdit* soilNameValue;
            QLineEdit* satValue;
            QLineEdit* fcValue;
            QLineEdit* wpValue;
            QLineEdit* awValue;
            QLineEdit* potFCValue;
            QAction* restoreData;
            QAction* saveChanges;
            QAction* addHorizon;
            QAction* deleteHorizon;
            QAction* useWaterRetentionData;
            QAction* airEntryFixed;
            QAction* parameterRestriction;
            QAction* exportEstimatedParamTable;
            QAction* copyEstimatedParamTable;
            QAction* exportParamFromDbTable;
            QAction* copyParamFromDbTable;

            QPixmap pic;
            QString picPath;
            QPainter painter;
            QLabel *labelPic;
            bool changed;
            // USDA textural triangle size inside picture pic
            constexpr static const double widthTriangle = 362.0;
            constexpr static const double heightTriangle = 314.0;

            void setFittingMenu();

            void on_actionOpenSoilDB();
            void on_actionSave();
            void on_actionExportParamFromDbTable();
            void on_actionExportEstimatedParamTable();
            void on_actionNewSoil();
            void on_actionDeleteSoil();
            void on_actionCopyParamFromDbTable();
            void on_actionCopyEstimatedParamTable();
            void on_actionUseWaterRetentionData();
            void on_actionAirEntry();
            void on_actionParameterRestriction();
            void on_actionChooseSoil(QString);
            void on_actionAddHorizon();
            void on_actionDeleteHorizon();
            void on_actionRestoreData();
            void tabChanged(int index);
            void cleanInfoGroup();

            private slots:
            void setInfoTextural(int);
            void updateAll();
            void updateByTabWR();
    };

#endif // SOILWIDGET_H
