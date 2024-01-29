#ifndef TABHORIZONS_H
#define TABHORIZONS_H

    #ifndef SOIL_H
        #include "soil.h"
    #endif
    #ifndef SOILTABLE_H
        #include "soilTable.h"
    #endif
    #ifndef BARHORIZON_H
        #include "barHorizon.h"
    #endif
    #include <QtWidgets>

    class TabHorizons : public QWidget
    {
        Q_OBJECT

    public:
        TabHorizons();
        void insertSoilHorizons(soil::Crit3DSoil* mySoil, std::vector<soil::Crit3DTextureClass> *textureClassList,
                                std::vector<soil::Crit3DGeotechnicsClass> *geotechnicsClassList,
                                soil::Crit3DFittingOptions *fittingOptions);
        void updateTableModel(soil::Crit3DSoil *soil);
        bool checkDepths();
        bool checkHorizonData(int horizonNum);
        void checkMissingItem(int horizonNr);
        void checkComputedValues(int horizonNum);
        void setInvalidTableModelRow(int horizonNum);
        void clearSelections();
        void tableDbVerticalHeaderClick(int index);
        void cellChanged(int row, int column);
        void cellClickedDb(int row, int column);
        void cellClickedModel(int row, int column);
        void addRowClicked();
        void removeRowClicked();
        void resetAll();
        bool getInsertSoilElement() const;
        void setInsertSoilElement(bool value);
        void updateBarHorizon(soil::Crit3DSoil* mySoil);
        void copyTableDb();
        void copyTableModel();
        void exportTableDb(QString csvFile);
        void exportTableModel(QString csvFile);

    private:
        Crit3DSoilTable* tableDb;
        Crit3DSoilTable* tableModel;
        BarHorizonList barHorizons;
        QPushButton* addRow;
        QPushButton* deleteRow;
        soil::Crit3DSoil* mySoil;
        std::vector<soil::Crit3DTextureClass>* myTextureClassList;
        std::vector<soil::Crit3DGeotechnicsClass>* myGeotechnicsClassList;
        soil::Crit3DFittingOptions* myFittingOptions;
        bool insertSoilElement;
    private slots:
        void widgetClicked(int index);

    signals:
        void horizonSelected(int nHorizon);
        void updateSignal();


    };

#endif // TABHORIZONS_H
