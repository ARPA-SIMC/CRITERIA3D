#ifndef MAINWINDOW_H
#define MAINWINDOW_H

    #include "soilWidget.h"
    #include "tileSources/WebTileSource.h"
    #include "Position.h"
    #include "MapGraphicsView.h"
    #include "MapGraphicsScene.h"

    #include "mapGraphicsRasterObject.h"
    #include "stationMarker.h"
    #include "colorLegend.h"
    #include "viewer3d.h"

    #include <QMainWindow>

    class QActionGroup;

    enum visualizationType {notShown, showLocation, showCurrentVariable, showElaboration,
                        showAnomalyAbsolute, showAnomalyPercentage, showClimate};

    namespace Ui
    {
        class MainWindow;
    }

    class MainWindow : public QMainWindow
    {
        Q_OBJECT

    public:

        explicit MainWindow(QWidget *parent = nullptr);

    private slots:

        void on_actionOpenProject_triggered();
        void on_actionCloseProject_triggered();
        void on_actionLoad_DEM_triggered();
        void on_actionLoad_soil_map_triggered();
        void on_actionLoad_soil_data_triggered();
        void on_actionLoad_Crop_data_triggered();
        void on_actionLoad_MeteoPoints_triggered();
        void on_actionMeteoPointsImport_data_triggered();

        void on_dateEdit_dateChanged(const QDate &date);
        void on_timeEdit_valueChanged(int myHour);

        void on_variableButton_clicked();
        void on_opacitySliderRasterInput_sliderMoved(int position);
        void on_opacitySliderRasterOutput_sliderMoved(int position);

        void on_actionProjectSettings_triggered();
        void on_actionVariableQualitySpatial_triggered();
        void on_actionInterpolationSettings_triggered();
        void on_actionRadiationSettings_triggered();

        void on_actionCompute_solar_radiation_triggered();
        void on_actionCompute_AllMeteoMaps_triggered();

        void on_actionView_3D_triggered();
        void on_actionView_SoilMap_triggered();
        void on_actionHide_soil_map_triggered();
        void on_actionView_Boundary_triggered();
        void on_actionView_Slope_triggered();
        void on_actionView_Aspect_triggered();
        void on_actionView_PointsHide_triggered();
        void on_actionView_PointsLocation_triggered();
        void on_actionView_PointsCurrentVariable_triggered();

        void on_actionView_Transmissivity_triggered();
        void on_actionView_Global_radiation_triggered();
        void on_actionView_Air_temperature_triggered();
        void on_actionView_Precipitation_triggered();
        void on_actionView_Air_relative_humidity_triggered();
        void on_actionView_Wind_intensity_triggered();
        void on_actionView_ET0_triggered();
        void on_actionView_None_triggered();
        void on_actionViewMeteoVariable_None_triggered();

        void on_actionMapOpenStreetMap_triggered();
        void on_actionMapESRISatellite_triggered();
        void on_actionMapTerrain_triggered();
        void on_actionMapGoogle_hybrid_satellite_triggered();

        void on_actionCriteria3D_Initialize_triggered();
        void on_viewer3DClosed();

        void on_actionRun_models_triggered();

        void updateMaps();
        void updateGUI();
        void mouseMove(const QPoint &eventPos);

        void callNewMeteoWidget(std::string id, std::string name, bool isGrid);
        void callAppendMeteoWidget(std::string id, std::string name, bool isGrid);

        void on_actionNew_meteoPointsDB_from_csv_triggered();

    protected:
        /*!
         * \brief mouseReleaseEvent call moveCenter
         * \param event
         */
        void mouseReleaseEvent(QMouseEvent *event);

        /*!
         * \brief mouseDoubleClickEvent implements zoom In and zoom Out
         * \param event
         */
        void mouseDoubleClickEvent(QMouseEvent * event);

        // void mouseMoveEvent(QMouseEvent * event);

        void mousePressEvent(QMouseEvent *event);

        void resizeEvent(QResizeEvent * event);

    private:
        Ui::MainWindow* ui;

        Position* startCenter;
        MapGraphicsScene* mapScene;
        MapGraphicsView* mapView;

        RasterObject* rasterDEM;
        RasterObject* rasterOutput;
        QList<StationMarker*> pointList;

        ColorLegend *inputRasterColorLegend;
        ColorLegend *outputRasterColorLegend;
        ColorLegend *meteoPointsLegend;

        QActionGroup *showPointsGroup;

        visualizationType currentPointsVisualization;

        Viewer3D *viewer3D;
        Crit3DSoilWidget *soilWidget;

        void setTileMapSource(WebTileSource::WebTileType tileSource);
        void setProjectTileMap();

        QPoint getMapPos(const QPoint& pos);
        bool isInsideMap(const QPoint& pos);

        void updateVariable();
        void updateDateTime();
        void resetMeteoPoints();
        void redrawMeteoPoints(visualizationType myType, bool updateColorSCale);

        bool loadMeteoPointsDB(QString dbName);
        bool loadMeteoGridDB(QString xmlName);
        void setCurrentRasterInput(gis::Crit3DRasterGrid *myRaster);
        void setCurrentRasterOutput(gis::Crit3DRasterGrid *myRaster);
        void interpolateDemGUI();
        bool initializeViewer3D();
        bool checkMapVariable(bool isComputed);

        void openSoilWidget(QPoint mapPos);
        void contextMenuRequested(QPoint localPos, QPoint globalPos);

        void setInputRasterVisible(bool value);
        void setOutputRasterVisible(bool value);

        void addMeteoPoints();
        void drawProject();
        void renderDEM();
        void clearDEM_GUI();
        void drawMeteoPoints();
        void clearMeteoPoints_GUI();

        void setMeteoVariable(meteoVariable myVar, gis::Crit3DRasterGrid *myGrid);
        void showMeteoVariable(meteoVariable var);

        bool runModels(QDateTime dateTime1, QDateTime dateTime2, bool saveOutput, bool saveState);
    };


#endif // MAINWINDOW_H
