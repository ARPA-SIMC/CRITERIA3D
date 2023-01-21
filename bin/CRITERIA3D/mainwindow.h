#ifndef MAINWINDOW_H
#define MAINWINDOW_H

    #include "soilWidget.h"
    #include "tileSources/WebTileSource.h"
    #include "Position.h"
    #include "MapGraphicsView.h"
    #include "MapGraphicsScene.h"
    #include "mapGraphicsRasterObject.h"
    #include "stationMarker.h"
    #include "squareMarker.h"
    #include "colorLegend.h"
    #include "rubberBand.h"
    #include "viewer3D.h"
    #include "ArrowObject.h"

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

        void mouseMove(QPoint eventPos);
        void updateMaps();
        void updateOutputMap();

        void on_actionOpenProject_triggered();
        void on_actionCloseProject_triggered();
        void on_actionLoad_DEM_triggered();
        void on_actionLoad_soil_map_triggered();
        void on_actionLoad_soil_data_triggered();
        void on_actionLoad_Crop_data_triggered();
        void on_actionLoad_MeteoPoints_triggered();
        void on_actionMeteoPointsImport_data_triggered();

        void on_actionNew_meteoPointsDB_from_csv_triggered();

        void on_dateEdit_dateChanged(const QDate &date);
        void on_timeEdit_valueChanged(int myHour);
        void on_dayBeforeButton_clicked();
        void on_dayAfterButton_clicked();

        void on_variableButton_clicked();
        void on_opacitySliderRasterInput_sliderMoved(int position);
        void on_opacitySliderRasterOutput_sliderMoved(int position);

        void on_flagView_not_active_points_toggled(bool state);
        void on_actionView_PointsHide_triggered();
        void on_actionView_PointsLocation_triggered();
        void on_actionView_PointsCurrentVariable_triggered();

        void on_actionView_Boundary_triggered();
        void on_actionView_Slope_triggered();
        void on_actionView_Aspect_triggered();

        void on_actionView_Transmissivity_triggered();
        void on_actionView_Global_irradiance_triggered();
        void on_actionView_Beam_irradiance_triggered();
        void on_actionView_Diffuse_irradiance_triggered();
        void on_actionView_Reflected_irradiance_triggered();
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
        void on_actionMapGoogle_satellite_triggered();

        void on_actionProjectSettings_triggered();

        void on_actionVariableQualitySpatial_triggered();
        void on_actionInterpolationSettings_triggered();
        void on_actionProxy_analysis_triggered();
        void on_actionComputePeriod_meteoVariables_triggered();
        void on_actionComputeHour_meteoVariables_triggered();

        void callNewMeteoWidget(std::string id, std::string name, bool isGrid);
        void callAppendMeteoWidget(std::string id, std::string name, bool isGrid);

        void on_actionRadiation_settings_triggered();
        void on_actionRadiation_compute_current_hour_triggered();
        void on_actionRadiation_run_model_triggered();

        void on_actionSnow_settings_triggered();
        void on_actionSnow_initialize_triggered();
        void on_actionSnow_compute_current_hour_triggered();
        void on_actionSnow_run_model_triggered();

        void on_actionView_Snow_water_equivalent_triggered();
        void on_actionView_Snow_surface_temperature_triggered();
        void on_actionView_Snow_internal_energy_triggered();
        void on_actionView_Snow_fall_triggered();
        void on_actionView_Snow_surface_internal_energy_triggered();
        void on_actionView_Snow_liquid_water_content_triggered();
        void on_actionView_Snow_age_triggered();
        void on_actionView_Snowmelt_triggered();

        void on_actionSave_state_triggered();
        void on_actionLoad_state_triggered();
        void on_flagSave_state_daily_step_toggled(bool isChecked);

        void on_actionCriteria3D_settings_triggered();
        void on_actionCriteria3D_Initialize_triggered();
        void on_actionCriteria3D_run_models_triggered();

        void on_buttonModelPause_clicked();
        void on_buttonModelStop_clicked();
        void on_buttonModelStart_clicked();

        // menu meteo points
        void on_actionDelete_Points_Selected_triggered();
        void on_actionDelete_Points_NotActive_triggered();
        void on_actionPoints_activate_all_triggered();
        void on_actionPoints_deactivate_all_triggered();
        void on_actionPoints_activate_selected_triggered();
        void on_actionPoints_deactivate_selected_triggered();
        void on_actionPoints_activate_from_point_list_triggered();

        void on_actionPoints_deactivate_from_point_list_triggered();

        void on_actionPoints_activate_with_criteria_triggered();

        void on_actionPoints_deactivate_with_criteria_triggered();

        void on_actionPoints_delete_data_selected_triggered();

        void on_actionPoints_clear_selection_triggered();

        void on_actionPoints_delete_data_not_active_triggered();

        void on_actionPoints_deactivate_with_no_data_triggered();

        void on_actionOutputPoints_clear_selection_triggered();

        void on_actionOutputPoints_deactivate_all_triggered();

        void on_actionOutputPoints_deactivate_selected_triggered();

        void on_actionView_SoilMap_triggered();

        void on_flagHide_outputPoints_toggled(bool isChecked);

        void on_flagView_not_active_outputPoints_toggled(bool isChecked);

        void on_actionOutputPoints_activate_all_triggered();

        void on_actionOutputPoints_activate_selected_triggered();

        void on_actionHide_soil_map_triggered();

        void on_actionOutputPoints_newFile_triggered();

        void on_actionOutputDB_new_triggered();

        void on_actionOutputDB_open_triggered();

        void on_actionOutputPoints_delete_selected_triggered();

        void on_flagOutputPoints_save_output_toggled(bool isChecked);

        void on_flagCompute_only_points_toggled(bool isChecked);

        void on_actionLoad_OutputPoints_triggered();

        void on_actionOutputPoints_add_triggered();

        void on_flagView_values_toggled(bool arg1);

        void on_actionView_Snow_sensible_heat_triggered();

        void on_actionView_Snow_latent_heat_triggered();

        void on_actionLoad_external_state_triggered();

        void on_actionTopographicDistanceMapWrite_triggered();

        void on_actionTopographicDistanceMapLoad_triggered();

        void on_actionCriteria3D_compute_current_hour_triggered();

        void on_actionShow_3D_viewer_triggered();

        void on_viewer3DClosed();
        void on_slopeChanged();

    protected:
        /*!
         * \brief mouseReleaseEvent call moveCenter
         * \param event
         */
        void mouseReleaseEvent(QMouseEvent *event) override;

        /*!
         * \brief mouseDoubleClickEvent implements zoom In and zoom Out
         * \param event
         */
        void mouseDoubleClickEvent(QMouseEvent * event) override;

        void mousePressEvent(QMouseEvent *event) override;

        void resizeEvent(QResizeEvent * event) override;

    private:
        Ui::MainWindow* ui;

        Viewer3D* viewer3D;

        Position* startCenter;
        MapGraphicsScene* mapScene;
        MapGraphicsView* mapView;

        RasterObject* rasterDEM;
        RasterObject* rasterOutput;
        QList<StationMarker*> meteoPointList;
        QList<SquareMarker*> outputPointList;
        QList<ArrowObject*> windVectorList;

        ColorLegend *inputRasterColorLegend;
        ColorLegend *outputRasterColorLegend;
        ColorLegend *meteoPointsLegend;

        RubberBand *rubberBand;

        QActionGroup *showPointsGroup;

        visualizationType currentPointsVisualization;
        bool viewNotActivePoints;
        bool viewOutputPoints;
        bool viewNotActiveOutputPoints;

        Crit3DSoilWidget *soilWidget;

        void setTileMapSource(WebTileSource::WebTileType tileSource);
        void setProjectTileMap();

        QPoint getMapPos(const QPoint& pos);
        bool isInsideMap(const QPoint& pos);

        bool updateSelection(const QPoint& position);
        void updateCurrentVariable();
        void updateDateTime();
        void resetMeteoPointMarkers();
        void redrawMeteoPoints(visualizationType myType, bool updateColorSCale);

        bool loadMeteoPointsDB_GUI(QString dbName);
        void setCurrentRasterInput(gis::Crit3DRasterGrid *myRaster);
        void setCurrentRasterOutput(gis::Crit3DRasterGrid *myRaster);
        void interpolateCurrentVariable();
        bool initializeViewer3D();
        bool checkMapVariable(bool isComputed);

        bool isSoil(QPoint mapPos);
        void openSoilWidget(QPoint mapPos);
        void showSoilMap();
        bool contextMenuRequested(QPoint localPos, QPoint globalPos);

        void setInputRasterVisible(bool value);
        void setOutputRasterVisible(bool value);

        void addMeteoPoints();
        void drawWindVector(int i);
        void drawProject();
        void renderDEM();
        void drawMeteoPoints();
        void clearMaps_GUI();
        void clearMeteoPoints_GUI();

        void setMeteoVariable(meteoVariable myVar, gis::Crit3DRasterGrid *myGrid);
        void setOutputVariable(meteoVariable myVar, gis::Crit3DRasterGrid *myGrid);

        void showMeteoVariable(meteoVariable var);
        void showSnowVariable(meteoVariable var);

        bool setRadiationAsCurrentVariable();
        bool startModels(QDateTime firstTime, QDateTime lastTime);
        bool runModels(QDateTime firstTime, QDateTime lastTime);

        void testOutputPoints();
        void addOutputPointsGUI();
        void redrawOutputPoints();
        void resetOutputPointMarkers();
        void clearWindVectorObjects();
        void loadMeteoPointsDataSingleDay(const QDate &date, bool showInfo);
    };

    bool selectDates(QDateTime &firstTime, QDateTime &lastTime);


#endif // MAINWINDOW_H
