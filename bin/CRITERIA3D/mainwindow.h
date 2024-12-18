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
    #include "project3D.h"
    #include "mapGraphicsRasterUtm.h"

    #include <QMainWindow>

    class QActionGroup;

    enum visualizationType {notShown, showLocation, showCurrentVariable, showElaboration};

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

        void callNewMeteoWidget(std::string id, std::string name, std::string dataset, double altitude, std::string lapseRateCode, bool isGrid);
        void callAppendMeteoWidget(std::string id, std::string name, std::string dataset, double altitude, std::string lapseRateCode, bool isGrid);

        void on_dateEdit_dateChanged(const QDate &date);
        void on_timeEdit_valueChanged(int myHour);
        void on_dayBeforeButton_clicked();
        void on_dayAfterButton_clicked();

        void on_variableButton_clicked();
        void on_opacitySliderRasterInput_sliderMoved(int position);
        void on_opacitySliderRasterOutput_sliderMoved(int position);

        // Menu File
        void on_actionOpenProject_triggered();
        void on_actionCloseProject_triggered();
        void on_actionLoad_DEM__triggered();
        void on_actionExtract_sub_basin_triggered();
        void on_actionLoad_soil_map_triggered();
        void on_actionLoad_soil_data_triggered();
        void on_actionLoad_MeteoPoints_triggered();
        void on_actionMeteoPointsImport_data_triggered();
        void on_actionNew_meteoPointsDB_from_csv_triggered();

        // Menu Show
        void on_actionShow_3D_viewer_triggered();
        void on_viewer3DClosed();
        void on_slopeChanged();

        void on_flagView_not_active_points_toggled(bool state);
        void on_actionView_PointsHide_triggered();
        void on_actionView_PointsLocation_triggered();
        void on_actionView_PointsCurrentVariable_triggered();

        void on_flagView_values_toggled(bool arg1);

        void on_actionView_SoilMap_triggered();
        void on_actionHide_Soil_map_triggered();

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
        void on_actionView_MeteoVariable_None_triggered();
        void on_actionView_Radiation_None_triggered();

        void on_actionView_Snow_water_equivalent_triggered();
        void on_actionView_Snow_surface_temperature_triggered();
        void on_actionView_Snow_internal_energy_triggered();
        void on_actionView_Snow_fall_triggered();
        void on_actionView_Snow_surface_internal_energy_triggered();
        void on_actionView_Snow_liquid_water_content_triggered();
        void on_actionView_Snow_age_triggered();
        void on_actionView_Snowmelt_triggered();

        void on_actionView_Snow_sensible_heat_triggered();
        void on_actionView_Snow_latent_heat_triggered();

        void on_actionView_Crop_LAI_triggered();
        void on_actionView_Crop_degreeDays_triggered();

        void on_actionView_Factor_of_safety_triggered();
        void on_actionView_Factor_of_safety_minimum_triggered();

        void on_actionView_DegreeOfSaturation_automatic_range_triggered();
        void on_actionView_DegreeOfSaturation_fixed_range_triggered();

        void on_actionView_SurfaceWaterContent_automatic_range_triggered();
        void on_actionView_SurfaceWaterContent_fixed_range_triggered();

        // menu meteo points
        void on_actionPoints_clear_selection_triggered();
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
        void on_actionPoints_delete_data_not_active_triggered();
        void on_actionPoints_deactivate_with_no_data_triggered();

        // Menu data spatialization
        void on_actionVariableQualitySpatial_triggered();
        void on_actionInterpolationSettings_triggered();
        void on_actionProxy_analysis_triggered();

        void on_actionTopographicDistanceMapWrite_triggered();
        void on_actionTopographicDistanceMapLoad_triggered();

        void on_actionComputePeriod_meteoVariables_triggered();
        void on_actionComputeHour_meteoVariables_triggered();

        // Menu soalr radiation
        void on_actionRadiation_settings_triggered();
        void on_actionRadiation_compute_current_hour_triggered();
        void on_actionRadiation_run_model_triggered();

        // menu 3D model
        void on_actionSnow_settings_triggered();
        void on_actionCriteria3D_Initialize_triggered();
        void on_actionCriteria3D_compute_next_hour_triggered();
        void on_actionCriteria3D_run_models_triggered();
        void on_actionCriteria3D_update_subHourly_triggered(bool isChecked);
        void on_actionCriteria3D_load_state_triggered();
        void on_actionCriteria3D_load_external_state_triggered();
        void on_actionCriteria3D_save_state_triggered();

        void on_flagSave_state_daily_step_toggled(bool isChecked);
        void on_flagSave_state_endRun_triggered(bool isChecked);

        void on_buttonModelPause_clicked();
        void on_buttonModelStop_clicked();
        void on_buttonModelStart_clicked();

        // menu output points
        void on_actionOutputPoints_clear_selection_triggered();
        void on_actionOutputPoints_deactivate_all_triggered();
        void on_actionOutputPoints_deactivate_selected_triggered();
        void on_flagHide_outputPoints_toggled(bool isChecked);
        void on_flagView_not_active_outputPoints_toggled(bool isChecked);
        void on_actionOutputPoints_activate_all_triggered();
        void on_actionOutputPoints_activate_selected_triggered();
        void on_actionOutputPoints_newFile_triggered();
        void on_actionOutputDB_new_triggered();
        void on_actionOutputDB_open_triggered();
        void on_actionOutputPoints_delete_selected_triggered();
        void on_flagOutputPoints_save_output_toggled(bool isChecked);
        void on_flagCompute_only_points_toggled(bool isChecked);
        void on_actionLoad_OutputPoints_triggered();
        void on_actionOutputPoints_add_triggered();

        // Menu settings
        void on_actionMapOpenStreetMap_triggered();
        void on_actionMapESRISatellite_triggered();
        void on_actionMapTerrain_triggered();
        void on_actionMapGoogle_hybrid_satellite_triggered();
        void on_actionMapGoogle_satellite_triggered();
        void on_actionProjectSettings_triggered();


        void on_actionLoad_land_use_map_triggered();

        void on_actionHide_LandUseMap_triggered();

        void on_actionView_LandUseMap_triggered();

        void on_actionHide_Geomap_triggered();

        void on_actionLoad_crop_data_triggered();

        void on_actionView_SoilMoisture_triggered();

        void on_layerNrEdit_valueChanged(int layerIndex);

        void on_flag_increase_slope_triggered(bool isChecked);

        void on_actionView_Water_potential_triggered();

        void on_actionCreate_new_land_use_map_triggered();

        void on_actionCriteria3D_set_processes_triggered();

        void on_actionCriteria3D_waterFluxes_settings_triggered();

        void on_actionView_SurfacePond_triggered();

        void on_actionSave_outputRaster_triggered();

        void on_actionCriteria3D_Water_content_summary_triggered();

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

        RasterUtmObject* rasterDEM;
        RasterUtmObject* rasterOutput;

        QList<StationMarker*> meteoPointList;
        QList<SquareMarker*> outputPointList;
        QList<ArrowObject*> windVectorList;

        ColorLegend *inputRasterColorLegend;
        ColorLegend *outputRasterColorLegend;
        ColorLegend *meteoPointsLegend;

        RubberBand *rubberBand;

        QActionGroup *showPointsGroup;

        visualizationType currentPointsVisualization;
        criteria3DVariable current3DVariable;
        int current3DlayerIndex;

        bool view3DVariable;
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
        void updateModelTime();
        void resetMeteoPointMarkers();
        void redrawMeteoPoints(visualizationType myType, bool updateColorSCale);

        bool loadMeteoPointsDB_GUI(QString dbName);
        void setCurrentRasterInput(gis::Crit3DRasterGrid *myRaster);
        void setCurrentRasterOutput(gis::Crit3DRasterGrid *rasterPointer);
        void interpolateCurrentVariable();
        bool initializeViewer3D();
        void refreshViewer3D();

        bool checkMapVariable(bool isComputed);

        bool isSoil(QPoint mapPos);
        void openSoilWidget(QPoint mapPos);
        void showSoilMap();

        bool isLandUse(QPoint mapPos);
        void showLandUseMap();

        bool contextMenuRequested(QPoint localPos);

        void setInputRasterVisible(bool isVisible);
        void setOutputRasterVisible(bool isVisible);

        void addMeteoPoints();
        void drawWindVector(int i);
        void drawProject();
        void renderDEM();
        void drawMeteoPoints();
        void clearRaster_GUI();
        void clearMeteoPoints_GUI();

        void setMeteoVariable(meteoVariable myVar, gis::Crit3DRasterGrid *myGrid);
        void setOutputMeteoVariable(meteoVariable myVar, gis::Crit3DRasterGrid *myGrid);

        void showMeteoVariable(meteoVariable var);
        void showSnowVariable(meteoVariable var);
        void showCriteria3DVariable(criteria3DVariable var, int layerIndex, bool isFixedRange,
                                    bool isHideOutliers, double minimum, double maximum);

        bool setRadiationAsCurrentVariable();
        bool startModels(QDateTime firstTime, QDateTime lastTime);

        void testOutputPoints();
        void addOutputPointsGUI();
        void redrawOutputPoints();
        void resetOutputPointMarkers();
        void clearWindVectorObjects();
        void loadMeteoPointsDataSingleDay(const QDate &date, bool showInfo);

        void initializeCriteria3DInterface();
    };

    bool selectDates(QDateTime &firstTime, QDateTime &lastTime);


#endif // MAINWINDOW_H
