#ifndef MAINWINDOW_H
#define MAINWINDOW_H

    #include <QMainWindow>
    #include <QActionGroup>

    #include "tileSources/WebTileSource.h"
    #include "Position.h"
    #include "MapGraphicsView.h"
    #include "rubberBand.h"
    #include "MapGraphicsScene.h"
    #include "stationMarker.h"
    #include "mapGraphicsRasterObject.h"
    #include "colorLegend.h"

    enum visualizationType {showNone, showLocation, showCurrentVariable};

    namespace Ui
    {
        class MainWindow;
    }

    /*!
     * \brief The MainWindow class
     */
    class MainWindow : public QMainWindow
    {
        Q_OBJECT

    public:

        explicit MainWindow(QWidget *parent = nullptr);
        ~MainWindow();


    private slots:

        void callNewMeteoWidget(std::string id, std::string name, std::string dataset, double altitude, std::string lapseRateCode, bool isGrid);
        void callAppendMeteoWidget(std::string id, std::string name, std::string dataset, double altitude, std::string lapseRateCode, bool isGrid);

        void on_mnuFileOpenProject_triggered();

        void on_actionShowPointsHide_triggered();
        void on_actionShowPointsLocation_triggered();
        void on_actionShowPointsVariable_triggered();

        void on_actionVariableQualitySpatial_triggered();

        void on_rasterOpacitySlider_sliderMoved(int position);
        void on_rasterScaleButton_clicked();

        void on_actionMapOpenStreetMap_triggered();
        void on_actionMapESRISatellite_triggered();
        void on_actionMapTerrain_triggered();

        void on_variableButton_clicked();

        void on_rasterRestoreButton_clicked();
        void on_timeEdit_timeChanged(const QTime &time);
        void on_dateEdit_dateChanged(const QDate &date);

        void on_actionInterpolation_to_DEM_triggered();
        void on_actionInterpolationSettings_triggered();

        void on_actionShow_boundary_triggered();
        void on_actionShow_DEM_triggered();

        void on_actionVine3D_InitializeWaterBalance_triggered();
        void on_actionParameters_triggered();

        void on_actionRun_models_triggered();
        void on_actionRadiation_settings_triggered();

        void updateMaps();

        void on_actionShow_model_cases_map_triggered();

        void on_actionCriteria3D_settings_triggered();

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
        void mouseMoveEvent(QMouseEvent * event);
        void mousePressEvent(QMouseEvent *event);
        void resizeEvent(QResizeEvent * event);

    private:
        Ui::MainWindow* ui;

        Position* startCenter;
        MapGraphicsScene* mapScene;
        MapGraphicsView* mapView;
        RasterObject* rasterObj;
        ColorLegend *rasterLegend;
        ColorLegend *meteoPointsLegend;
        QList<StationMarker*> meteoPointList;
        RubberBand *rubberBand;

        visualizationType currentPointsVisualization;
        QActionGroup *showPointsGroup;

        bool showPoints;

        void setMapSource(WebTileSource::WebTileType mySource);

        QPoint getMapPos(const QPoint& pos);
        bool isInsideMap(const QPoint& pos);

        void updateVariable();
        void updateDateTime();
        void resetMeteoPointMarkers();
        void addMeteoPoints();

        void redrawMeteoPoints(visualizationType myType, bool updateColorSCale);

        void renderDEM();
        void drawMeteoPoints();

        bool loadMeteoPointsDB(QString dbName);
        void setCurrentRaster(gis::Crit3DRasterGrid *myRaster);
        void interpolateDemGUI();
    };


#endif // MAINWINDOW_H
