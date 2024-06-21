#include "commonConstants.h"
#include "gis.h"
#include "waterBalance.h"
#include "vine3DProject.h"
#include "utilities.h"
#include "spatialControl.h"
#include "dialogInterpolation.h"
#include "dialogRadiation.h"
#include "dialogSettings.h"
#include "dialogSelection.h"
#include "formTimePeriod.h"

#include "mainWindow.h"
#include "ui_mainWindow.h"


extern Vine3DProject myProject;

#define MAPBORDER 10
#define TOOLSWIDTH 270

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    this->showPoints = true;

    this->rubberBand = nullptr;

    // Set the MapGraphics Scene and View
    this->mapScene = new MapGraphicsScene(this);
    this->mapView = new MapGraphicsView(mapScene, this->ui->widgetMap);

    this->rasterLegend = new ColorLegend(this->ui->widgetColorLegendRaster);
    this->rasterLegend->resize(this->ui->widgetColorLegendRaster->size());

    this->meteoPointsLegend = new ColorLegend(this->ui->widgetColorLegendPoints);
    this->meteoPointsLegend->resize(this->ui->widgetColorLegendPoints->size());
    this->meteoPointsLegend->colorScale = myProject.meteoPointsColorScale;

    this->currentPointsVisualization = showNone;
    // show menu
    showPointsGroup = new QActionGroup(this);
    showPointsGroup->setExclusive(true);
    showPointsGroup->addAction(this->ui->actionShowPointsHide);
    showPointsGroup->addAction(this->ui->actionShowPointsLocation);
    showPointsGroup->addAction(this->ui->actionShowPointsVariable);
    showPointsGroup->setEnabled(false);

    // Set tiles source
    this->setMapSource(WebTileSource::GOOGLE_Terrain);

    // Set start size and position
    this->startCenter = new Position (myProject.gisSettings.startLocation.longitude, myProject.gisSettings.startLocation.latitude, 0.0);
    this->mapView->setZoomLevel(8);
    this->mapView->centerOn(startCenter->lonLat());
    connect(this->mapView, SIGNAL(zoomLevelChanged(quint8)), this, SLOT(updateMaps()));

    // Set raster objects
    this->rasterObj = new RasterObject(this->mapView);

    this->rasterObj->setOpacity(this->ui->rasterOpacitySlider->value() / 100.0);

    this->rasterObj->setColorLegend(this->rasterLegend);

    this->mapView->scene()->addObject(this->rasterObj);

    this->updateVariable();
    this->updateDateTime();

    this->setMouseTracking(true);
}


MainWindow::~MainWindow()
{
    delete rasterObj;
    delete rasterLegend;
    delete meteoPointsLegend;
    delete mapView;
    delete mapScene;
    delete ui;
}


void MainWindow::resizeEvent(QResizeEvent * event)
{
    Q_UNUSED(event)
        const int INFOHEIGHT = 40;

        ui->widgetMap->setGeometry(TOOLSWIDTH, 0, this->width()-TOOLSWIDTH, this->height() - INFOHEIGHT);
        mapView->resize(ui->widgetMap->size());

        ui->groupBoxVariable->move(MAPBORDER/2, this->height()/2
                              - ui->groupBoxVariable->height() - ui->groupBoxMeteoPoints->height() - MAPBORDER);
        ui->groupBoxVariable->resize(TOOLSWIDTH, ui->groupBoxVariable->height());

        ui->groupBoxMeteoPoints->move(MAPBORDER/2, ui->groupBoxVariable->y() + ui->groupBoxVariable->height() + MAPBORDER);
        ui->groupBoxMeteoPoints->resize(TOOLSWIDTH, ui->groupBoxMeteoPoints->height());

        ui->groupBoxRaster->move(MAPBORDER/2, ui->groupBoxMeteoPoints->y() + ui->groupBoxMeteoPoints->height() + MAPBORDER);
        ui->groupBoxRaster->resize(TOOLSWIDTH, ui->groupBoxRaster->height());

        // TODO sembrano non funzionare
        ui->widgetColorLegendRaster->resize(TOOLSWIDTH, ui->widgetColorLegendPoints->height());
        ui->widgetColorLegendPoints->resize(TOOLSWIDTH, ui->widgetColorLegendPoints->height());
}


void MainWindow::updateMaps()
{
    rasterObj->updateCenter();
}


void MainWindow::mouseReleaseEvent(QMouseEvent *event)
{
    Q_UNUSED(event);

    updateMaps();
}


void MainWindow::mouseDoubleClickEvent(QMouseEvent * event)
{
    QPoint mapPos = getMapPos(event->pos());
    if (! isInsideMap(mapPos)) return;

    Position newCenter = this->mapView->mapToScene(mapPos);
    this->ui->statusBar->showMessage(QString::number(newCenter.latitude()) + " " + QString::number(newCenter.longitude()));

    if (event->button() == Qt::LeftButton)
        this->mapView->zoomIn();
    else
        this->mapView->zoomOut();

    this->mapView->centerOn(newCenter.lonLat());
}


void MainWindow::mouseMoveEvent(QMouseEvent * event)
{
    QPoint mapPos = getMapPos(event->pos());
    if (! isInsideMap(mapPos)) return;

    Position geoPoint = this->mapView->mapToScene(mapPos);
    this->ui->statusBar->showMessage(QString::number(geoPoint.latitude()) + " " + QString::number(geoPoint.longitude()));

    if (rubberBand != nullptr && rubberBand->isActive)
    {
        QPoint widgetPos = mapPos + QPoint(MAPBORDER, MAPBORDER);
        rubberBand->setGeometry(QRect(rubberBand->getOrigin(), widgetPos).normalized());
    }
}


void MainWindow::mousePressEvent(QMouseEvent *event)
{
    QPoint mapPos = getMapPos(event->pos());
    if (! isInsideMap(mapPos)) return;

    if (event->button() == Qt::RightButton)
    {
        if (rubberBand != nullptr)
        {
            QPoint widgetPos = mapPos + QPoint(MAPBORDER, MAPBORDER);
            rubberBand->setOrigin(widgetPos);
            rubberBand->setGeometry(QRect(widgetPos, QSize()));
            rubberBand->isActive = true;
            rubberBand->show();
        }
    }
}


void MainWindow::on_rasterOpacitySlider_sliderMoved(int position)
{
    this->rasterObj->setOpacity(position / 100.0);
}


void MainWindow::on_actionMapTerrain_triggered()
{
    this->setMapSource(WebTileSource::GOOGLE_Terrain);
    ui->actionMapTerrain->setChecked(true);
    ui->actionMapOpenStreetMap->setChecked(false);
    ui->actionMapESRISatellite->setChecked(false);
}


void MainWindow::on_actionMapOpenStreetMap_triggered()
{
    this->setMapSource(WebTileSource::OPEN_STREET_MAP);
    ui->actionMapTerrain->setChecked(false);
    ui->actionMapOpenStreetMap->setChecked(true);
    ui->actionMapESRISatellite->setChecked(false);
}


void MainWindow::on_actionMapESRISatellite_triggered()
{
    this->setMapSource(WebTileSource::GOOGLE_Hybrid_Satellite);
    ui->actionMapTerrain->setChecked(false);
    ui->actionMapOpenStreetMap->setChecked(false);
    ui->actionMapESRISatellite->setChecked(true);
}


void MainWindow::renderDEM()
{
    setCurrentRaster(&(myProject.DEM));
    ui->labelRasterScale->setText(QString::fromStdString(getVariableString(noMeteoTerrain)));
    ui->rasterOpacitySlider->setEnabled(true);

    // center map
    gis::Crit3DGeoPoint* center = rasterObj->getRasterCenter();
    mapView->centerOn(qreal(center->longitude), qreal(center->latitude));

    // resize map
    float size = rasterObj->getRasterMaxSize();
    size = log2(1000.f/size);
    mapView->setZoomLevel(quint8(size));
    mapView->centerOn(qreal(center->longitude), qreal(center->latitude));

    updateMaps();
}


void MainWindow::drawMeteoPoints()
{
    resetMeteoPointMarkers();
    if (! myProject.meteoPointsLoaded || myProject.nrMeteoPoints == 0)
    {
        ui->groupBoxMeteoPoints->setEnabled(false);
        return;
    }

    addMeteoPoints();
    ui->groupBoxMeteoPoints->setEnabled(true);

    myProject.loadMeteoPointsData (myProject.getCurrentDate().addDays(-1), myProject.getCurrentDate(), true, true, true);

    showPointsGroup->setEnabled(true);
    currentPointsVisualization = showLocation;
    redrawMeteoPoints(currentPointsVisualization, true);

    updateDateTime();
}


void MainWindow::on_mnuFileOpenProject_triggered()
{
    QString myFileName = QFileDialog::getOpenFileName(this,tr("Open Project"), "", tr("Project files (*.ini)"));
    if (myFileName == "") return;

    myProject.loadVine3DProject(myFileName);

    if (myProject.DEM.isLoaded)
        renderDEM();

    if (myProject.meteoPointsLoaded)
    {
        if (myProject.getCurrentHour() == 24)
            myProject.setCurrentHour(23);
        drawMeteoPoints();
    }
}


void MainWindow::on_actionRun_models_triggered()
{
    if (! myProject.isProjectLoaded)
    {
        myProject.logError("Load a project before.");
        return;
    }

    QDateTime timeIni = myProject.getCurrentTime();
    QDateTime timeFin = timeIni.addSecs(3600);

    FormTimePeriod formTimePeriod(&timeIni, &timeFin);
    formTimePeriod.show();
    int myReturn = formTimePeriod.exec();
    if (myReturn == QDialog::Rejected) return;

    myProject.runModels(timeIni, timeFin, true);
}


QPoint MainWindow::getMapPos(const QPoint& pos)
{
    QPoint mapPos;
    int dx = ui->widgetMap->x();
    int dy = ui->widgetMap->y() + ui->menuBar->height();
    mapPos.setX(pos.x() - dx - MAPBORDER);
    mapPos.setY(pos.y() - dy - MAPBORDER);
    return mapPos;
}


bool MainWindow::isInsideMap(const QPoint& pos)
{
    if (pos.x() > 0 && pos.y() > 0 &&
        pos.x() < (mapView->width() - MAPBORDER*2) &&
        pos.y() < (mapView->height() - MAPBORDER*2) )
    {
        return true;
    }
    else return false;
}

void MainWindow::resetMeteoPointMarkers()
{
    for (int i = 0; i < meteoPointList.size(); i++)
    {
        mapView->scene()->removeObject(meteoPointList[i]);
        delete meteoPointList[i];
    }

    meteoPointList.clear();
}

void MainWindow::on_actionVariableQualitySpatial_triggered()
{
    myProject.checkSpatialQuality = ui->actionVariableQualitySpatial->isChecked();
    updateVariable();
}


void MainWindow::interpolateDemGUI()
{
    meteoVariable myVar = myProject.getCurrentVariable();
    if (myVar == noMeteoVar)
    {
        myProject.logError("Select a variable before.");
        return;
    }

    if (myProject.interpolationDemMain(myVar, myProject.getCrit3DCurrentTime(), &(myProject.dataRaster)))
    {
        setColorScale(myVar, myProject.dataRaster.colorScale);
        setCurrentRaster(&(myProject.dataRaster));

        ui->labelRasterScale->setText(QString::fromStdString(getVariableString(myVar)));
    }
    else
        myProject.logError();
}


void MainWindow::updateVariable()
{
    if (myProject.getCurrentVariable() != noMeteoVar)
    {
        this->ui->actionShowPointsLocation->setChecked(false);
    }

    //check
    if ((myProject.getCurrentVariable() == dailyAirTemperatureAvg)
            || (myProject.getCurrentVariable() == dailyAirTemperatureMax)
            || (myProject.getCurrentVariable() == dailyAirTemperatureMin))
        myProject.setCurrentVariable(airTemperature);

    else if ((myProject.getCurrentVariable() == dailyAirRelHumidityAvg)
             || (myProject.getCurrentVariable() == dailyAirRelHumidityMax)
             || (myProject.getCurrentVariable() == dailyAirRelHumidityMin))
         myProject.setCurrentVariable(airRelHumidity);

    else if (myProject.getCurrentVariable() == dailyPrecipitation)
            myProject.setCurrentVariable(precipitation);

    else if (myProject.getCurrentVariable() == dailyGlobalRadiation)
        myProject.setCurrentVariable(globalIrradiance);

    std::string myString = getVariableString(myProject.getCurrentVariable());
    ui->labelVariable->setText(QString::fromStdString(myString));

    redrawMeteoPoints(currentPointsVisualization, true);
}

void MainWindow::updateDateTime()
{
    int myHour = myProject.getCurrentHour();
    this->ui->dateEdit->setDate(myProject.getCurrentDate());
    this->ui->timeEdit->setTime(QTime(myHour,0,0));
}


void MainWindow::on_timeEdit_timeChanged(const QTime &time)
{
    //hour
    if (time.hour() != myProject.getCurrentHour())
    {
        myProject.setCurrentHour(time.hour());
    }

    redrawMeteoPoints(currentPointsVisualization, true);
}


void MainWindow::redrawMeteoPoints(visualizationType myType, bool updateColorSCale)
{
    currentPointsVisualization = myType;

    if (myProject.nrMeteoPoints == 0)
        return;

    // hide all meteo points
    for (int i = 0; i < myProject.nrMeteoPoints; i++)
        meteoPointList[i]->setVisible(false);

    meteoPointsLegend->setVisible(true);

    switch(currentPointsVisualization)
    {
        case showNone:
        {
            meteoPointsLegend->setVisible(false);
            this->ui->actionShowPointsHide->setChecked(true);
            break;
        }
        case showLocation:
        {
            this->ui->actionShowPointsLocation->setChecked(true);
            for (int i = 0; i < myProject.nrMeteoPoints; i++)
            {
                    myProject.meteoPoints[i].currentValue = NODATA;
                    meteoPointList[i]->setFillColor(QColor(Qt::white));
                    meteoPointList[i]->setRadius(5);
                    meteoPointList[i]->setCurrentValue(NODATA);
                    meteoPointList[i]->setToolTip();
                    meteoPointList[i]->setVisible(true);
            }

            myProject.meteoPointsColorScale->setRange(NODATA, NODATA);
            meteoPointsLegend->update();
            break;
        }
        case showCurrentVariable:
        {
            this->ui->actionShowPointsVariable->setChecked(true);

            // quality control
            std::string errorStdStr;
            checkData(myProject.quality, myProject.getCurrentVariable(),
                      myProject.meteoPoints, myProject.nrMeteoPoints, myProject.getCrit3DCurrentTime(),
                      &myProject.qualityInterpolationSettings, myProject.meteoSettings,
                      &(myProject.climateParameters), myProject.checkSpatialQuality, errorStdStr);

            if (updateColorSCale)
            {
                float minimum, maximum;
                myProject.getMeteoPointsRange(minimum, maximum, true);

                myProject.meteoPointsColorScale->setRange(minimum, maximum);
            }

            roundColorScale(myProject.meteoPointsColorScale, 4, true);
            setColorScale(myProject.getCurrentVariable(), myProject.meteoPointsColorScale);

            Crit3DColor *myColor;
            for (int i = 0; i < myProject.nrMeteoPoints; i++)
            {
                if (int(myProject.meteoPoints[i].currentValue) != NODATA)
                {
                    if (myProject.meteoPoints[i].quality == quality::accepted)
                    {
                        meteoPointList[i]->setRadius(5);
                        myColor = myProject.meteoPointsColorScale->getColor(myProject.meteoPoints[i].currentValue);
                        meteoPointList[i]->setFillColor(QColor(myColor->red, myColor->green, myColor->blue));
                        meteoPointList[i]->setOpacity(1.0);
                    }
                    else
                    {
                        // Wrong data
                        meteoPointList[i]->setRadius(10);
                        meteoPointList[i]->setFillColor(QColor(Qt::black));
                        meteoPointList[i]->setOpacity(0.5);
                    }

                    meteoPointList[i]->setCurrentValue(myProject.meteoPoints[i].currentValue);
                    meteoPointList[i]->setQuality(myProject.meteoPoints[i].quality);
                    meteoPointList[i]->setToolTip();
                    meteoPointList[i]->setVisible(true);
                }
            }

            meteoPointsLegend->update();
            break;
        }
    }
}

void MainWindow::addMeteoPoints()
{
    myProject.clearSelectedPoints();
    for (int i = 0; i < myProject.nrMeteoPoints; i++)
    {
        StationMarker* point = new StationMarker(5.0, true, QColor(Qt::white));

        point->setFlag(MapGraphicsObject::ObjectIsMovable, false);
        point->setLatitude(myProject.meteoPoints[i].latitude);
        point->setLongitude(myProject.meteoPoints[i].longitude);
        point->setId(myProject.meteoPoints[i].id);
        point->setName(myProject.meteoPoints[i].name);
        point->setDataset(myProject.meteoPoints[i].dataset);
        point->setAltitude(myProject.meteoPoints[i].point.z);
        point->setMunicipality(myProject.meteoPoints[i].municipality);
        point->setCurrentValue(myProject.meteoPoints[i].currentValue);
        point->setQuality(myProject.meteoPoints[i].quality);

        this->meteoPointList.append(point);
        this->mapView->scene()->addObject(this->meteoPointList[i]);

        point->setToolTip();
        connect(point, SIGNAL(newStationClicked(std::string, std::string, std::string, double, std::string, bool)), this, SLOT(callNewMeteoWidget(std::string, std::string, std::string, double, std::string, bool)));
        connect(point, SIGNAL(appendStationClicked(std::string, std::string, std::string, double, std::string, bool)), this, SLOT(callAppendMeteoWidget(std::string, std::string, std::string, double, std::string, bool)));
    }
}

void MainWindow::setMapSource(WebTileSource::WebTileType mySource)
{
    QSharedPointer<WebTileSource> myTiles(new WebTileSource(mySource), &QObject::deleteLater);

    this->mapView->setTileSource(myTiles);
}


void MainWindow::on_rasterScaleButton_clicked()
{
    if (this->rasterObj->getRaster() == nullptr)
    {
        QMessageBox::information(nullptr, "No Raster", "Load raster before");
        return;
    }

    meteoVariable myVar = chooseColorScale();
    if (myVar != noMeteoVar)
    {
        setColorScale(myVar, this->rasterObj->getRaster()->colorScale);
        ui->labelRasterScale->setText(QString::fromStdString(getVariableString(myVar)));
    }
}

void MainWindow::on_variableButton_clicked()
{
    meteoVariable myVar = chooseMeteoVariable(myProject);
    if (myVar == noMeteoVar) return;

    myProject.setCurrentVariable(myVar);
    this->updateVariable();

    if (myProject.getCurrentFrequency() != noFrequency)
    {
        this->ui->actionShowPointsVariable->setEnabled(true);
        redrawMeteoPoints(showCurrentVariable, true);
    }
}

void MainWindow::on_rasterRestoreButton_clicked()
{
    if (rasterObj->getRaster() == nullptr)
    {
        QMessageBox::information(nullptr, "No Raster", "Load raster before");
        return;
    }

    setColorScale(noMeteoTerrain, myProject.DEM.colorScale);
    setCurrentRaster(&(myProject.DEM));
    ui->labelRasterScale->setText(QString::fromStdString(getVariableString(noMeteoTerrain)));
}

void MainWindow::setCurrentRaster(gis::Crit3DRasterGrid *myRaster)
{
    rasterObj->initializeUTM(myRaster, myProject.gisSettings, false);
    rasterLegend->colorScale = myRaster->colorScale;
    rasterObj->redrawRequested();
}

void MainWindow::on_dateEdit_dateChanged(const QDate &date)
{
    if (date != myProject.getCurrentDate())
    {
        myProject.loadMeteoPointsData(date.addDays(-1), date, true, true, true);
        //myProject.loadMeteoGridData(date, date, true);
        myProject.setCurrentDate(date);
    }

    redrawMeteoPoints(currentPointsVisualization, true);
}

void MainWindow::on_actionInterpolation_to_DEM_triggered()
{
    if (! myProject.DEM.isLoaded)
    {
        myProject.logError("Load a project before.");
        return;
    }

    interpolateDemGUI();
}

void MainWindow::on_actionInterpolationSettings_triggered()
{
    DialogInterpolation* myInterpolationDialog = new DialogInterpolation(&myProject);
    myInterpolationDialog->close();
}

void MainWindow::on_actionParameters_triggered()
{
    DialogSettings* mySettingsDialog = new DialogSettings(&myProject);
    mySettingsDialog->exec();
    if ((startCenter->latitude() - myProject.gisSettings.startLocation.latitude) > EPSILON || (startCenter->longitude() - myProject.gisSettings.startLocation.longitude) > EPSILON)
    {
        startCenter->setLatitude(myProject.gisSettings.startLocation.latitude);
        startCenter->setLongitude(myProject.gisSettings.startLocation.longitude);
        this->mapView->centerOn(startCenter->lonLat());
    }

    mySettingsDialog->close();
}

void MainWindow::on_actionShow_DEM_triggered()
{
    if (myProject.DEM.isLoaded)
    {
        setColorScale(noMeteoTerrain, myProject.DEM.colorScale);
        this->setCurrentRaster(&(myProject.DEM));
        ui->labelRasterScale->setText(QString::fromStdString(getVariableString(noMeteoTerrain)));
    }
    else
    {
        myProject.logError("Load a Digital Elevation Model before.");
        return;
    }
}

void MainWindow::on_actionShow_boundary_triggered()
{
    if (myProject.boundaryMap.isLoaded)
        {
            setColorScale(noMeteoTerrain, myProject.boundaryMap.colorScale);
            this->setCurrentRaster(&(myProject.boundaryMap));
            ui->labelRasterScale->setText("Boundary map");
        }
        else
        {
            myProject.logError("Initialize model before.");
            return;
        }
}


void MainWindow::on_actionVine3D_InitializeWaterBalance_triggered()
{
    if (! myProject.setVine3DSoilIndexMap()) return;

    if (myProject.initializeWaterBalance3D())
    {
        myProject.outputWaterBalanceMaps = new Crit3DWaterBalanceMaps(myProject.DEM);
        QMessageBox::information(nullptr, "", "3D water fluxes initialized.");
    }
}


void MainWindow::on_actionShowPointsHide_triggered()
{
    redrawMeteoPoints(showNone, true);
}

void MainWindow::on_actionShowPointsLocation_triggered()
{
    redrawMeteoPoints(showLocation, true);
}

void MainWindow::on_actionShowPointsVariable_triggered()
{
    redrawMeteoPoints(showCurrentVariable, true);
}

void MainWindow::on_actionRadiation_settings_triggered()
{
    DialogRadiation* myDialogRadiation = new DialogRadiation(&myProject);
    myDialogRadiation->close();
}

void MainWindow::callNewMeteoWidget(std::string id, std::string name, std::string dataset, double altitude, std::string lapseRateCode, bool isGrid)
{
    bool isAppend = false;
    if (isGrid)
    {
        myProject.showMeteoWidgetGrid(id, isAppend);
    }
    else
    {
        myProject.showMeteoWidgetPoint(id, name, dataset, altitude, lapseRateCode, isAppend);
    }
    return;
}


void MainWindow::callAppendMeteoWidget(std::string id, std::string name, std::string dataset, double altitude, std::string lapseRateCode, bool isGrid)
{
    bool isAppend = true;
    if (isGrid)
    {
        myProject.showMeteoWidgetGrid(id, isAppend);
    }
    else
    {
        myProject.showMeteoWidgetPoint(id, name, dataset, altitude, lapseRateCode, isAppend);
    }
    return;
}

