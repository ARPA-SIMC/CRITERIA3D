#include "ui_mainwindow.h"
#include "mainwindow.h"

#include "commonConstants.h"
#include "basicMath.h"
#include "soilDbTools.h"
#include "dialogSelection.h"
#include "spatialControl.h"
#include "dialogInterpolation.h"
#include "dialogSettings.h"
#include "dialogRadiation.h"
#include "dialogPointProperties.h"
#include "utilities.h"
#include "formPeriod.h"
#include "criteria3DProject.h"
#include "dialogSnowSettings.h"
#include "dialogLoadState.h"

#include <QDebug>


extern Crit3DProject myProject;

#define MAPBORDER 10
#define TOOLSWIDTH 270


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    //this->viewer3D = nullptr;

    // Set the MapGraphics Scene and View
    this->mapScene = new MapGraphicsScene(this);
    this->mapView = new MapGraphicsView(mapScene, this->ui->widgetMap);


    this->inputRasterColorLegend = new ColorLegend(ui->colorScaleInputRaster);
    this->inputRasterColorLegend->resize(ui->colorScaleInputRaster->size());

    this->outputRasterColorLegend = new ColorLegend(this->ui->colorScaleOutputRaster);
    this->outputRasterColorLegend->resize(ui->colorScaleOutputRaster->size());

    this->meteoPointsLegend = new ColorLegend(ui->colorScaleMeteoPoints);
    this->meteoPointsLegend->resize(ui->colorScaleMeteoPoints->size());
    this->meteoPointsLegend->colorScale = myProject.meteoPointsColorScale;

    // initialize
    ui->labelInputRaster->setText("");
    ui->labelOutputRaster->setText("");
    this->currentPointsVisualization = notShown;

    // show menu
    showPointsGroup = new QActionGroup(this);
    showPointsGroup->setExclusive(true);
    showPointsGroup->addAction(ui->actionView_PointsHide);
    showPointsGroup->addAction(ui->actionView_PointsLocation);
    showPointsGroup->addAction(ui->actionView_PointsCurrentVariable);
    showPointsGroup->setEnabled(false);

    this->setTileMapSource(WebTileSource::OPEN_STREET_MAP);

    // Set start size and position
    this->startCenter = new Position (myProject.gisSettings.startLocation.longitude,
                                     myProject.gisSettings.startLocation.latitude, 0.0);
    this->mapView->setZoomLevel(8);
    this->mapView->centerOn(startCenter->lonLat());
    connect(this->mapView, SIGNAL(zoomLevelChanged(quint8)), this, SLOT(updateMaps()));
    connect(this->mapView, SIGNAL(mouseMoveSignal(const QPoint&)), this, SLOT(mouseMove(const QPoint&)));

    // Set raster objects
    this->rasterDEM = new RasterObject(this->mapView);
    this->rasterDEM->setOpacity(this->ui->opacitySliderRasterInput->value() / 100.0);
    this->rasterDEM->setColorLegend(this->inputRasterColorLegend);
    this->mapView->scene()->addObject(this->rasterDEM);

    this->rasterOutput = new RasterObject(this->mapView);
    this->rasterOutput->setOpacity(this->ui->opacitySliderRasterOutput->value() / 100.0);
    this->rasterOutput->setColorLegend(this->outputRasterColorLegend);
    this->mapView->scene()->addObject(this->rasterOutput);

    this->updateVariable();
    this->updateDateTime();

    this->setMouseTracking(true);
}


void MainWindow::resizeEvent(QResizeEvent * event)
{
    Q_UNUSED(event)

    const int INFOHEIGHT = 40;
    int x1 = this->width() - TOOLSWIDTH - MAPBORDER;
    int dy = ui->groupBoxModel->height() + ui->groupBoxMeteoPoints->height() + ui->groupBoxDEM->height() + ui->groupBoxVariableMap->height() + MAPBORDER*4;
    int y1 = (this->height() - INFOHEIGHT - dy) / 2;

    ui->widgetMap->setGeometry(0, 0, x1, this->height() - INFOHEIGHT);
    mapView->resize(ui->widgetMap->size());

    ui->groupBoxModel->move(x1, y1);
    ui->groupBoxModel->resize(TOOLSWIDTH, ui->groupBoxModel->height());
    y1 += ui->groupBoxModel->height() + MAPBORDER*2;

    ui->groupBoxDEM->move(x1, y1);
    ui->groupBoxDEM->resize(TOOLSWIDTH, ui->groupBoxDEM->height());
    y1 += ui->groupBoxDEM->height() + MAPBORDER;

    ui->groupBoxMeteoPoints->move(x1, y1);
    ui->groupBoxMeteoPoints->resize(TOOLSWIDTH, ui->groupBoxMeteoPoints->height());
    y1 += ui->groupBoxMeteoPoints->height() + MAPBORDER;

    ui->groupBoxVariableMap->move(x1, y1);
    ui->groupBoxVariableMap->resize(TOOLSWIDTH, ui->groupBoxVariableMap->height());
    this->updateMaps();
}


void MainWindow::updateMaps()
{
    rasterDEM->updateCenter();
    rasterOutput->updateCenter();
    *startCenter = rasterDEM->getCurrentCenter();
}


void MainWindow::updateGUI()
{
    updateDateTime();
    rasterDEM->redrawRequested();
    rasterOutput->redrawRequested();
    qApp->processEvents();
}


void MainWindow::mouseMove(const QPoint& eventPos)
{
    if (! isInsideMap(eventPos)) return;

    Position geoPoint = this->mapView->mapToScene(eventPos);
    this->ui->statusBar->showMessage(QString::number(geoPoint.latitude()) + " " + QString::number(geoPoint.longitude()));
}


void MainWindow::mouseReleaseEvent(QMouseEvent *event)
{
    Q_UNUSED(event)
    updateMaps();
}


void MainWindow::mouseDoubleClickEvent(QMouseEvent * event)
{
    QPoint mapPos = getMapPos(event->pos());
    if (! isInsideMap(mapPos)) return;

    Position newCenter = this->mapView->mapToScene(mapPos);
    this->ui->statusBar->showMessage(QString::number(newCenter.latitude()) + " " + QString::number(newCenter.longitude()));

    if (event->button() == Qt::LeftButton)
    {
        this->mapView->zoomIn();
    }
    else if (event->button() == Qt::RightButton)
    {
        this->mapView->zoomOut();
    }

    this->mapView->centerOn(newCenter.lonLat());
}



void MainWindow::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::RightButton)
    {
        contextMenuRequested(event->pos(), event->globalPos());
    }
}


void MainWindow::addMeteoPoints()
{
    myProject.meteoPointsSelected.clear();
    for (int i = 0; i < myProject.nrMeteoPoints; i++)
    {
        StationMarker* point = new StationMarker(5.0, true, QColor((Qt::white)), this->mapView);

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

        this->pointList.append(point);
        this->mapView->scene()->addObject(this->pointList[i]);

        point->setToolTip();
        connect(point, SIGNAL(newStationClicked(std::string, std::string, bool)), this, SLOT(callNewMeteoWidget(std::string, std::string, bool)));
        connect(point, SIGNAL(appendStationClicked(std::string, std::string, bool)), this, SLOT(callAppendMeteoWidget(std::string, std::string, bool)));
    }
}

void MainWindow::callNewMeteoWidget(std::string id, std::string name, bool isGrid)
{
    bool isAppend = false;
    if (isGrid)
    {
        myProject.showMeteoWidgetGrid(id, isAppend);
    }
    else
    {
        myProject.showMeteoWidgetPoint(id, name, isAppend);
    }
    return;
}

void MainWindow::callAppendMeteoWidget(std::string id, std::string name, bool isGrid)
{
    bool isAppend = true;
    if (isGrid)
    {
        myProject.showMeteoWidgetGrid(id, isAppend);
    }
    else
    {
        myProject.showMeteoWidgetPoint(id, name, isAppend);
    }
    return;
}


void MainWindow::drawMeteoPoints()
{
    resetMeteoPoints();

    if (! myProject.meteoPointsLoaded || myProject.nrMeteoPoints == 0)
    {
        ui->groupBoxMeteoPoints->setEnabled(false);
        return;
    }

    addMeteoPoints();
    ui->groupBoxMeteoPoints->setEnabled(true);

    myProject.loadMeteoPointsData (myProject.getCurrentDate(), myProject.getCurrentDate(), true, true, true);

    showPointsGroup->setEnabled(true);

    currentPointsVisualization = showLocation;
    redrawMeteoPoints(currentPointsVisualization, true);

    updateDateTime();
}


void MainWindow::setProjectTileMap()
{
    if (myProject.currentTileMap != "")
    {
        if (myProject.currentTileMap.toUpper() == "ESRI")
        {
            this->setTileMapSource(WebTileSource::ESRI_WorldImagery);
        }
        else if (myProject.currentTileMap.toUpper() == "TERRAIN")
        {
            this->setTileMapSource(WebTileSource::GOOGLE_Terrain);
        }
        else if (myProject.currentTileMap.toUpper() == "GOOGLE")
        {
            this->setTileMapSource(WebTileSource::GOOGLE_Hybrid_Satellite);
        }
        else
        {
            this->setTileMapSource(WebTileSource::OPEN_STREET_MAP);
        }
    }
    else
    {
        // default: Open Street Map
        this->setTileMapSource(WebTileSource::OPEN_STREET_MAP);
    }
}


void MainWindow::drawProject()
{
    setProjectTileMap();

    if (myProject.DEM.isLoaded)
    {
        this->renderDEM();
    }
    else
    {
        startCenter = new Position (myProject.gisSettings.startLocation.longitude,
                                myProject.gisSettings.startLocation.latitude, 0.0);
        mapView->centerOn(startCenter->lonLat());
        mapView->setZoomLevel(8);
    }

    drawMeteoPoints();
    // drawMeteoGrid();

    QString title = "CRITERIA3D";
    if (myProject.projectName != "")
        title += " - " + myProject.projectName;

    this->setWindowTitle(title);
}


void MainWindow::clearDEM_GUI()
{
    rasterDEM->clear();
    rasterOutput->clear();

    ui->labelInputRaster->setText("");
    ui->labelOutputRaster->setText("");

    setInputRasterVisible(false);
    setOutputRasterVisible(false);

    ui->groupBoxDEM->setEnabled(false);
    ui->groupBoxVariableMap->setEnabled(false);
}


void MainWindow::clearMeteoPoints_GUI()
{
    resetMeteoPoints();
    meteoPointsLegend->setVisible(false);
    showPointsGroup->setEnabled(false);
}


void MainWindow::renderDEM()
{
    if (!myProject.DEM.isLoaded)
        return;

    ui->groupBoxDEM->setEnabled(true);
    ui->groupBoxVariableMap->setEnabled(true);
    ui->opacitySliderRasterInput->setEnabled(true);
    ui->opacitySliderRasterOutput->setEnabled(true);

    setCurrentRasterInput(&(myProject.DEM));
    ui->labelInputRaster->setText(QString::fromStdString(getVariableString(noMeteoTerrain)));

    // center map
    gis::Crit3DGeoPoint* center = this->rasterDEM->getRasterCenter();
    mapView->centerOn(qreal(center->longitude), qreal(center->latitude));

    // resize map
    double size = double(this->rasterDEM->getRasterMaxSize());
    size = log2(1000 / size);
    mapView->setZoomLevel(quint8(size));
    mapView->centerOn(qreal(center->longitude), qreal(center->latitude));

    updateMaps();

    /*
    if (viewer3D != nullptr)
    {
        initializeViewer3D();
        //this->viewer3D->close();
        //this->viewer3D = nullptr;
    }
    */
}


void MainWindow::on_actionLoad_DEM_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Digital Elevation Model"), "", tr("ESRI grid files (*.flt)"));

    if (fileName == "") return;

    if (! myProject.loadDEM(fileName)) return;

    this->renderDEM();
}


void MainWindow::on_actionOpenProject_triggered()
{
    QString projectPath = myProject.getDefaultPath() + PATH_PROJECT;
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open project file"), projectPath, tr("ini files (*.ini)"));
    if (fileName == "") return;

    if (myProject.isProjectLoaded)
    {
        clearMeteoPoints_GUI();
        clearDEM_GUI();
    }

    if (! myProject.loadCriteria3DProject(fileName))
    {
        myProject.logError("Error opening project: " + myProject.errorString);
        myProject.loadCriteria3DProject(myProject.getApplicationPath() + "default.ini");
    }

    drawProject();
}


void MainWindow::on_actionCloseProject_triggered()
{
    if (! myProject.isProjectLoaded) return;

    clearMeteoPoints_GUI();
    clearDEM_GUI();

    myProject.loadCriteria3DProject(myProject.getApplicationPath() + "default.ini");

    drawProject();
}


QPoint MainWindow::getMapPos(const QPoint& pos)
{
    QPoint mapPoint;
    int dx = ui->widgetMap->x();
    int dy = ui->widgetMap->y() + ui->menuBar->height();
    mapPoint.setX(pos.x() - dx - MAPBORDER);
    mapPoint.setY(pos.y() - dy - MAPBORDER);
    return mapPoint;
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


void MainWindow::resetMeteoPoints()
{
    for (int i = 0; i < pointList.size(); i++)
    {
        mapView->scene()->removeObject(pointList[i]);
        delete pointList[i];
    }

    pointList.clear();
}


void MainWindow::on_actionVariableQualitySpatial_triggered()
{
    myProject.checkSpatialQuality = ui->actionVariableQualitySpatial->isChecked();
    updateVariable();
}


void MainWindow::interpolateDemGUI()
{
    meteoVariable myVar = myProject.getCurrentVariable();

    if (myProject.interpolateHourlyMeteoVar(myVar, myProject.getCurrentTime()))
    {
        showMeteoVariable(myProject.getCurrentVariable());
    }
}


void MainWindow::updateVariable()
{
    std::string myString = getVariableString(myProject.getCurrentVariable());
    ui->labelVariable->setText(QString::fromStdString(myString));

    redrawMeteoPoints(currentPointsVisualization, true);
}


void MainWindow::updateDateTime()
{
    this->ui->dateEdit->setDate(myProject.getCurrentDate());
    this->ui->timeEdit->setValue(myProject.getCurrentHour());
}


void MainWindow::on_dateEdit_dateChanged(const QDate &date)
{
    if (date != myProject.getCurrentDate())
    {
        myProject.setCurrentDate(date);
        myProject.loadMeteoPointsData(date, date, true, true, true);
        myProject.loadMeteoGridData(date, date, true);
        myProject.setAllHourlyMeteoMapsComputed(false);
    }

    redrawMeteoPoints(currentPointsVisualization, true);
}


void MainWindow::on_timeEdit_valueChanged(int myHour)
{
    if (myHour != myProject.getCurrentHour())
    {
        myProject.setCurrentHour(myHour);
        myProject.setAllHourlyMeteoMapsComputed(false);
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
        pointList[i]->setVisible(false);

    meteoPointsLegend->setVisible(true);

    switch(currentPointsVisualization)
    {
        case notShown:
        {
            meteoPointsLegend->setVisible(false);
            this->ui->actionView_PointsHide->setChecked(true);
            break;
        }
        case showLocation:
        {
            this->ui->actionView_PointsLocation->setChecked(true);
            for (int i = 0; i < myProject.nrMeteoPoints; i++)
            {
                    myProject.meteoPoints[i].currentValue = NODATA;
                    pointList[i]->setFillColor(QColor(Qt::white));
                    pointList[i]->setRadius(5);
                    pointList[i]->setCurrentValue(NODATA);
                    pointList[i]->setToolTip();
                    pointList[i]->setVisible(true);
            }

            myProject.meteoPointsColorScale->setRange(NODATA, NODATA);
            meteoPointsLegend->update();
            break;
        }
        case showCurrentVariable:
        {
            this->ui->actionView_PointsCurrentVariable->setChecked(true);
            // quality control
            checkData(myProject.quality, myProject.getCurrentVariable(),
                      myProject.meteoPoints, myProject.nrMeteoPoints, myProject.getCrit3DCurrentTime(),
                      &(myProject.qualityInterpolationSettings), myProject.meteoSettings,
                      &(myProject.climateParameters), myProject.checkSpatialQuality);

            if (updateColorSCale)
            {
                float minimum, maximum;
                myProject.getMeteoPointsRange(&minimum, &maximum);

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
                        pointList[i]->setRadius(5);
                        myColor = myProject.meteoPointsColorScale->getColor(myProject.meteoPoints[i].currentValue);
                        pointList[i]->setFillColor(QColor(myColor->red, myColor->green, myColor->blue));
                        pointList[i]->setOpacity(1.0);
                    }
                    else
                    {
                        // Wrong data
                        pointList[i]->setRadius(10);
                        pointList[i]->setFillColor(QColor(Qt::black));
                        pointList[i]->setOpacity(0.5);
                    }

                    pointList[i]->setCurrentValue(myProject.meteoPoints[i].currentValue);
                    pointList[i]->setQuality(myProject.meteoPoints[i].quality);
                    pointList[i]->setToolTip();
                    pointList[i]->setVisible(true);
                }
            }

            meteoPointsLegend->update();
            break;
        }
        default:
        {
            meteoPointsLegend->setVisible(false);
            this->ui->actionView_PointsHide->setChecked(true);
            break;
        }
    }
}


bool MainWindow::loadMeteoPointsDB(QString dbName)
{
    myProject.logInfoGUI("Load " + dbName);
    bool success = myProject.loadMeteoPointsDB(dbName);
    myProject.closeLogInfo();

    if (success)
        drawMeteoPoints();

    return success;
}


void MainWindow::on_opacitySliderRasterInput_sliderMoved(int position)
{
    this->rasterDEM->setOpacity(position / 100.0);
}


void MainWindow::on_opacitySliderRasterOutput_sliderMoved(int position)
{
    this->rasterOutput->setOpacity(position / 100.0);
}


void MainWindow::on_variableButton_clicked()
{
    myProject.setCurrentVariable(chooseMeteoVariable(&myProject));
    this->currentPointsVisualization = showCurrentVariable;
    this->updateVariable();
}


void MainWindow::setInputRasterVisible(bool value)
{
    inputRasterColorLegend->setVisible(value);
    ui->labelInputRaster->setVisible(value);
    rasterDEM->setVisible(value);
}

void MainWindow::setOutputRasterVisible(bool value)
{
    outputRasterColorLegend->setVisible(value);
    ui->labelOutputRaster->setVisible(value);
    rasterOutput->setVisible(value);
}


void MainWindow::setCurrentRasterInput(gis::Crit3DRasterGrid *myRaster)
{
    setInputRasterVisible(true);

    rasterDEM->initializeUTM(myRaster, myProject.gisSettings, false);
    inputRasterColorLegend->colorScale = myRaster->colorScale;

    inputRasterColorLegend->repaint();
    rasterDEM->redrawRequested();
}


void MainWindow::setCurrentRasterOutput(gis::Crit3DRasterGrid *myRaster)
{
    setOutputRasterVisible(true);

    rasterOutput->initializeUTM(myRaster, myProject.gisSettings, false);
    outputRasterColorLegend->colorScale = myRaster->colorScale;
    outputRasterColorLegend->repaint();
    rasterOutput->redrawRequested();
    updateMaps();
}



void MainWindow::on_actionInterpolationSettings_triggered()
{
    if (myProject.meteoPointsDbHandler == nullptr)
    {
        myProject.logInfoGUI("Open a Meteo Points DB before.");
        return;
    }

    DialogInterpolation* myInterpolationDialog = new DialogInterpolation(&myProject);
    myInterpolationDialog->close();
}


void MainWindow::on_actionRadiationSettings_triggered()
{
    DialogRadiation* myDialogRadiation = new DialogRadiation(&myProject);
    myDialogRadiation->close();
}


void MainWindow::on_actionProjectSettings_triggered()
{
    DialogSettings* settingsDialog = new DialogSettings(&myProject);
    int result = settingsDialog->exec();
    settingsDialog->close();

    if (result == QDialog::Accepted)
    {
        if (! isEqual(startCenter->latitude(), myProject.gisSettings.startLocation.latitude) ||
            ! isEqual(startCenter->longitude(), myProject.gisSettings.startLocation.longitude))
        {
            startCenter->setLatitude(myProject.gisSettings.startLocation.latitude);
            startCenter->setLongitude(myProject.gisSettings.startLocation.longitude);
            this->mapView->centerOn(startCenter->lonLat());
        }
    }
}


void MainWindow::on_actionCriteria3D_Initialize_triggered()
{
    myProject.initializeCriteria3DModel();
}


/*
void MainWindow::on_viewer3DClosed()
{
    this->viewer3D = nullptr;
}


bool MainWindow::initializeViewer3D()
{
    if (viewer3D == nullptr) return false;

    if (! myProject.isCriteria3DInitialized)
    {
        myProject.logError("Initialize 3D model before");
        return false;
    }
    else
    {
        viewer3D->initialize(&myProject);
        return true;
    }
}


void MainWindow::on_actionView_3D_triggered()
{
    if (! myProject.DEM.isLoaded)
    {
        myProject.logInfoGUI("Load a Digital Elevation Model before.");
        return;
    }

    if (viewer3D == nullptr || ! viewer3D->isVisible())
    {
        viewer3D = new Viewer3D(this);
        if (! initializeViewer3D()) return;
    }

    viewer3D->show();
    connect (viewer3D, SIGNAL(destroyed()), this, SLOT(on_viewer3DClosed()));
}
*/


void MainWindow::on_actionView_SoilMap_triggered()
{
    if (myProject.soilMap.isLoaded)
    {
        setColorScale(airTemperature, myProject.soilMap.colorScale);
        setCurrentRasterOutput(&(myProject.soilMap));
        ui->labelOutputRaster->setText("Soil index");
    }
    else
    {
        myProject.logError("Load a soil map before.");
        return;
    }
}


void MainWindow::on_actionHide_soil_map_triggered()
{
    if (ui->labelOutputRaster->text() == "Soil index")
    {
        setOutputRasterVisible(false);
    }
}


void MainWindow::on_actionView_Boundary_triggered()
{
    if (myProject.boundaryMap.isLoaded)
    {
        setColorScale(noMeteoTerrain, myProject.boundaryMap.colorScale);
        setCurrentRasterOutput(&(myProject.boundaryMap));
        ui->labelOutputRaster->setText("Boundary map");
    }
    else
    {
        myProject.logInfoGUI("Initialize 3D Model before.");
        return;
    }
}


void MainWindow::on_actionView_None_triggered()
{
    setOutputRasterVisible(false);
}

void MainWindow::on_actionViewMeteoVariable_None_triggered()
{
    setOutputRasterVisible(false);
}

void MainWindow::on_actionView_Slope_triggered()
{
    if (myProject.DEM.isLoaded)
    {
        setColorScale(noMeteoTerrain, myProject.radiationMaps->slopeMap->colorScale);
        setCurrentRasterOutput(myProject.radiationMaps->slopeMap);
        ui->labelOutputRaster->setText("Slope °");
    }
    else
    {
        myProject.logInfoGUI("Load a Digital Elevation Model before.");
        return;
    }
}


void MainWindow::on_actionView_Aspect_triggered()
{
    if (myProject.DEM.isLoaded)
    {
        setColorScale(airRelHumidity, myProject.radiationMaps->aspectMap->colorScale);
        setCurrentRasterOutput(myProject.radiationMaps->aspectMap);
        ui->labelOutputRaster->setText("Aspect °");
    }
    else
    {
        myProject.logInfoGUI("Load a Digital Elevation Model before.");
        return;
    }
}


bool MainWindow::checkMapVariable(bool isComputed)
{
    if (! myProject.DEM.isLoaded)
    {
        myProject.logInfoGUI("Load a Digital Elevation Model before.");
        return false;
    }

    if (! isComputed)
    {
        myProject.logInfoGUI("Interpolate variables before.");
        return false;
    }

    return true;
}


void MainWindow::setMeteoVariable(meteoVariable myVar, gis::Crit3DRasterGrid *myGrid)
{   
    myProject.setCurrentVariable(myVar);

    setColorScale(myVar, myGrid->colorScale);
    setCurrentRasterOutput(myGrid);
    ui->labelOutputRaster->setText(QString::fromStdString(getVariableString(myVar)));

    currentPointsVisualization = showCurrentVariable;
    updateVariable();
}


void MainWindow::setSnowVariable(meteoVariable myVar, gis::Crit3DRasterGrid *myGrid)
{
    setColorScale(myVar, myGrid->colorScale);
    setCurrentRasterOutput(myGrid);
    ui->labelOutputRaster->setText(QString::fromStdString(getVariableString(myVar)));
}


void MainWindow::showMeteoVariable(meteoVariable var)
{
    if (myProject.hourlyMeteoMaps == nullptr)
    {
        myProject.logInfoGUI("Open a project before.");
        return;
    }

    switch(var)
    {
    case airTemperature:
        if (checkMapVariable(myProject.hourlyMeteoMaps->getComputed()))
            setMeteoVariable(airTemperature, myProject.hourlyMeteoMaps->mapHourlyTair);
        break;

    case precipitation:
        if (checkMapVariable(myProject.hourlyMeteoMaps->getComputed()))
            setMeteoVariable(precipitation, myProject.hourlyMeteoMaps->mapHourlyPrec);
        break;

    case airRelHumidity:
        if (checkMapVariable(myProject.hourlyMeteoMaps->getComputed()))
            setMeteoVariable(airRelHumidity, myProject.hourlyMeteoMaps->mapHourlyRelHum);
        break;

    case windScalarIntensity:
        if (checkMapVariable(myProject.hourlyMeteoMaps->getComputed()))
            setMeteoVariable(windScalarIntensity, myProject.hourlyMeteoMaps->mapHourlyWindScalarInt);
        break;

    case globalIrradiance:
        if (checkMapVariable(myProject.radiationMaps->getComputed()))
            setMeteoVariable(globalIrradiance, myProject.radiationMaps->globalRadiationMap);
        break;

    case atmTransmissivity:
        if (checkMapVariable(myProject.radiationMaps->getComputed()))
            setMeteoVariable(atmTransmissivity, myProject.radiationMaps->transmissivityMap);
        break;

    case referenceEvapotranspiration:
        if (checkMapVariable(myProject.hourlyMeteoMaps->getComputed()))
            setMeteoVariable(referenceEvapotranspiration, myProject.hourlyMeteoMaps->mapHourlyET0);
        break;

    default:
        {}
    }
}


void MainWindow::showSnowVariable(meteoVariable var)
{
    if (! myProject.DEM.isLoaded)
    {
        myProject.logInfoGUI("Load a Digital Elevation Model before.");
        return;
    }

    if (! myProject.snowMaps.isInitialized)
    {
        myProject.logInfoGUI("Initialize snow model before.");
        return;
    }

    switch(var)
    {
    case snowWaterEquivalent:
        setSnowVariable(snowWaterEquivalent, myProject.snowMaps.getSnowWaterEquivalentMap());
        break;

    case snowFall:
        setSnowVariable(snowFall, myProject.snowMaps.getSnowFallMap());
        break;

    case snowSurfaceTemperature:
        setSnowVariable(snowSurfaceTemperature, myProject.snowMaps.getSnowSurfaceTempMap());
        break;

    case snowInternalEnergy:
        setSnowVariable(snowInternalEnergy, myProject.snowMaps.getInternalEnergyMap());
        break;

    case snowSurfaceInternalEnergy:
        setSnowVariable(snowSurfaceInternalEnergy, myProject.snowMaps.getSurfaceInternalEnergyMap());
        break;

    default:
        {}
    }
}


void MainWindow::on_actionView_Air_temperature_triggered()
{
    showMeteoVariable(airTemperature);
}

void MainWindow::on_actionView_Transmissivity_triggered()
{
    showMeteoVariable(atmTransmissivity);
}

void MainWindow::on_actionView_Global_radiation_triggered()
{
    showMeteoVariable(globalIrradiance);
}

void MainWindow::on_actionView_ET0_triggered()
{
    showMeteoVariable(referenceEvapotranspiration);
}

void MainWindow::on_actionView_Precipitation_triggered()
{
    showMeteoVariable(precipitation);
}

void MainWindow::on_actionView_Air_relative_humidity_triggered()
{
    showMeteoVariable(airRelHumidity);
}

void MainWindow::on_actionView_Wind_intensity_triggered()
{
    showMeteoVariable(windScalarIntensity);
}

void MainWindow::on_actionView_Snow_water_equivalent_triggered()
{
    showSnowVariable(snowWaterEquivalent);
}

void MainWindow::on_actionView_Snow_fall_triggered()
{
    showSnowVariable(snowFall);
}

void MainWindow::on_actionView_Snow_surface_temperature_triggered()
{
    showSnowVariable(snowSurfaceTemperature);
}

void MainWindow::on_actionView_Snow_internal_energy_triggered()
{
    showSnowVariable(snowInternalEnergy);
}

void MainWindow::on_actionView_Snow_surface_internal_energy_triggered()
{
    showSnowVariable(snowSurfaceInternalEnergy);
}


void MainWindow::on_actionView_PointsHide_triggered()
{
    redrawMeteoPoints(notShown, true);
}


void MainWindow::on_actionView_PointsLocation_triggered()
{
    redrawMeteoPoints(showLocation, true);
}


void MainWindow::on_actionView_PointsCurrentVariable_triggered()
{
    redrawMeteoPoints(showCurrentVariable, true);
}


void MainWindow::on_actionMapTerrain_triggered()
{
    this->setTileMapSource(WebTileSource::GOOGLE_Terrain);
}


void MainWindow::on_actionMapOpenStreetMap_triggered()
{
    this->setTileMapSource(WebTileSource::OPEN_STREET_MAP);
}


void MainWindow::on_actionMapESRISatellite_triggered()
{
    this->setTileMapSource(WebTileSource::ESRI_WorldImagery);
}


void MainWindow::on_actionMapGoogle_hybrid_satellite_triggered()
{
    this->setTileMapSource(WebTileSource::GOOGLE_Hybrid_Satellite);
}


void MainWindow::setTileMapSource(WebTileSource::WebTileType tileSource)
{
    // set menu
    ui->actionMapOpenStreetMap->setChecked(false);
    ui->actionMapTerrain->setChecked(false);
    ui->actionMapESRISatellite->setChecked(false);
    ui->actionMapGoogle_hybrid_satellite->setChecked(false);

    if (tileSource == WebTileSource::OPEN_STREET_MAP)
    {
        ui->actionMapOpenStreetMap->setChecked(true);
    }
    else if (tileSource == WebTileSource::GOOGLE_Hybrid_Satellite)
    {
        ui->actionMapGoogle_hybrid_satellite->setChecked(true);
    }
    else if (tileSource == WebTileSource::GOOGLE_Terrain)
    {
        ui->actionMapTerrain->setChecked(true);
    }
    else if (tileSource == WebTileSource::ESRI_WorldImagery)
    {
        ui->actionMapESRISatellite->setChecked(true);
    }

    // set tiles source
    QSharedPointer<WebTileSource> myTiles(new WebTileSource(tileSource), &QObject::deleteLater);

    this->mapView->setTileSource(myTiles);
}


void MainWindow::on_actionCompute_solar_radiation_triggered()
{
    if (myProject.nrMeteoPoints == 0)
    {
        myProject.logInfoGUI("Open a Meteo Points DB before.");
        return;
    }

    myProject.setCurrentVariable(globalIrradiance);
    this->currentPointsVisualization = showCurrentVariable;
    this->updateVariable();

    this->interpolateDemGUI();
}


void MainWindow::on_actionCompute_AllMeteoMaps_triggered()
{
    if (! myProject.DEM.isLoaded)
    {
        myProject.logInfoGUI("Load a Digital Elevation Model before.");
        return;
    }

    if (myProject.nrMeteoPoints == 0)
    {
        myProject.logInfoGUI("Open a meteo points DB before.");
        return;
    }

    setOutputRasterVisible(false);

    if (! myProject.computeAllMeteoMaps(myProject.getCurrentTime(), true))
    {
        myProject.logError();
        return;
    }

    showMeteoVariable(myProject.getCurrentVariable());
}


void MainWindow::openSoilWidget(QPoint mapPos)
{
    double x, y;
    Position geoPos = mapView->mapToScene(mapPos);
    gis::latLonToUtmForceZone(myProject.gisSettings.utmZone, geoPos.latitude(), geoPos.longitude(), &x, &y);
    QString soilCode = myProject.getCrit3DSoilCode(x, y);

    if (soilCode == "")
    {
        myProject.logInfoGUI("No soil.");
    }
    else if (soilCode == "NOT FOUND")
    {
        myProject.logError("Soil code not found: check soil database.");
    }
    else
    {
        QString dbSoilName = myProject.getCompleteFileName(myProject.soilDbFileName, PATH_SOIL);
        QSqlDatabase dbSoil;
        if (! openDbSoil(dbSoilName, &dbSoil, &(myProject.errorString)))
        {
            myProject.logError();
            return;
        }
        soilWidget = new Crit3DSoilWidget();
        soilWidget->show();
        soilWidget->setDbSoil(dbSoil, soilCode);
    }
}


void MainWindow::contextMenuRequested(QPoint localPos, QPoint globalPos)
{
    QMenu submenu;
    int nrItems = 0;

    QPoint mapPos = getMapPos(localPos);
    if (! isInsideMap(mapPos)) return;

    if (myProject.soilMap.isLoaded)
    {
        submenu.addAction("Show soil data");
        nrItems++;
    }
    if (nrItems == 0) return;

    QAction* myAction = submenu.exec(globalPos);

    if (myAction)
    {
        if (myAction->text().contains("Show soil data") )
        {
            if (myProject.nrSoils > 0) {
                openSoilWidget(mapPos);
            }
            else {
                myProject.logInfoGUI("Load soil database before.");
            }
        }
    }
}


void MainWindow::on_actionLoad_soil_map_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open soil map"), "", tr("ESRI grid files (*.flt)"));
    if (fileName == "") return;

    if (myProject.loadSoilMap(fileName))
    {
        on_actionView_SoilMap_triggered();
    }
}


void MainWindow::on_actionLoad_soil_data_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open DB soil"), "", tr("SQLite files (*.db)"));
    if (fileName == "") return;

    myProject.loadSoilDatabase(fileName);
}


void MainWindow::on_actionLoad_Crop_data_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open DB Crop"), "", tr("SQLite files (*.db)"));
    if (fileName == "") return;

    myProject.loadCropDatabase(fileName);
}


void MainWindow::on_actionLoad_MeteoPoints_triggered()
{
    QString dbName = QFileDialog::getOpenFileName(this, tr("Open meteo points DB"), "", tr("DB files (*.db)"));
    if (dbName != "") loadMeteoPointsDB(dbName);
}


void MainWindow::on_actionMeteoPointsImport_data_triggered()
{
    if (! myProject.meteoPointsLoaded)
    {
        myProject.logInfoGUI("Load meteo points database before.");
        return;
    }

    QString fileName = QFileDialog::getOpenFileName(this, tr("Import meteo data (.csv)"), "", tr("csv files (*.csv)"));
    if (fileName == "") return;

    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, "Import data", "Do you want to import all csv files in the directory?", QMessageBox::Yes|QMessageBox::No);

    bool importAllFiles = (reply == QMessageBox::Yes);
    myProject.importHourlyMeteoData(fileName, importAllFiles, true);
}


void MainWindow::on_actionNew_meteoPointsDB_from_csv_triggered()
{
    QString templateFileName = myProject.getDefaultPath() + PATH_TEMPLATE + "template_meteo.db";
    QString meteoPointsPath = myProject.getDefaultPath() + PATH_METEOPOINT;

    QString dbName = QFileDialog::getSaveFileName(this, tr("Save as"), meteoPointsPath, tr("DB files (*.db)"));
    if (dbName == "")
    {
        qDebug() << "missing db file name";
        return;
    }

    QString csvFileName = QFileDialog::getOpenFileName(this, tr("Open properties csv file"), "", tr("csv files (*.csv)"));
    if (csvFileName.isEmpty())
        return;

    myProject.closeMeteoPointsDB();

    QFile dbFile(dbName);
    if (dbFile.exists())
    {
        dbFile.close();
        dbFile.setPermissions(QFile::ReadOther | QFile::WriteOther);
        if (! dbFile.remove())
        {
            myProject.logError("Remove file failed: " + dbName + "\n" + dbFile.errorString());
            return;
        }
    }

    if (! QFile::copy(templateFileName, dbName))
    {
        myProject.logError("Copy file failed: " + templateFileName);
        return;
    }
    myProject.meteoPointsDbHandler = new Crit3DMeteoPointsDbHandler(dbName);

    QList<QString> pointPropertiesList;
    if (!myProject.meteoPointsDbHandler->getNameColumn("point_properties", &pointPropertiesList))
    {
        myProject.logError("Error in read table point_properties");
        return;
    }
    QList<QString> csvFields;
    if (!myProject.parseMeteoPointsPropertiesCSV(csvFileName, &csvFields))
    {
        myProject.logError("Error in parse properties");
        return;
    }

    DialogPointProperties dialogPointProp(pointPropertiesList, csvFields);
    if (dialogPointProp.result() != QDialog::Accepted)
    {
        return;
    }
    else
    {
        QList<QString> joinedList = dialogPointProp.getJoinedList();
        if (! myProject.writeMeteoPointsProperties(joinedList))
        {
            myProject.logError("Error in write points properties");
            return;
        }
    }

    loadMeteoPointsDB(dbName);
}


bool selectDates(QDateTime &firstTime, QDateTime &lastTime)
{
    if (! myProject.meteoPointsLoaded)
    {
        myProject.logError("Load meteo points db before.");
        return false;
    }

    firstTime = myProject.getCurrentTime();
    lastTime = myProject.getCurrentTime();
    firstTime.setTime(QTime(0,0,0));
    lastTime.setTime(QTime(23,0,0));

    QDateTime firstDateH = myProject.meteoPointsDbHandler->getFirstDate(hourly);
    QDateTime lastDateH = myProject.meteoPointsDbHandler->getLastDate(hourly);

    formPeriod myForm(&firstTime, &lastTime);
    myForm.setMinimumDate(firstDateH.date());
    myForm.setMaximumDate(lastDateH.date());
    myForm.show();

    if (myForm.exec() == QDialog::Rejected)
        return false;

    if (lastTime < firstTime)
    {
        myProject.logError("Wrong period");
        return false;
    }

    return true;
}


bool MainWindow::startModels(QDateTime firstTime, QDateTime lastTime)
{
    if (myProject.isWater && (! myProject.isCriteria3DInitialized))
    {
        myProject.logError("Initialize 3D water fluxes or load a state before.");
        return false;
    }

    // TODO: check on crop

    if (myProject.isSnow && (! myProject.snowMaps.isInitialized))
    {
        myProject.logError("Initialize Snow model or load a state before.");
        return false;
    }

    myProject.logInfoGUI("Loading meteo data...");
    if (! myProject.loadMeteoPointsData(firstTime.date().addDays(-1), lastTime.date().addDays(+1), true, false, false))
    {
        myProject.logError();
        return false;
    }

    // set model interface
    myProject.modelFirstTime = firstTime;
    myProject.modelLastTime = lastTime;
    myProject.modelPause = false;
    myProject.modelStop = false;
    ui->groupBoxModel->setEnabled(true);
    ui->buttonModelPause->setEnabled(true);
    ui->buttonModelStart->setDisabled(true);

    //myProject.logInfoGUI("Run models from: " + firstTime.toString() + " to: " + lastTime.toString());
    myProject.closeLogInfo();

    return runModels(firstTime, lastTime);
}


bool MainWindow::runModels(QDateTime firstTime, QDateTime lastTime)
{
    QDate firstDate = firstTime.date();
    QDate lastDate = lastTime.date();
    int hour1 = firstTime.time().hour();
    int hour2 = lastTime.time().hour();

    QString outputPathHourly;

    // cycle on days
    for (QDate myDate = firstDate; myDate <= lastDate; myDate = myDate.addDays(1))
    {
        myProject.setCurrentDate(myDate);

        if (myProject.saveOutput)
        {
            // create output directory
            outputPathHourly = myProject.getProjectPath() + "OUTPUT/hourly/" + myDate.toString("yyyy/MM/dd/");
            if (! QDir().mkpath(outputPathHourly))
            {
                myProject.logError("Creation hourly output directory failed." );
                myProject.saveOutput = false;
            }
        }

        // cycle on hours
        int firstHour = (myDate == firstDate) ? hour1 : 0;
        int lastHour = (myDate == lastDate) ? hour2 : 23;

        for (int hour = firstHour; hour <= lastHour; hour++)
        {
            myProject.setCurrentHour(hour);
            QDateTime myTime = QDateTime(myDate, QTime(hour, 0, 0));

            if (! myProject.modelHourlyCycle(myTime, outputPathHourly))
            {
                myProject.logError();
                return false;
            }

            updateGUI();

            if (myProject.modelPause || myProject.modelStop)
                return true;
        }

        if (myProject.saveOutput && firstHour <=1 && lastHour >= 23)
        {
            myProject.saveDailyOutput(myDate, outputPathHourly);
        }

        if (myProject.saveDailyState)
        {
            //save model state
        }
    }

    myProject.closeLogInfo();
    return true;
}


//-----------------  WATER FLUXES  -----------------


void MainWindow::on_actionRun_models_triggered()
{
    if (! myProject.isCriteria3DInitialized)
    {
        myProject.logError("Initialize 3D water fluxes before");
        return;
    }

    QDateTime firstTime, lastTime;
    if (! selectDates(firstTime, lastTime))
        return;

    myProject.isCrop = true;
    myProject.isWater = true;
    myProject.isSnow = false;
    runModels(firstTime, lastTime);

    updateDateTime();
    updateMaps();
}


//----------------- SNOW MODEL -----------------

void MainWindow::on_actionInitialize_snow_triggered()
{
    if (myProject.initializeSnowModel())
    {
        myProject.logInfoGUI("Snow model successfully initialized.");
    }
}


void MainWindow::on_actionRun_snow_model_triggered()
{
    if (! myProject.snowMaps.isInitialized)
    {
        if (! myProject.initializeSnowModel())
            return;
    }

    QDateTime firstTime, lastTime;
    if (! selectDates (firstTime, lastTime))
        return;

    myProject.isCrop = false;
    myProject.isWater = false;
    myProject.isSnow = true;
    startModels(firstTime, lastTime);
}


void MainWindow::on_actionSnow_settings_triggered()
{
    DialogSnowSettings dialogSnowSetting;
    dialogSnowSetting.setAllRainfallThresholdValue(myProject.snowModel.snowParameters.tempMaxWithSnow);
    dialogSnowSetting.setAllSnowThresholdValue(myProject.snowModel.snowParameters.tempMinWithRain);
    dialogSnowSetting.setWaterHoldingValue(myProject.snowModel.snowParameters.snowWaterHoldingCapacity);
    dialogSnowSetting.setSurfaceThickValue(myProject.snowModel.snowParameters.snowSkinThickness);
    dialogSnowSetting.setVegetationHeightValue(myProject.snowModel.snowParameters.snowVegetationHeight);
    dialogSnowSetting.setSoilAlbedoValue(myProject.snowModel.snowParameters.soilAlbedo);

    dialogSnowSetting.exec();
    if (dialogSnowSetting.result() != QDialog::Accepted)
    {
        return;
    }
    else
    {
        double tempMaxWithSnow = dialogSnowSetting.getAllRainfallThresholdValue();
        double tempMinWithRain = dialogSnowSetting.getAllSnowThresholdValue();
        double snowWaterHoldingCapacity = dialogSnowSetting.getWaterHoldingValue();
        double snowSkinThickness = dialogSnowSetting.getSurfaceThickValue();
        double snowVegetationHeight = dialogSnowSetting.getVegetationHeightValue();
        double soilAlbedo = dialogSnowSetting.getSoilAlbedoValue();
        myProject.snowModel.snowParameters.tempMinWithRain = tempMinWithRain;
        myProject.snowModel.snowParameters.tempMaxWithSnow = tempMaxWithSnow;
        myProject.snowModel.snowParameters.snowWaterHoldingCapacity = snowWaterHoldingCapacity;
        myProject.snowModel.snowParameters.snowSkinThickness = snowSkinThickness;
        myProject.snowModel.snowParameters.snowVegetationHeight = snowVegetationHeight;
        myProject.snowModel.snowParameters.soilAlbedo = soilAlbedo;
        if (!myProject.writeCriteria3DParameters())
        {
            myProject.logError("Error writing snow parameters");
        }
    }
    return;
}



void MainWindow::on_buttonModelPause_clicked()
{
    myProject.modelPause = true;
    ui->buttonModelPause->setDisabled(true);
    ui->buttonModelStart->setEnabled(true);
}


void MainWindow::on_buttonModelStop_clicked()
{
    myProject.modelStop = true;
    ui->groupBoxModel->setDisabled(true);
}

void MainWindow::on_buttonModelStart_clicked()
{
    if (myProject.modelPause)
    {
        myProject.modelPause = false;
        ui->buttonModelPause->setEnabled(true);
        ui->buttonModelStart->setDisabled(true);
        QDateTime newFirstTime = QDateTime(myProject.getCurrentDate(), QTime(myProject.getCurrentHour(), 0, 0));
        newFirstTime = newFirstTime.addSecs(3600);
        runModels(newFirstTime, myProject.modelLastTime);
    }
}

void MainWindow::on_actionSave_state_triggered()
{
    if (myProject.isProjectLoaded)
    {
        if (myProject.saveModelState())
        {
            myProject.logInfoGUI("Snow model successfully saved.");
        }
    }
    else
    {
        myProject.logError("Open a project before");
    }
    return;
}

void MainWindow::on_actionLoad_state_triggered()
{
    if (myProject.isProjectLoaded)
    {
        QList<QString> stateList = myProject.getAllSavedState();
        DialogLoadState dialogLoadState(stateList);
        if (dialogLoadState.result() != QDialog::Accepted)
        {
            return;
        }
        else
        {
            myProject.loadModelState(dialogLoadState.getSelectedState());
        }
    }
    else
    {
        myProject.logError("Open a project before");
    }
    return;
}
