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
#include "formTimePeriod.h"
#include "criteria3DProject.h"
#include "dialogSnowSettings.h"
#include "dialogLoadState.h"
#include "utilities.h"


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

    this->rubberBand = new RubberBand(QRubberBand::Rectangle, this->mapView);

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
    ui->flagSave_state_daily_step->setChecked(false);
    this->viewNotActivePoints = true;
    ui->flagView_not_active_points->setChecked(this->viewNotActivePoints);
    this->viewOutputPoints = true;
    this->viewNotActiveOutputPoints = true;
    ui->flagView_not_active_outputPoints->setChecked(this->viewNotActiveOutputPoints);
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
    this->rasterDEM->setVisible(false);
    this->mapView->scene()->addObject(this->rasterDEM);

    this->rasterOutput = new RasterObject(this->mapView);
    this->rasterOutput->setOpacity(this->ui->opacitySliderRasterOutput->value() / 100.0);
    this->rasterOutput->setColorLegend(this->outputRasterColorLegend);
    this->rasterOutput->setVisible(false);
    this->mapView->scene()->addObject(this->rasterOutput);

    this->updateCurrentVariable();
    this->updateDateTime();

    myProject.saveDailyState = ui->flagSave_state_daily_step->isChecked();

    this->setMouseTracking(true);

    this->testOutputPoints();
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
    outputRasterColorLegend->update();

    *startCenter = rasterDEM->getCurrentCenter();
}


void MainWindow::updateGUI()
{
    updateDateTime();
    emit rasterDEM->redrawRequested();
    emit rasterOutput->redrawRequested();
    qApp->processEvents();
}


// ------------------- SLOT -----------------------
void MainWindow::mouseMove(const QPoint& eventPos)
{
    if (! isInsideMap(eventPos)) return;

    // rubber band
    if (rubberBand != nullptr && rubberBand->isActive)
    {
        QPoint widgetPos = eventPos + QPoint(MAPBORDER, MAPBORDER);
        rubberBand->setGeometry(QRect(rubberBand->getOrigin(), widgetPos).normalized());
        return;
    }

    Position pos = this->mapView->mapToScene(eventPos);

    QString infoStr = "Lat:"+QString::number(pos.latitude())
                      + "  Lon:" + QString::number(pos.longitude());

    float value = NODATA;
    if (rasterOutput->visible())
    {
        value = rasterOutput->getValue(pos);
    }
    else if (rasterDEM->visible())
    {
        value = rasterDEM->getValue(pos);
    }
    if (! isEqual(value, NODATA))
        infoStr += "  Value:" + QString::number(double(value));

    this->ui->statusBar->showMessage(infoStr);
}


bool MainWindow::updateSelection(const QPoint& position)
{
    if (rubberBand == nullptr || !rubberBand->isActive || !rubberBand->isVisible() )
        return false;

    QPoint lastCornerOffset = getMapPos(position);
    QPoint firstCornerOffset = rubberBand->getOrigin() - QPoint(MAPBORDER, MAPBORDER);
    QPoint pixelTopLeft;
    QPoint pixelBottomRight;
    bool isAdd = false;

    if (firstCornerOffset.y() > lastCornerOffset.y())
    {
        if (firstCornerOffset.x() > lastCornerOffset.x())
        {
            // bottom to left
            pixelTopLeft = lastCornerOffset;
            pixelBottomRight = firstCornerOffset;
            isAdd = false;
        }
        else
        {
            // bottom to right
            pixelTopLeft = QPoint(firstCornerOffset.x(), lastCornerOffset.y());
            pixelBottomRight = QPoint(lastCornerOffset.x(), firstCornerOffset.y());
            isAdd = true;
        }
    }
    else
    {
        if (firstCornerOffset.x() > lastCornerOffset.x())
        {
            // top to left
            pixelTopLeft = QPoint(lastCornerOffset.x(), firstCornerOffset.y());
            pixelBottomRight = QPoint(firstCornerOffset.x(), lastCornerOffset.y());
            isAdd = false;
        }
        else
        {
            // top to right
            pixelTopLeft = firstCornerOffset;
            pixelBottomRight = lastCornerOffset;
            isAdd = true;
        }
    }

    QPointF topLeft = this->mapView->mapToScene(pixelTopLeft);
    QPointF bottomRight = this->mapView->mapToScene(pixelBottomRight);
    QRectF rectF(topLeft, bottomRight);

    for (int i = 0; i < meteoPointList.size(); i++)
    {
        if (rectF.contains(meteoPointList[i]->longitude(), meteoPointList[i]->latitude()))
        {
            if (isAdd)
            {
                myProject.meteoPoints[i].selected = true;
            }
            else
            {
                myProject.meteoPoints[i].selected = false;
            }
        }
    }

    for (int i = 0; i < outputPointList.size(); i++)
    {
        if (rectF.contains(outputPointList[i]->longitude(), outputPointList[i]->latitude()))
        {
            if (isAdd)
            {
                myProject.outputPoints[i].selected = true;
            }
            else
            {
                myProject.outputPoints[i].selected = false;
            }
        }
    }

    rubberBand->isActive = false;
    rubberBand->hide();

    return true;
}


void MainWindow::mouseReleaseEvent(QMouseEvent *event)
{
    this->updateMaps();
    if (this->updateSelection(event->pos()))
    {
        this->redrawMeteoPoints(currentPointsVisualization, false);
        this->redrawOutputPoints();
    }
}


void MainWindow::mouseDoubleClickEvent(QMouseEvent * event)
{
    QPoint mapPos = getMapPos(event->pos());
    if (! isInsideMap(mapPos)) return;

    Position newCenter = this->mapView->mapToScene(mapPos);

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
        if (contextMenuRequested(event->pos(), event->globalPos()))
            return;

        if (rubberBand != nullptr)
        {
            QPoint mapPos = getMapPos(event->pos());
            QPoint widgetPos = mapPos + QPoint(MAPBORDER, MAPBORDER);
            rubberBand->setOrigin(widgetPos);
            rubberBand->setGeometry(QRect(widgetPos, QSize()));
            rubberBand->isActive = true;
            rubberBand->show();
            return;
        }
    }
}


bool MainWindow::contextMenuRequested(QPoint localPos, QPoint globalPos)
{
    QMenu submenu;
    int nrItems = 0;

    QPoint mapPos = getMapPos(localPos);
    if (! isInsideMap(mapPos))
        return false;

    if (myProject.soilMap.isLoaded && ui->labelOutputRaster->text() == "Soil")
    {
        if (isSoil(mapPos))
        {
            submenu.addAction("Show soil data");
            nrItems++;
        }
    }

    if (nrItems == 0)
        return false;

    QAction* myAction = submenu.exec(globalPos);

    if (myAction)
    {
        if (myAction->text().contains("Show soil data") )
        {
            if (myProject.nrSoils > 0) {
                openSoilWidget(mapPos);
            }
            else {
                myProject.logError("Load soil database before.");
            }
        }
    }

    return true;
}


void MainWindow::addOutputPointsGUI()
{
    resetOutputPointMarkers();

    for (unsigned int i = 0; i < myProject.outputPoints.size(); i++)
    {
        SquareMarker* point = new SquareMarker(7, true, QColor((Qt::green)));
        point->setId(myProject.outputPoints[i].id);
        point->setLatitude(myProject.outputPoints[i].latitude);
        point->setLongitude(myProject.outputPoints[i].longitude);

        this->outputPointList.append(point);
        this->mapView->scene()->addObject(this->outputPointList[i]);
        outputPointList[i]->setToolTip();
    }

    redrawOutputPoints();
}


void MainWindow::testOutputPoints()
{
    myProject.outputPoints.clear();

    gis::Crit3DOutputPoint p;
    p.initialize("01", true, 44.5, 11.5, 50, myProject.gisSettings.utmZone);
    myProject.outputPoints.push_back(p);
    p.initialize("02", true, 44.6, 11.6, 50, myProject.gisSettings.utmZone);
    myProject.outputPoints.push_back(p);
    p.initialize("03", true, 44.6, 11.4, 50, myProject.gisSettings.utmZone);
    myProject.outputPoints.push_back(p);

    addOutputPointsGUI();
}


void MainWindow::addMeteoPoints()
{
    myProject.clearSelectedPoints();

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

        this->meteoPointList.append(point);
        this->mapView->scene()->addObject(this->meteoPointList[i]);

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
    resetMeteoPointMarkers();

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
    this->setProjectTileMap();

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

    this->drawMeteoPoints();
    // drawMeteoGrid();

    QString title = "CRITERIA3D";
    if (myProject.projectName != "")
        title += " - " + myProject.projectName;

    this->setWindowTitle(title);
}


void MainWindow::clearMaps_GUI()
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
    resetMeteoPointMarkers();
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

    this->setCurrentRasterInput(&(myProject.DEM));
    ui->labelInputRaster->setText(QString::fromStdString(getVariableString(noMeteoTerrain)));

    // center map
    gis::Crit3DGeoPoint* center = this->rasterDEM->getRasterCenter();
    mapView->centerOn(qreal(center->longitude), qreal(center->latitude));

    // resize map
    double size = double(this->rasterDEM->getRasterMaxSize());
    size = log2(1000 / size);
    mapView->setZoomLevel(quint8(size));
    mapView->centerOn(qreal(center->longitude), qreal(center->latitude));

    this->updateMaps();

    /*
    if (viewer3D != nullptr)
    {
        initializeViewer3D();
        //this->viewer3D->close();
        //this->viewer3D = nullptr;
    }
    */
}


// ----------------- DATE/TIME EDIT ---------------------------

void MainWindow::updateDateTime()
{
    this->ui->dateEdit->setDate(myProject.getCurrentDate());
    this->ui->timeEdit->setValue(myProject.getCurrentHour());
}

void MainWindow::on_dateEdit_dateChanged(const QDate &date)
{
    if (date != myProject.getCurrentDate())
    {
        myProject.loadMeteoPointsData(date, date, true, true, true);
        //myProject.loadMeteoGridData(date, date, true);
        myProject.setAllHourlyMeteoMapsComputed(false);
        myProject.setCurrentDate(date);
    }

    redrawMeteoPoints(currentPointsVisualization, true);
}

void MainWindow::on_dayBeforeButton_clicked()
{
    this->ui->dateEdit->setDate(this->ui->dateEdit->date().addDays(-1));
}

void MainWindow::on_dayAfterButton_clicked()
{
    this->ui->dateEdit->setDate(this->ui->dateEdit->date().addDays(1));
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
        clearMaps_GUI();
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
    clearMaps_GUI();

    myProject.loadCriteria3DProject(myProject.getApplicationPath() + "default.ini");

    drawProject();
}

QPoint MainWindow::getMapPos(const QPoint& pos)
{
    QPoint mapPoint;
    int x0 = ui->widgetMap->x();
    int y0 = ui->widgetMap->y() + ui->menuBar->height();
    mapPoint.setX(pos.x() - x0 - MAPBORDER);
    mapPoint.setY(pos.y() - y0 - MAPBORDER);

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

void MainWindow::resetMeteoPointMarkers()
{
    for (int i = 0; i < meteoPointList.size(); i++)
    {
        mapView->scene()->removeObject(meteoPointList[i]);
        delete meteoPointList[i];
    }

    meteoPointList.clear();
}


void MainWindow::resetOutputPointMarkers()
{
    for (int i = 0; i < outputPointList.size(); i++)
    {
        mapView->scene()->removeObject(outputPointList[i]);
        delete outputPointList[i];
    }

    outputPointList.clear();
}


void MainWindow::on_actionVariableQualitySpatial_triggered()
{
    myProject.checkSpatialQuality = ui->actionVariableQualitySpatial->isChecked();
    updateCurrentVariable();
}

void MainWindow::interpolateCurrentVariable()
{
    meteoVariable myVar = myProject.getCurrentVariable();

    if (myProject.interpolateHourlyMeteoVar(myVar, myProject.getCurrentTime()))
    {
        showMeteoVariable(myProject.getCurrentVariable());
    }
}

void MainWindow::updateCurrentVariable()
{
    std::string myString = getVariableString(myProject.getCurrentVariable());
    ui->labelVariable->setText(QString::fromStdString(myString));

    redrawMeteoPoints(currentPointsVisualization, true);
}


void MainWindow::redrawOutputPoints()
{
    for (unsigned int i = 0; i < myProject.outputPoints.size(); i++)
    {
        outputPointList[i]->setVisible(this->viewOutputPoints);

        if (myProject.outputPoints[i].selected)
        {
            outputPointList[i]->setFillColor(QColor(Qt::yellow));
        }
        else
        {
            if (myProject.outputPoints[i].active)
            {
                outputPointList[i]->setFillColor(QColor(Qt::green));
            }
            else
            {
                outputPointList[i]->setFillColor(QColor(Qt::red));
                if (! this->viewNotActiveOutputPoints)
                {
                    outputPointList[i]->setVisible(false);
                }
            }
        }


    }
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
                meteoPointList[i]->setRadius(5);
                meteoPointList[i]->setCurrentValue(NODATA);
                meteoPointList[i]->setToolTip();

                // color
                if (myProject.meteoPoints[i].selected)
                {
                    meteoPointList[i]->setFillColor(QColor(Qt::yellow));
                }
                else
                {
                    if (myProject.meteoPoints[i].active)
                    {
                        meteoPointList[i]->setFillColor(QColor(Qt::white));
                    }
                    else if (! myProject.meteoPoints[i].active)
                    {
                        meteoPointList[i]->setFillColor(QColor(Qt::red));
                    }
                }

                // hide not active points
                bool isVisible = (myProject.meteoPoints[i].active || viewNotActivePoints);
                meteoPointList[i]->setVisible(isVisible);
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
                myProject.getMeteoPointsRange(minimum, maximum, viewNotActivePoints);

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

                    // hide not active points
                    bool isVisible = (myProject.meteoPoints[i].active || viewNotActivePoints);
                    meteoPointList[i]->setVisible(isVisible);
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
    this->updateCurrentVariable();
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
    emit rasterDEM->redrawRequested();
}

void MainWindow::setCurrentRasterOutput(gis::Crit3DRasterGrid *myRaster)
{
    setOutputRasterVisible(true);

    rasterOutput->initializeUTM(myRaster, myProject.gisSettings, false);
    outputRasterColorLegend->colorScale = myRaster->colorScale;
    outputRasterColorLegend->repaint();
    emit rasterOutput->redrawRequested();
    updateMaps();
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


// ---------  3D VIEW (TODO)
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
        myProject.logError(ERROR_STR_MISSING_DEM);
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

// ---------------- SHOW METEOPOINTS --------------------------------

void MainWindow::on_flagView_not_active_points_toggled(bool isChecked)
{
    this->viewNotActivePoints = isChecked;
    redrawMeteoPoints(currentPointsVisualization, true);
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

void MainWindow::on_actionView_None_triggered()
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
        myProject.logError(ERROR_STR_MISSING_DEM);
        return;
    }
}

void MainWindow::on_actionView_Aspect_triggered()
{
    if (myProject.DEM.isLoaded)
    {
        setColorScale(noMeteoTerrain, myProject.radiationMaps->aspectMap->colorScale);
        setCurrentRasterOutput(myProject.radiationMaps->aspectMap);
        ui->labelOutputRaster->setText("Aspect °");
    }
    else
    {
        myProject.logError(ERROR_STR_MISSING_DEM);
        return;
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
        myProject.logError("Initialize 3D Model before.");
        return;
    }
}

void MainWindow::on_actionView_SoilMap_triggered()
{
    showSoilMap();
}

// -------------------- METEO VARIABLES -------------------------

bool MainWindow::checkMapVariable(bool isComputed)
{
    if (! myProject.DEM.isLoaded)
    {
        myProject.logError(ERROR_STR_MISSING_DEM);
        return false;
    }

    if (! isComputed)
    {
        myProject.logError("Compute meteo variables before.");
        return false;
    }

    return true;
}

void MainWindow::setMeteoVariable(meteoVariable myVar, gis::Crit3DRasterGrid *myGrid)
{   
    setOutputVariable(myVar, myGrid);
    myProject.setCurrentVariable(myVar);
    currentPointsVisualization = showCurrentVariable;
    updateCurrentVariable();
}

void MainWindow::setOutputVariable(meteoVariable myVar, gis::Crit3DRasterGrid *myGrid)
{
    setColorScale(myVar, myGrid->colorScale);
    setCurrentRasterOutput(myGrid);
    ui->labelOutputRaster->setText(QString::fromStdString(getVariableString(myVar)));
}

void MainWindow::showMeteoVariable(meteoVariable var)
{
    if (myProject.hourlyMeteoMaps == nullptr)
    {
        myProject.logError(ERROR_STR_MISSING_PROJECT);
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

    case directIrradiance:
        if (checkMapVariable(myProject.radiationMaps->getComputed()))
            setMeteoVariable(directIrradiance, myProject.radiationMaps->beamRadiationMap);
        break;

    case diffuseIrradiance:
        if (checkMapVariable(myProject.radiationMaps->getComputed()))
            setMeteoVariable(diffuseIrradiance, myProject.radiationMaps->diffuseRadiationMap);
        break;

    case reflectedIrradiance:
        if (checkMapVariable(myProject.radiationMaps->getComputed()))
            setMeteoVariable(reflectedIrradiance, myProject.radiationMaps->reflectedRadiationMap);
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

void MainWindow::on_actionViewMeteoVariable_None_triggered()
{
    setOutputRasterVisible(false);
}

void MainWindow::on_actionView_Air_temperature_triggered()
{
    showMeteoVariable(airTemperature);
}

void MainWindow::on_actionView_Transmissivity_triggered()
{
    showMeteoVariable(atmTransmissivity);
}

void MainWindow::on_actionView_Global_irradiance_triggered()
{
    showMeteoVariable(globalIrradiance);
}

void MainWindow::on_actionView_Beam_irradiance_triggered()
{
    showMeteoVariable(directIrradiance);
}

void MainWindow::on_actionView_Diffuse_irradiance_triggered()
{
    showMeteoVariable(diffuseIrradiance);
}

void MainWindow::on_actionView_Reflected_irradiance_triggered()
{
    showMeteoVariable(reflectedIrradiance);
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


// ------------------ SNOW VARIABLES ------------------------------

void MainWindow::showSnowVariable(meteoVariable var)
{
    if (! myProject.snowMaps.isInitialized)
    {
        myProject.logError("Initialize snow model before.");
        return;
    }

    switch(var)
    {
    case snowWaterEquivalent:
        setOutputVariable(snowWaterEquivalent, myProject.snowMaps.getSnowWaterEquivalentMap());
        break;

    case snowFall:
        setOutputVariable(snowFall, myProject.snowMaps.getSnowFallMap());
        break;

    case snowSurfaceTemperature:
        setOutputVariable(snowSurfaceTemperature, myProject.snowMaps.getSnowSurfaceTempMap());
        break;

    case snowInternalEnergy:
        setOutputVariable(snowInternalEnergy, myProject.snowMaps.getInternalEnergyMap());
        break;

    case snowSurfaceInternalEnergy:
        setOutputVariable(snowSurfaceInternalEnergy, myProject.snowMaps.getSurfaceInternalEnergyMap());
        break;

    case snowLiquidWaterContent:
        setOutputVariable(snowLiquidWaterContent, myProject.snowMaps.getLWContentMap());
        break;

    case snowAge:
        setOutputVariable(snowAge, myProject.snowMaps.getAgeOfSnowMap());
        break;

    case snowMelt:
        setOutputVariable(snowMelt, myProject.snowMaps.getSnowMeltMap());
        break;

    default:
    {}
    }
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

void MainWindow::on_actionView_Snow_liquid_water_content_triggered()
{
    showSnowVariable(snowLiquidWaterContent);
}

void MainWindow::on_actionView_Snow_age_triggered()
{
    showSnowVariable(snowAge);
}

void MainWindow::on_actionView_Snowmelt_triggered()
{
    showSnowVariable(snowMelt);
}

// ------------- TILES -----------------------------

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

void MainWindow::on_actionMapGoogle_satellite_triggered()
{
    this->setTileMapSource(WebTileSource::GOOGLE_Satellite);
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
    ui->actionMapGoogle_satellite->setChecked(false);
    ui->actionMapGoogle_hybrid_satellite->setChecked(false);

    if (tileSource == WebTileSource::OPEN_STREET_MAP)
    {
        ui->actionMapOpenStreetMap->setChecked(true);
    }
    else if (tileSource == WebTileSource::GOOGLE_Satellite)
    {
        ui->actionMapGoogle_satellite->setChecked(true);
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


// --------------- SHOW SOIL --------------------------------

bool MainWindow::isSoil(QPoint mapPos)
{
    if (! myProject.soilMap.isLoaded)
        return false;

    double x, y;
    Position geoPos = mapView->mapToScene(mapPos);
    gis::latLonToUtmForceZone(myProject.gisSettings.utmZone, geoPos.latitude(), geoPos.longitude(), &x, &y);

    int idSoil = myProject.getCrit3DSoilId(x, y);
    return (idSoil != NODATA);
}

void MainWindow::showSoilMap()
{
    if (myProject.soilMap.isLoaded)
    {
        setColorScale(airTemperature, myProject.soilMap.colorScale);
        setCurrentRasterOutput(&(myProject.soilMap));
        ui->labelOutputRaster->setText("Soil");
    }
    else
    {
        myProject.logError("Load a soil map before.");
    }
}


void MainWindow::openSoilWidget(QPoint mapPos)
{
    double x, y;
    Position geoPos = mapView->mapToScene(mapPos);
    gis::latLonToUtmForceZone(myProject.gisSettings.utmZone, geoPos.latitude(), geoPos.longitude(), &x, &y);
    QString soilCode = myProject.getCrit3DSoilCode(x, y);

    if (soilCode == "")
    {
        myProject.logError("No soil.");
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


// --------------- METEOPOINTS DB ----------------------------------

bool MainWindow::loadMeteoPointsDB_GUI(QString dbName)
{
    myProject.logInfoGUI("Load " + dbName);
    bool success = myProject.loadMeteoPointsDB(dbName);
    myProject.closeLogInfo();

    if (success)
        drawMeteoPoints();

    return success;
}

void MainWindow::on_actionLoad_MeteoPoints_triggered()
{
    QString dbName = QFileDialog::getOpenFileName(this, tr("Open meteo points DB"), "", tr("DB files (*.db)"));
    if (dbName != "") this->loadMeteoPointsDB_GUI(dbName);
}

void MainWindow::on_actionMeteoPointsImport_data_triggered()
{
    if (! myProject.meteoPointsLoaded)
    {
        myProject.logError(ERROR_STR_MISSING_DB);
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
        return;

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

    this->loadMeteoPointsDB_GUI(dbName);
}


// --------------- LOAD DATA ------------------------------------

void MainWindow::on_actionLoad_soil_map_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open soil map"), "", tr("ESRI grid files (*.flt)"));
    if (fileName == "") return;

    if (myProject.loadSoilMap(fileName))
    {
        showSoilMap();
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


//------------------- MENU INTERPOLATION --------------------
void MainWindow::on_actionInterpolationSettings_triggered()
{
    DialogInterpolation* myInterpolationDialog = new DialogInterpolation(&myProject);
    myInterpolationDialog->close();
}

void MainWindow::on_actionProxy_analysis_triggered()
{
    if (myProject.meteoPointsDbHandler == nullptr)
    {
        myProject.logError(ERROR_STR_MISSING_DB);
        return;
    }

    std::vector<Crit3DProxy> proxy = myProject.interpolationSettings.getCurrentProxy();
    if (proxy.size() == 0)
    {
        myProject.logError("No proxy loaded");
        return;
    }

    return myProject.showProxyGraph();
}

void MainWindow::on_actionCompute_hour_meteoVariables_triggered()
{
    if (myProject.nrMeteoPoints == 0)
    {
        myProject.logError(ERROR_STR_MISSING_DB);
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

void MainWindow::on_actionComputePeriod_meteoVariables_triggered()
{
    QDateTime firstTime, lastTime;
    if (! selectDates (firstTime, lastTime))
        return;

    myProject.isMeteo = true;
    myProject.isRadiation = false;
    myProject.isSnow = false;
    myProject.isCrop = false;
    myProject.isWater = false;

    startModels(firstTime, lastTime);
}


// ------------------------ MODEL CYCLE ----------------------------
bool selectDates(QDateTime &firstTime, QDateTime &lastTime)
{
    if (! myProject.meteoPointsLoaded)
    {
        myProject.logError(ERROR_STR_MISSING_DB);
        return false;
    }

    firstTime = myProject.getCurrentTime();
    firstTime = firstTime.addSecs(3600);
    lastTime = firstTime;
    lastTime.setTime(QTime(23,0,0));

    QDateTime firstDateH = myProject.meteoPointsDbHandler->getFirstDate(hourly);
    QDateTime lastDateH = myProject.meteoPointsDbHandler->getLastDate(hourly);

    FormTimePeriod formTimePeriod(&firstTime, &lastTime);
    formTimePeriod.setMinimumDate(firstDateH.date());
    formTimePeriod.setMaximumDate(lastDateH.date());
    formTimePeriod.show();

    if (formTimePeriod.exec() == QDialog::Rejected)
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
    if (! myProject.DEM.isLoaded)
    {
        myProject.logError(ERROR_STR_MISSING_DEM);
        return false;
    }

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
    if (! myProject.DEM.isLoaded)
    {
        myProject.logError(ERROR_STR_MISSING_DEM);
        return false;
    }

    // initialize
    myProject.hourlyMeteoMaps->initialize();
    myProject.radiationMaps->initialize();

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
            QDateTime myTime = QDateTime(myDate, QTime(hour, 0, 0), Qt::UTC);

            if (! myProject.modelHourlyCycle(myTime, outputPathHourly))
            {
                myProject.logError();
                return false;
            }

            this->updateGUI();

            if (myProject.modelPause || myProject.modelStop)
            {
                return true;
            }
        }

        if (myProject.saveOutput && firstHour <=1 && lastHour >= 23)
        {
            myProject.saveDailyOutput(myDate, outputPathHourly);
        }

        if (myProject.saveDailyState)
        {
            myProject.saveModelState();
        }
    }

    myProject.closeLogInfo();
    return true;
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
        QDateTime newFirstTime = QDateTime(myProject.getCurrentDate(), QTime(myProject.getCurrentHour(), 0, 0), Qt::UTC);
        newFirstTime = newFirstTime.addSecs(3600);
        runModels(newFirstTime, myProject.modelLastTime);
    }
}


//------------------- MENU SOLAR RADIATION MODEL -----------------
void MainWindow::on_actionRadiation_settings_triggered()
{
    DialogRadiation* myDialogRadiation = new DialogRadiation(&myProject);
    myDialogRadiation->close();
}

bool MainWindow::setRadiationAsCurrentVariable()
{
    if (myProject.nrMeteoPoints == 0)
    {
        myProject.logError(ERROR_STR_MISSING_DB);
        return false;
    }

    myProject.setCurrentVariable(globalIrradiance);
    currentPointsVisualization = showCurrentVariable;
    updateCurrentVariable();

    return true;
}

void MainWindow::on_actionRadiation_compute_current_hour_triggered()
{
    if (! setRadiationAsCurrentVariable())
        return;

    this->interpolateCurrentVariable();
}

void MainWindow::on_actionRadiation_run_model_triggered()
{
    if (! setRadiationAsCurrentVariable())
        return;

    QDateTime firstTime, lastTime;
    if (! selectDates (firstTime, lastTime))
        return;

    myProject.isMeteo = false;
    myProject.isRadiation = true;
    myProject.isSnow = false;
    myProject.isCrop = false;
    myProject.isWater = false;
    startModels(firstTime, lastTime);
}


//-------------------- MENU SNOW MODEL -----------------------
void MainWindow::on_actionSnow_initialize_triggered()
{
    if (myProject.initializeSnowModel())
    {
        myProject.logInfoGUI("Snow model successfully initialized.");
    }
}

void MainWindow::on_actionSnow_run_model_triggered()
{
    if (! myProject.snowMaps.isInitialized)
    {
        if (! myProject.initializeSnowModel())
            return;
    }

    QDateTime firstTime, lastTime;
    if (! selectDates (firstTime, lastTime))
        return;

    myProject.isMeteo = true;
    myProject.isRadiation = true;
    myProject.isSnow = true;
    myProject.isCrop = false;
    myProject.isWater = false;
    startModels(firstTime, lastTime);
}

void MainWindow::on_actionSnow_compute_current_hour_triggered()
{
    if (! myProject.snowMaps.isInitialized)
    {
        if (! myProject.initializeSnowModel())
            return;
    }

    QDateTime currentTime = myProject.getCurrentTime();

    myProject.isMeteo = true;
    myProject.isRadiation = true;
    myProject.isSnow = true;
    myProject.isCrop = false;
    myProject.isWater = false;
    startModels(currentTime, currentTime);
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


//----------------- MENU WATER FLUXES  -----------------
void MainWindow::on_actionCriteria3D_settings_triggered()
{
    // TODO
}

void MainWindow::on_actionCriteria3D_Initialize_triggered()
{
    myProject.initializeCriteria3DModel();
}

void MainWindow::on_actionCriteria3D_run_models_triggered()
{
    if (! myProject.isCriteria3DInitialized)
    {
        myProject.logError("Initialize 3D water fluxes before");
        return;
    }

    QDateTime firstTime, lastTime;
    if (! selectDates(firstTime, lastTime))
        return;

    myProject.isMeteo = true;
    myProject.isRadiation = true;
    myProject.isSnow = true;
    myProject.isCrop = true;
    myProject.isWater = true;
    runModels(firstTime, lastTime);

    updateDateTime();
    updateMaps();
}


//------------------- STATES ----------------------

void MainWindow::on_flagSave_state_daily_step_toggled(bool isChecked)
{
    myProject.saveDailyState = isChecked;
}

void MainWindow::on_actionSave_state_triggered()
{
    if (myProject.isProjectLoaded)
    {
        if (myProject.saveModelState())
        {
            myProject.logInfoGUI("State model successfully saved: " + myProject.getCurrentDate().toString()
                                 + " H:" + QString::number(myProject.getCurrentHour()));
        }
    }
    else
    {
        myProject.logError(ERROR_STR_MISSING_PROJECT);
    }
    return;
}

void MainWindow::on_actionLoad_state_triggered()
{
    if (! myProject.isProjectLoaded)
    {
        myProject.logError(ERROR_STR_MISSING_PROJECT);
        return;
    }

    QList<QString> stateList = myProject.getAllSavedState();
    if (stateList.size() == 0)
    {
        myProject.logError();
        return;
    }

    DialogLoadState dialogLoadState(stateList);
    if (dialogLoadState.result() != QDialog::Accepted)
        return;

    if (myProject.loadModelState(dialogLoadState.getSelectedState()))
    {
        updateDateTime();
        myProject.logInfoGUI("Model state successfully loaded: " + myProject.getCurrentDate().toString()
                             + " H:" + QString::number(myProject.getCurrentHour()));
    }
    else
    {
        myProject.logError();
    }
}


//-------------------- MENU METEO POINTS -----------------------------
void MainWindow::on_actionPoints_clear_selection_triggered()
{
    myProject.clearSelectedPoints();
    redrawMeteoPoints(currentPointsVisualization, false);
}

void MainWindow::on_actionPoints_activate_all_triggered()
{
    if (myProject.meteoPointsDbHandler == nullptr)
    {
        myProject.logError(ERROR_STR_MISSING_DB);
        return;
    }

    if (!myProject.meteoPointsDbHandler->setAllPointsActive())
    {
        myProject.logError("Failed to activate all points.");
        return;
    }

    for (int i = 0; i < myProject.nrMeteoPoints; i++)
    {
        myProject.meteoPoints[i].active = true;
    }

    myProject.clearSelectedPoints();
    redrawMeteoPoints(currentPointsVisualization, true);
}

void MainWindow::on_actionPoints_deactivate_all_triggered()
{
    if (myProject.meteoPointsDbHandler == nullptr)
    {
        myProject.logError(ERROR_STR_MISSING_DB);
        return;
    }

    if (!myProject.meteoPointsDbHandler->setAllPointsNotActive())
    {
        myProject.logError("Failed to deactivate all points.");
        return;
    }

    for (int i = 0; i < myProject.nrMeteoPoints; i++)
    {
        myProject.meteoPoints[i].active = false;
    }

    myProject.clearSelectedPoints();
    redrawMeteoPoints(currentPointsVisualization, true);
}

void MainWindow::on_actionPoints_activate_selected_triggered()
{
    if (myProject.setActiveStateSelectedPoints(true))
        redrawMeteoPoints(currentPointsVisualization, true);
}

void MainWindow::on_actionPoints_deactivate_selected_triggered()
{
    if (myProject.setActiveStateSelectedPoints(false))
        redrawMeteoPoints(currentPointsVisualization, true);
}

void MainWindow::on_actionPoints_activate_from_point_list_triggered()
{
    if (myProject.meteoPointsDbHandler == nullptr)
    {
        myProject.logError(ERROR_STR_MISSING_DB);
        return;
    }

    QString fileName = QFileDialog::getOpenFileName(this, tr("Open point list file"), "", tr("text files (*.txt)"));
    if (fileName == "") return;

    if (myProject.setActiveStatePointList(fileName, true))
        redrawMeteoPoints(currentPointsVisualization, true);
}

void MainWindow::on_actionPoints_deactivate_from_point_list_triggered()
{
    if (myProject.meteoPointsDbHandler == nullptr)
    {
        myProject.logError(ERROR_STR_MISSING_DB);
        return;
    }

    QString fileName = QFileDialog::getOpenFileName(this, tr("Open point list file"), "", tr("text files (*.txt)"));
    if (fileName == "") return;

    if (myProject.setActiveStatePointList(fileName, false))
        redrawMeteoPoints(currentPointsVisualization, true);
}

void MainWindow::on_actionPoints_activate_with_criteria_triggered()
{
    if (myProject.setActiveStateWithCriteria(true))
    {
        // reload meteoPoint, point properties table is changed
        QString dbName = myProject.dbPointsFileName;
        myProject.closeMeteoPointsDB();
        this->loadMeteoPointsDB_GUI(dbName);
    }
}

void MainWindow::on_actionPoints_deactivate_with_criteria_triggered()
{
    if (myProject.setActiveStateWithCriteria(false))
    {
        // reload meteoPoint, point properties table is changed
        QString dbName = myProject.dbPointsFileName;
        myProject.closeMeteoPointsDB();
        this->loadMeteoPointsDB_GUI(dbName);
    }
}



void MainWindow::on_actionPoints_deactivate_with_no_data_triggered()
{
    if (myProject.meteoPointsDbHandler == nullptr)
    {
        myProject.logError(ERROR_STR_MISSING_DB);
        return;
    }

    QList<QString> pointList;
    myProject.setProgressBar("Checking points...", myProject.nrMeteoPoints);
    for (int i = 0; i < myProject.nrMeteoPoints; i++)
    {
        myProject.updateProgressBar(i);
        if (myProject.meteoPoints[i].active)
        {
            bool existData = myProject.meteoPointsDbHandler->existData(&myProject.meteoPoints[i], daily) || myProject.meteoPointsDbHandler->existData(&myProject.meteoPoints[i], hourly);
            if (!existData)
            {
                pointList.append(QString::fromStdString(myProject.meteoPoints[i].id));
            }
        }
    }
    myProject.closeProgressBar();

    if (pointList.isEmpty())
    {
        myProject.logError("All active points have valid data.");
        return;
    }

    myProject.logInfoGUI("Deactive points...");
    bool isOk = myProject.meteoPointsDbHandler->setActiveStatePointList(pointList, false);
    myProject.closeLogInfo();

    if (! isOk)
    {
        myProject.logError("Failed to set to not active NODATA points");
        return;
    }

    for (int j = 0; j < pointList.size(); j++)
    {
        for (int i = 0; i < myProject.nrMeteoPoints; i++)
        {
            if (myProject.meteoPoints[i].id == pointList[j].toStdString())
            {
                myProject.meteoPoints[i].active = false;
            }
        }
    }

    redrawMeteoPoints(currentPointsVisualization, true);
}


void MainWindow::on_actionDelete_Points_Selected_triggered()
{
    if (myProject.meteoPointsDbHandler == nullptr)
    {
        myProject.logError(ERROR_STR_MISSING_DB);
        return;
    }

    QList<QString> pointList;
    for (int i = 0; i < myProject.nrMeteoPoints; i++)
    {
        if (myProject.meteoPoints[i].selected)
        {
            pointList << QString::fromStdString(myProject.meteoPoints[i].id);
        }
    }
    if (pointList.isEmpty())
    {
        myProject.logError("No meteo point selected.");
        return;
    }

    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, "Are you sure?" ,
                                  QString::number(pointList.size()) + " selected points will be deleted",
                                  QMessageBox::Yes|QMessageBox::No);
    if (reply == QMessageBox::Yes)
    {
        if (myProject.deleteMeteoPoints(pointList))
            drawMeteoPoints();
    }
}


void MainWindow::on_actionDelete_Points_NotActive_triggered()
{
    if (myProject.meteoPointsDbHandler == nullptr)
    {
        myProject.logError(ERROR_STR_MISSING_DB);
        return;
    }

    QList<QString> pointList;
    for (int i = 0; i < myProject.nrMeteoPoints; i++)
    {
        if (!myProject.meteoPoints[i].active)
        {
            pointList << QString::fromStdString(myProject.meteoPoints[i].id);
        }
    }
    if (pointList.isEmpty())
    {
        myProject.logError("All points are active");
        return;
    }

    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, "Are you sure?",
                                  QString::number(pointList.size()) + " not active points will be deleted",
                                  QMessageBox::Yes|QMessageBox::No);
    if (reply == QMessageBox::Yes)
    {
        if (myProject.deleteMeteoPoints(pointList))
            drawMeteoPoints();
    }
}


void MainWindow::on_actionPoints_delete_data_selected_triggered()
{
    if (myProject.meteoPointsDbHandler == nullptr)
    {
        myProject.logError(ERROR_STR_MISSING_DB);
        return;
    }

    QList<QString> pointList;
    for (int i = 0; i < myProject.nrMeteoPoints; i++)
    {
        if (myProject.meteoPoints[i].selected)
        {
            pointList << QString::fromStdString(myProject.meteoPoints[i].id);
        }
    }

    if (pointList.isEmpty())
    {
        myProject.logError("No meteo points selected.");
        return;
    }

    if (!myProject.deleteMeteoPointsData(pointList))
    {
        myProject.logError("Failed to delete data.");
    }

    QDate currentDate = myProject.getCurrentDate();
    myProject.loadMeteoPointsData(currentDate, currentDate, true, true, true);
}


void MainWindow::on_actionPoints_delete_data_not_active_triggered()
{
    if (myProject.meteoPointsDbHandler == nullptr)
    {
        myProject.logError(ERROR_STR_MISSING_DB);
        return;
    }

    QList<QString> pointList;
    for (int i = 0; i < myProject.nrMeteoPoints; i++)
    {
        if (!myProject.meteoPoints[i].active)
        {
            pointList << QString::fromStdString(myProject.meteoPoints[i].id);
        }
    }

    if (pointList.isEmpty())
    {
        myProject.logError("All meteo points are active.");
        return;
    }

    if (!myProject.deleteMeteoPointsData(pointList))
    {
        myProject.logError("Failed to delete data.");
    }

    QDate currentDate = myProject.getCurrentDate();
    myProject.loadMeteoPointsData(currentDate, currentDate, true, true, true);
}

void MainWindow::on_flagHide_outputPoints_toggled(bool isChecked)
{
    viewOutputPoints = !isChecked;
    redrawOutputPoints();
}

void MainWindow::on_flagView_not_active_outputPoints_toggled(bool isChecked)
{
    viewNotActiveOutputPoints = isChecked;
    redrawOutputPoints();
}

void MainWindow::on_actionOutputPoints_clear_selection_triggered()
{
    myProject.clearSelectedOutputPoints();
    redrawOutputPoints();
}

void MainWindow::on_actionOutputPoints_deactivate_all_triggered()
{
    for (unsigned int i = 0; i < myProject.outputPoints.size(); i++)
    {
        myProject.outputPoints[i].active = false;
    }

    // TODO Laura update file

    myProject.clearSelectedOutputPoints();
    redrawOutputPoints();
}

void MainWindow::on_actionOutputPoints_deactivate_selected_triggered()
{
    for (unsigned int i = 0; i < myProject.outputPoints.size(); i++)
    {
        if (myProject.outputPoints[i].selected)
        {
            myProject.outputPoints[i].active = false;
        }
    }

    // TODO Laura update file

    myProject.clearSelectedOutputPoints();
    redrawOutputPoints();
}

void MainWindow::on_actionOutputPoints_activate_all_triggered()
{
    for (unsigned int i = 0; i < myProject.outputPoints.size(); i++)
    {
        myProject.outputPoints[i].active = true;
    }

    // TODO Laura update file

    myProject.clearSelectedOutputPoints();
    redrawOutputPoints();
}

void MainWindow::on_actionOutputPoints_activate_selected_triggered()
{
    for (unsigned int i = 0; i < myProject.outputPoints.size(); i++)
    {
        if (myProject.outputPoints[i].selected)
        {
            myProject.outputPoints[i].active = true;
        }
    }

    // TODO Laura update file

    myProject.clearSelectedOutputPoints();
    redrawOutputPoints();
}

