/*!
    \copyright 2018 Fausto Tomei, Gabriele Antolini,
    Alberto Pistocchi, Marco Bittelli, Antonio Volta, Laura Costantini

    This file is part of CRITERIA3D.
    CRITERIA3D has been developed under contract issued by ARPAE Emilia-Romagna

    CRITERIA3D is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CRITERIA3D is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with CRITERIA3D.  If not, see <http://www.gnu.org/licenses/>.

    contacts:
    ftomei@arpae.it
    gantolini@arpae.it
*/

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
#include "dialogNewPoint.h"
#include "glWidget.h"
#include "dialogWaterFluxesSettings.h"
#include "utilities.h"

#include <QTime>


extern Crit3DProject myProject;

#define MAPBORDER 10
#define TOOLSWIDTH 270


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    viewer3D = nullptr;

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

    current3DlayerIndex = 0;
    view3DVariable = false;

    ui->flagView_values->setChecked(false);

    // show menu
    showPointsGroup = new QActionGroup(this);
    showPointsGroup->setExclusive(true);
    showPointsGroup->addAction(ui->actionView_PointsHide);
    showPointsGroup->addAction(ui->actionView_PointsLocation);
    showPointsGroup->addAction(ui->actionView_PointsCurrentVariable);
    showPointsGroup->setEnabled(false);

    this->setTileMapSource(WebTileSource::GOOGLE_Terrain);

    // Set start size and position
    this->startCenter = new Position (myProject.gisSettings.startLocation.longitude,
                                     myProject.gisSettings.startLocation.latitude, 0.0);
    this->mapView->setZoomLevel(8);
    this->mapView->centerOn(startCenter->lonLat());
    connect(this->mapView, SIGNAL(zoomLevelChanged(quint8)), this, SLOT(updateMaps()));
    connect(this->mapView, SIGNAL(mouseMoveSignal(QPoint)), this, SLOT(mouseMove(QPoint)));

    // Set raster objects
    this->rasterDEM = new RasterUtmObject(this->mapView);
    this->rasterDEM->setOpacity(this->ui->opacitySliderRasterInput->value() / 100.0);
    this->rasterDEM->setColorLegend(this->inputRasterColorLegend);
    this->rasterDEM->setVisible(false);
    this->mapView->scene()->addObject(this->rasterDEM);

    this->rasterOutput = new RasterUtmObject(this->mapView);
    this->rasterOutput->setOpacity(this->ui->opacitySliderRasterOutput->value() / 100.0);
    this->rasterOutput->setColorLegend(this->outputRasterColorLegend);
    this->rasterOutput->setVisible(false);
    this->mapView->scene()->addObject(this->rasterOutput);

    this->updateCurrentVariable();
    this->updateDateTime();

    myProject.setSaveDailyState(false);
    ui->flagSave_state_daily_step->setChecked(myProject.isSaveDailyState());

    myProject.setSaveEndOfRunState(false);
    ui->flagSave_state_endRun->setChecked(myProject.isSaveEndOfRunState());

    myProject.setSaveOutputPoints(false);
    myProject.setComputeOnlyPoints(false);
    ui->flagOutputPoints_save_output->setChecked(myProject.isSaveOutputPoints());
    ui->flagCompute_only_points->setChecked(myProject.getComputeOnlyPoints());

    this->setMouseTracking(true);

    connect(&myProject, &Crit3DProject::updateOutputSignal, this, &MainWindow::updateOutputMap);
}


void MainWindow::resizeEvent(QResizeEvent * event)
{
    Q_UNUSED(event)

    const int INFOHEIGHT = 42;
    const int STEPY = 24;
    int x1 = this->width() - TOOLSWIDTH - MAPBORDER;
    int dy = ui->groupBoxModel->height() + ui->groupBoxMeteoPoints->height() + ui->groupBoxDEM->height() + ui->groupBoxVariableMap->height() + STEPY*3;
    int y1 = (this->height() - INFOHEIGHT - dy) / 2;

    ui->widgetMap->setGeometry(0, 0, x1, this->height() - INFOHEIGHT);
    mapView->resize(ui->widgetMap->size());

    ui->groupBoxModel->move(x1, y1);
    ui->groupBoxModel->resize(TOOLSWIDTH, ui->groupBoxModel->height());
    y1 += ui->groupBoxModel->height() + STEPY;

    ui->groupBoxDEM->move(x1, y1);
    ui->groupBoxDEM->resize(TOOLSWIDTH, ui->groupBoxDEM->height());
    y1 += ui->groupBoxDEM->height() + STEPY;

    ui->groupBoxMeteoPoints->move(x1, y1);
    ui->groupBoxMeteoPoints->resize(TOOLSWIDTH, ui->groupBoxMeteoPoints->height());
    y1 += ui->groupBoxMeteoPoints->height() + STEPY;

    ui->groupBoxVariableMap->move(x1, y1);
    ui->groupBoxVariableMap->resize(TOOLSWIDTH, ui->groupBoxVariableMap->height());
    this->updateMaps();
}


// ------------------- SLOT -----------------------

void MainWindow::updateMaps()
{
    rasterDEM->updateCenter();
    rasterOutput->updateCenter();
    inputRasterColorLegend->update();
    outputRasterColorLegend->update();

    *startCenter = rasterDEM->getCurrentCenter();
}


void MainWindow::updateOutputMap()
{
    updateDateTime();
    updateModelTime();

    if (myProject.isCriteria3DInitialized)
    {
        myProject.getCriteria3DMap(myProject.criteria3DMap, current3DVariable, current3DlayerIndex);
    }

    emit rasterOutput->redrawRequested();
    outputRasterColorLegend->update();
    qApp->processEvents();
}


void MainWindow::mouseMove(QPoint eventPos)
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
                myProject.outputPoints[unsigned(i)].selected = true;
            }
            else
            {
                myProject.outputPoints[unsigned(i)].selected = false;
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
        if (contextMenuRequested(event->pos()))
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


bool MainWindow::contextMenuRequested(QPoint localPos)
{
    QMenu contextMenu;
    int nrItems = 0;

    QPoint mapPos = getMapPos(localPos);
    if (! isInsideMap(mapPos))
        return false;

    if (myProject.soilMap.isLoaded)
    {
        if (isSoil(mapPos))
        {
            contextMenu.addAction("View soil data");
            nrItems++;
        }
    }
    if (myProject.landUseMap.isLoaded && myProject.landUnitList.size() != 0)
    {
        if (isLandUse(mapPos))
        {
            contextMenu.addAction("View land use");
            nrItems++;
            contextMenu.addAction("View crop");
            nrItems++;
        }
    }

    if (nrItems == 0)
        return false;

    QAction *selection =  contextMenu.exec(QCursor::pos());

    if (selection != nullptr)
    {
        if (selection->text().contains("View soil data"))
        {
            if (myProject.nrSoils > 0) {
                openSoilWidget(mapPos);
            }
            else
            {
                myProject.logError("Load soil database before.");
            }
        }

        if (selection->text().contains("View land use"))
        {
            Position geoPos = mapView->mapToScene(mapPos);
            int id = myProject.getLandUnitIdGeo(geoPos.latitude(), geoPos.longitude());
            if (id != NODATA)
            {
                int index = getLandUnitIndex(myProject.landUnitList, id);
                if (index != NODATA)
                {
                    Crit3DLandUnit landUnit = myProject.landUnitList[index];

                    QString infoStr = "LAND UNIT: " + QString::number(landUnit.id) + "\n\n";
                    infoStr += "name: " + landUnit.name + "\n";

                    if (landUnit.description != "" && landUnit.description != landUnit.name)
                        infoStr += "description: " + landUnit.description + "\n";

                    infoStr += "land use: " + landUnit.idLandUse + "\n";
                    infoStr += "crop: " + landUnit.idCrop + "\n";

                    infoStr += "roughness: " + QString::number(landUnit.roughness) + " [s m-1/3]\n";
                    infoStr += "pond: " + QString::number(landUnit.pond) + " [m]";

                    myProject.logInfoGUI(infoStr);
                }
            }
        }

        if (selection->text().contains("View crop"))
        {
            Position geoPos = mapView->mapToScene(mapPos);
            int id = myProject.getLandUnitIdGeo(geoPos.latitude(), geoPos.longitude());
            if (id != NODATA)
            {
                int index = getLandUnitIndex(myProject.landUnitList, id);
                if (index != NODATA)
                {
                    QString infoStr;
                    if (myProject.landUnitList[index].idCrop != "")
                    {
                        // same index of landUnitsList
                        Crit3DCrop crop = myProject.cropList[index];
                        infoStr = "CROP: " + QString::fromStdString(crop.idCrop) + "\n";
                        infoStr += "name: " + QString::fromStdString(crop.name) + "\n";
                        infoStr += "LAI min: " + QString::number(crop.LAImin) + "\n";
                        infoStr += "LAI max: " + QString::number(crop.LAImax) + "\n";
                        infoStr += "thermal threshold: " + QString::number(crop.thermalThreshold) + " °C\n";
                        infoStr += "degree days (phase 1): " + QString::number(crop.degreeDaysIncrease) + " °C\n";
                        infoStr += "degree days (phase 2): " + QString::number(crop.degreeDaysDecrease) + " °C\n";
                        infoStr += "kcmax: " + QString::number(crop.kcMax) + "\n";
                    }
                    else
                    {
                        infoStr = "CROP: none";
                    }

                    myProject.logInfoGUI(infoStr);
                }
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
        point->setId(QString::fromStdString(myProject.outputPoints[i].id));
        point->setLatitude(myProject.outputPoints[i].latitude);
        point->setLongitude(myProject.outputPoints[i].longitude);

        this->outputPointList.append(point);
        this->mapView->scene()->addObject(this->outputPointList[i]);
        outputPointList[i]->setToolTip();
    }

    redrawOutputPoints();
}


void MainWindow::drawWindVector(int i)
{
    float dx = myProject.meteoPoints[i].getMeteoPointValue(myProject.getCrit3DCurrentTime(),
                                                           windVectorX,  myProject.meteoSettings);
    float dy = myProject.meteoPoints[i].getMeteoPointValue(myProject.getCrit3DCurrentTime(),
                                                           windVectorY,  myProject.meteoSettings);
    if (isEqual(dx, NODATA) || isEqual(dy, NODATA))
        return;

    ArrowObject* arrow = new ArrowObject(qreal(dx * 10), qreal(dy * 10), QColor(Qt::black));
    arrow->setLatitude(myProject.meteoPoints[i].latitude);
    arrow->setLongitude(myProject.meteoPoints[i].longitude);
    windVectorList.append(arrow);

    mapView->scene()->addObject(windVectorList.last());
}


void MainWindow::addMeteoPoints()
{
    myProject.clearSelectedPoints();

    for (int i = 0; i < myProject.nrMeteoPoints; i++)
    {
        // default: white
        StationMarker* point = new StationMarker(5.0, true, QColor(Qt::white));

        if (myProject.meteoPoints[i].lapseRateCode == secondary)
        {
            point->setFillColor(QColor(Qt::black));
        }
        else if (myProject.meteoPoints[i].lapseRateCode == supplemental)
        {
            point->setFillColor(QColor(Qt::gray));
        }

        point->setFlag(MapGraphicsObject::ObjectIsMovable, false);
        point->setLatitude(myProject.meteoPoints[i].latitude);
        point->setLongitude(myProject.meteoPoints[i].longitude);
        point->setId(myProject.meteoPoints[i].id);
        point->setName(myProject.meteoPoints[i].name);
        point->setDataset(myProject.meteoPoints[i].dataset);
        point->setAltitude(myProject.meteoPoints[i].point.z);
        point->setMunicipality(myProject.meteoPoints[i].municipality);
        point->setCurrentValue(qreal(myProject.meteoPoints[i].currentValue));
        point->setQuality(myProject.meteoPoints[i].quality);
        point->setLapseRateCode(myProject.meteoPoints[i].lapseRateCode);

        this->meteoPointList.append(point);
        this->mapView->scene()->addObject(this->meteoPointList[i]);

        point->setToolTip();
        connect(point, SIGNAL(newStationClicked(std::string, std::string, std::string, double, std::string, bool)), this, SLOT(callNewMeteoWidget(std::string, std::string, std::string, double, std::string, bool)));
        connect(point, SIGNAL(appendStationClicked(std::string, std::string, std::string, double, std::string, bool)), this, SLOT(callAppendMeteoWidget(std::string, std::string, std::string, double, std::string, bool)));
    }
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


void MainWindow::drawMeteoPoints()
{
    resetMeteoPointMarkers();
    clearWindVectorObjects();

    if (! myProject.meteoPointsLoaded || myProject.nrMeteoPoints == 0)
    {
        ui->groupBoxMeteoPoints->setEnabled(false);
        return;
    }

    addMeteoPoints();
    ui->groupBoxMeteoPoints->setEnabled(true);

    loadMeteoPointsDataSingleDay(myProject.getCurrentDate(), true);

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
        // default: Google terrain
        this->setTileMapSource(WebTileSource::GOOGLE_Terrain);
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
    this->addOutputPointsGUI();

    QString title = "CRITERIA3D";
    if (myProject.projectName != "")
        title += " - " + myProject.projectName;

    this->setWindowTitle(title);
}


void MainWindow::clearRaster_GUI()
{
    rasterOutput->clear();
    rasterDEM->clear();

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
    resetOutputPointMarkers();
    clearWindVectorObjects();
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
    Position center = this->rasterDEM->getRasterCenter();
    mapView->centerOn(center.longitude(), center.latitude());

    // resize map
    double size = double(this->rasterDEM->getRasterMaxSize());
    size = log2(1000 / size);
    mapView->setZoomLevel(quint8(size));
    mapView->centerOn(center.longitude(), center.latitude());

    this->updateMaps();
}


// ----------------- DATE/TIME EDIT ---------------------------

void MainWindow::updateDateTime()
{
    this->ui->dateEdit->setDate(myProject.getCurrentDate());
    this->ui->timeEdit->setValue(myProject.getCurrentHour()); 
}


void MainWindow::updateModelTime()
{
    QDate date = myProject.getCurrentDate();
    int hour = myProject.getCurrentHour();
    int minutes = int(floor(myProject.currentSeconds / 60));
    int seconds = myProject.currentSeconds - (minutes * 60);
    if (minutes == 60)
    {
        hour++;
        if (hour == 24)
        {
            date = date.addDays(1);
            hour = 0;
        }
        minutes = 0;
    }
    QDateTime currentDateTime;
    currentDateTime.setDate(date);
    currentDateTime.setTime(QTime(hour, minutes, seconds));
    this->ui->modelTimeEdit->setText(currentDateTime.toString("yyyy-MM-dd HH:mm:ss"));
}


void MainWindow::loadMeteoPointsDataSingleDay(const QDate &date, bool showInfo)
{
    myProject.loadMeteoPointsData(date.addDays(-1), date, true, true, showInfo);
}


void MainWindow::on_dateEdit_dateChanged(const QDate &date)
{
    if (date != myProject.getCurrentDate())
    {
        loadMeteoPointsDataSingleDay(date, true);
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
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Digital Elevation Model"), "",
                                    tr("ESRI float (*.flt);; ENVI image (*.img)"));

    if (fileName == "") return;

    clearRaster_GUI();

    if (! myProject.loadDEM(fileName)) return;

    renderDEM();
}


void MainWindow::on_actionOpenProject_triggered()
{
    QString projectPath = myProject.getDefaultPath() + PATH_PROJECT;
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open project file"), projectPath, tr("ini files (*.ini)"));
    if (fileName == "") return;

    clearMeteoPoints_GUI();
    clearRaster_GUI();

    if (! myProject.loadCriteria3DProject(fileName))
    {
        myProject.logError("Error opening project: " + myProject.errorString);
        myProject.loadCriteria3DProject(myProject.getApplicationPath() + "default.ini");
    }

    drawProject();
}

void MainWindow::on_actionCloseProject_triggered()
{
    clearMeteoPoints_GUI();
    clearRaster_GUI();

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


void MainWindow::clearWindVectorObjects()
{
    for (int i = 0; i < windVectorList.size(); i++)
    {
        mapView->scene()->removeObject(windVectorList[i]);
        delete windVectorList[i];
    }

    windVectorList.clear();
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
    for (int i = 0; i < int(myProject.outputPoints.size()); i++)
    {
        outputPointList[i]->setVisible(this->viewOutputPoints);

        if (myProject.outputPoints[unsigned(i)].selected)
        {
            outputPointList[i]->setFillColor(QColor(Qt::yellow));
        }
        else
        {
            if (myProject.outputPoints[unsigned(i)].active)
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

    clearWindVectorObjects();

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

                // set color - default is white
                meteoPointList[i]->setFillColor(Qt::white);

                if (myProject.meteoPoints[i].selected)
                {
                    meteoPointList[i]->setFillColor(Qt::yellow);
                }
                else if (! myProject.meteoPoints[i].active)
                {
                    meteoPointList[i]->setFillColor(Qt::red);
                }
                else
                {
                    if (myProject.meteoPoints[i].lapseRateCode == secondary)
                    {
                        meteoPointList[i]->setFillColor(QColor(Qt::black));
                    }
                    else if (myProject.meteoPoints[i].lapseRateCode == supplemental)
                    {
                        meteoPointList[i]->setFillColor(QColor(Qt::gray));
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
            meteoVariable currentVar = myProject.getCurrentVariable();

            this->ui->actionView_PointsCurrentVariable->setChecked(true);
            // quality control
            std::string errorStdStr;
            checkData(myProject.quality, currentVar,
                      myProject.meteoPoints, myProject.nrMeteoPoints, myProject.getCrit3DCurrentTime(),
                      &(myProject.qualityInterpolationSettings), myProject.meteoSettings,
                      &(myProject.climateParameters), myProject.checkSpatialQuality, errorStdStr);

            if (updateColorSCale)
            {
                float minimum, maximum;
                myProject.getMeteoPointsRange(minimum, maximum, viewNotActivePoints);

                myProject.meteoPointsColorScale->setRange(minimum, maximum);
            }

            roundColorScale(myProject.meteoPointsColorScale, 4, true);
            setColorScale(currentVar, myProject.meteoPointsColorScale);
            bool isWindVector = (currentVar == windVectorIntensity || currentVar == windVectorDirection);

            for (int i = 0; i < myProject.nrMeteoPoints; i++)
            {
                if (int(myProject.meteoPoints[i].currentValue) != NODATA)
                {
                    if (myProject.meteoPoints[i].quality == quality::accepted)
                    {
                        meteoPointList[i]->setRadius(5);
                        Crit3DColor *myColor = myProject.meteoPointsColorScale->getColor(myProject.meteoPoints[i].currentValue);
                        meteoPointList[i]->setFillColor(QColor(myColor->red, myColor->green, myColor->blue));
                        meteoPointList[i]->setOpacity(1.0);
                        if (isWindVector)
                            drawWindVector(i);
                    }
                    else
                    {
                        // Wrong data
                        meteoPointList[i]->setRadius(10);
                        meteoPointList[i]->setFillColor(Qt::black);
                        meteoPointList[i]->setOpacity(0.5);
                    }

                    meteoPointList[i]->setCurrentValue(qreal(myProject.meteoPoints[i].currentValue));
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
    myProject.setCurrentVariable(chooseMeteoVariable(myProject));
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

    rasterDEM->initialize(myRaster, myProject.gisSettings);
    inputRasterColorLegend->colorScale = myRaster->colorScale;

    inputRasterColorLegend->repaint();
    emit rasterDEM->redrawRequested();
}

void MainWindow::setCurrentRasterOutput(gis::Crit3DRasterGrid *myRaster)
{
    setOutputRasterVisible(true);

    rasterOutput->initialize(myRaster, myProject.gisSettings);
    outputRasterColorLegend->colorScale = myRaster->colorScale;

    emit rasterOutput->redrawRequested();
    outputRasterColorLegend->update();

    rasterOutput->updateCenter();
    view3DVariable = (myRaster == &(myProject.criteria3DMap));
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
        myProject.radiationMaps->aspectMap->colorScale->setMinimum(0);
        myProject.radiationMaps->aspectMap->colorScale->setMaximum(360);
        myProject.radiationMaps->aspectMap->colorScale->setRangeBlocked(true);
        setCircolarScale(myProject.radiationMaps->aspectMap->colorScale);
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
        setBlackScale(myProject.boundaryMap.colorScale);
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


void MainWindow::on_actionHide_soil_map_triggered()
{
    if (ui->labelOutputRaster->text() == "Soil")
    {
        setOutputRasterVisible(false);
    }
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
    setOutputMeteoVariable(myVar, myGrid);
    myProject.setCurrentVariable(myVar);
    currentPointsVisualization = showCurrentVariable;
    updateCurrentVariable();
}

void MainWindow::setOutputMeteoVariable(meteoVariable myVar, gis::Crit3DRasterGrid *myGrid)
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
        setOutputMeteoVariable(snowWaterEquivalent, myProject.snowMaps.getSnowWaterEquivalentMap());
        break;

    case snowSurfaceTemperature:
        setOutputMeteoVariable(snowSurfaceTemperature, myProject.snowMaps.getSnowSurfaceTempMap());
        break;

    case snowInternalEnergy:
        setOutputMeteoVariable(snowInternalEnergy, myProject.snowMaps.getInternalEnergyMap());
        break;

    case snowSurfaceEnergy:
        setOutputMeteoVariable(snowSurfaceEnergy, myProject.snowMaps.getSurfaceEnergyMap());
        break;

    case snowLiquidWaterContent:
        setOutputMeteoVariable(snowLiquidWaterContent, myProject.snowMaps.getLWContentMap());
        break;

    case snowAge:
        setOutputMeteoVariable(snowAge, myProject.snowMaps.getAgeOfSnowMap());
        break;

    case snowFall:
        setOutputMeteoVariable(snowFall, myProject.snowMaps.getSnowFallMap());
        break;

    case snowMelt:
        setOutputMeteoVariable(snowMelt, myProject.snowMaps.getSnowMeltMap());
        break;

    case sensibleHeat:
        setOutputMeteoVariable(sensibleHeat, myProject.snowMaps.getSensibleHeatMap());
        break;

    case latentHeat:
        setOutputMeteoVariable(latentHeat, myProject.snowMaps.getLatentHeatMap());
        break;

    default:
    {}
    }
}

void MainWindow::on_actionView_Snow_water_equivalent_triggered()
{
    showSnowVariable(snowWaterEquivalent);
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
    showSnowVariable(snowSurfaceEnergy);
}

void MainWindow::on_actionView_Snow_liquid_water_content_triggered()
{
    showSnowVariable(snowLiquidWaterContent);
}

void MainWindow::on_actionView_Snow_age_triggered()
{
    showSnowVariable(snowAge);
}

void MainWindow::on_actionView_Snow_fall_triggered()
{
    showSnowVariable(snowFall);
}

void MainWindow::on_actionView_Snowmelt_triggered()
{
    showSnowVariable(snowMelt);
}

void MainWindow::on_actionView_Snow_sensible_heat_triggered()
{
    showSnowVariable(sensibleHeat);
}

void MainWindow::on_actionView_Snow_latent_heat_triggered()
{
    showSnowVariable(latentHeat);
}


// ------------- CROP MAPS ---------------------------------------------------
void MainWindow::on_actionView_degree_days_triggered()
{
    if (! myProject.isCropInitialized)
    {
        myProject.logError("Initialize crop before.");
        return;
    }

    setOutputMeteoVariable(dailyHeatingDegreeDays, &(myProject.degreeDaysMap));
}


void MainWindow::on_actionView_Crop_LAI_triggered()
{
    if (! myProject.isCropInitialized)
    {
        myProject.logError("Initialize crop before.");
        return;
    }

    setOutputMeteoVariable(leafAreaIndex, &(myProject.laiMap));
}


// ------------- TILES -------------------------------------------------------

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


// --------------- SOIL AND LAND USE --------------------------------

bool MainWindow::isSoil(QPoint mapPos)
{
    if (! myProject.soilMap.isLoaded)
        return false;

    double x, y;
    Position geoPos = mapView->mapToScene(mapPos);
    gis::latLonToUtmForceZone(myProject.gisSettings.utmZone, geoPos.latitude(), geoPos.longitude(), &x, &y);

    int idSoil = myProject.getSoilId(x, y);
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


bool MainWindow::isLandUse(QPoint mapPos)
{
    if (! myProject.landUseMap.isLoaded)
        return false;

    Position geoPos = mapView->mapToScene(mapPos);
    return (myProject.getLandUnitIdGeo(geoPos.latitude(), geoPos.longitude()) != NODATA);
}


void MainWindow::showLandUseMap()
{
    if (myProject.landUseMap.isLoaded)
    {
        setColorScale(noMeteoTerrain, myProject.landUseMap.colorScale);
        setCurrentRasterOutput(&(myProject.landUseMap));
        ui->labelOutputRaster->setText("Land use");
    }
    else
    {
        myProject.logError("Load a land use map before.");
    }
}


void MainWindow::openSoilWidget(QPoint mapPos)
{
    double x, y;
    Position geoPos = mapView->mapToScene(mapPos);
    gis::latLonToUtmForceZone(myProject.gisSettings.utmZone, geoPos.latitude(), geoPos.longitude(), &x, &y);
    QString soilCode = myProject.getSoilCode(x, y);

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
        if (! openDbSoil(dbSoilName, dbSoil, myProject.errorString))
        {
            myProject.logError();
            return;
        }
        soilWidget = new Crit3DSoilWidget();
        soilWidget->show();
        soilWidget->setDbSoil(dbSoil, soilCode);
    }
}


// --------------- METEOPOINTS ----------------------------------

bool MainWindow::loadMeteoPointsDB_GUI(QString dbName)
{
    bool success = myProject.loadMeteoPointsDB(dbName);

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
    if (!myProject.meteoPointsDbHandler->getFieldList("point_properties", pointPropertiesList))
    {
        myProject.logError("Error in read table point_properties");
        return;
    }
    QList<QString> csvFields;
    QList<QList<QString>> csvData;
    if (! parseCSV(csvFileName, csvFields, csvData, myProject.errorString))
    {
        myProject.logError("Error in parse properties: " + myProject.errorString);
        return;
    }

    DialogPointProperties dialogPointProp(pointPropertiesList, csvFields);
    if (dialogPointProp.result() != QDialog::Accepted)
    {
        return;
    }
    else
    {
        QList<QString> joinedPropertiesList = dialogPointProp.getJoinedList();
        if (! myProject.writeMeteoPointsProperties(joinedPropertiesList, csvFields, csvData))
        {
            myProject.logError("Error in write points properties");
            return;
        }
    }

    loadMeteoPointsDB_GUI(dbName);
}


// ----------------------- SOIL and CROP DATA ------------------------------------

void MainWindow::on_actionLoad_soil_map_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open soil map"), "",
                                                    tr("ESRI float (*.flt);; ENVI image (*.img)"));
    if (fileName == "") return;

    if (myProject.loadSoilMap(fileName))
    {
        showSoilMap();
    }
}

void MainWindow::on_actionLoad_soil_data_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Load soil data"), "", tr("SQLite files (*.db)"));
    if (fileName == "") return;

    myProject.loadSoilDatabase(fileName);
}


void MainWindow::on_actionLoad_crop_data_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Load crop and land units data"), "", tr("SQLite files (*.db)"));
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

void MainWindow::on_actionComputeHour_meteoVariables_triggered()
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

    myProject.processes.initialize();
    myProject.processes.computeMeteo = true;
    myProject.processes.computeRadiation = true;

    startModels(firstTime, lastTime);
}


// ------------------------ MODEL CYCLE ---------------------------

bool selectDates(QDateTime &firstTime, QDateTime &lastTime)
{
    if (! myProject.meteoPointsLoaded)
    {
        myProject.logError(ERROR_STR_MISSING_DB);
        return false;
    }

    firstTime.setTimeZone(QTimeZone::utc());
    if (myProject.getCurrentHour() == 24)
    {
        firstTime.setDate(myProject.getCurrentDate().addDays(1));
        firstTime.setTime(QTime(0,0,0,0));
    }
    else
    {
        firstTime.setDate(myProject.getCurrentDate());
        firstTime.setTime(QTime(myProject.getCurrentHour(),0,0,0));
    }
    firstTime = firstTime.addSecs(HOUR_SECONDS);

    lastTime.setTimeZone(QTimeZone::utc());
    lastTime = firstTime;
    lastTime.setTime(QTime(23,0,0));

    FormTimePeriod formTimePeriod(&firstTime, &lastTime);
    formTimePeriod.setMinimumDate(myProject.meteoPointsDbFirstTime.date());
    formTimePeriod.setMaximumDate(myProject.meteoPointsDbLastTime.date());
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

    if (myProject.processes.computeSnow && (! myProject.snowMaps.isInitialized))
    {
        myProject.logError("Initialize Snow model or load a state before.");
        return false;
    }

    if (myProject.processes.computeWater && (! myProject.isCriteria3DInitialized))
    {
        myProject.logError("Initialize 3D water fluxes or load a state before.");
        return false;
    }

    if (myProject.processes.computeCrop)
    {
        if (myProject.landUnitList.size() == 0)
        {
            myProject.logError("load land units map before.");
            return false;
        }
    }

    // Load meteo data
    myProject.logInfoGUI("Loading meteo data...");
    if (! myProject.loadMeteoPointsData(firstTime.date().addDays(-1), lastTime.date().addDays(+1), true, false, false))
    {
        myProject.logError();
        return false;
    }
    myProject.closeLogInfo();

    // output points
    if (myProject.isSaveOutputPoints())
    {
        if (! myProject.writeOutputPointsTables())
        {
            myProject.logError();
            return false;
        }
    }

    // set model interface
    myProject.modelFirstTime = firstTime;
    myProject.modelLastTime = lastTime;
    myProject.modelPause = false;
    myProject.modelStop = false;

    ui->groupBoxModel->setEnabled(true);
    ui->buttonModelPause->setEnabled(true);
    ui->buttonModelStart->setDisabled(true);
    ui->buttonModelStop->setEnabled(true);

    return myProject.runModels(firstTime, lastTime);
}


void MainWindow::on_buttonModelPause_clicked()
{
    myProject.modelPause = true;
    ui->buttonModelPause->setDisabled(true);
    ui->buttonModelStart->setEnabled(true);
    ui->buttonModelStop->setEnabled(true);
    qApp->processEvents();
}


void MainWindow::on_buttonModelStop_clicked()
{
    myProject.modelStop = true;
    ui->buttonModelPause->setDisabled(true);
    ui->buttonModelStart->setDisabled(true);
    ui->buttonModelStop->setDisabled(true);
}


void MainWindow::on_buttonModelStart_clicked()
{
    if (myProject.modelPause)
    {
        myProject.modelPause = false;
        ui->buttonModelPause->setEnabled(true);
        ui->buttonModelStart->setDisabled(true);
        ui->buttonModelStop->setEnabled(true);

        QDateTime newFirstTime = QDateTime(myProject.getCurrentDate(), QTime(myProject.getCurrentHour(), 0, 0), Qt::UTC);
        newFirstTime = newFirstTime.addSecs(3600);

        myProject.runModels(newFirstTime, myProject.modelLastTime);
    }
}


//------------------- MENU SOLAR RADIATION MODEL ----------------

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

    myProject.processes.initialize();
    myProject.processes.computeRadiation = true;

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
        myProject.logInfoGUI("Initialize snow model before.");
        return;
    }

    QDateTime firstTime, lastTime;
    if (! selectDates (firstTime, lastTime))
        return;

    myProject.processes.initialize();
    myProject.processes.setComputeSnow(true);

    startModels(firstTime, lastTime);
}


void MainWindow::on_actionSnow_compute_next_hour_triggered()
{
    if (! myProject.snowMaps.isInitialized)
    {
        if (! myProject.initializeSnowModel())
            return;
    }

    QDateTime currentTime;
    if (myProject.getCurrentHour() == 23)
    {
        currentTime.setDate(myProject.getCurrentDate().addDays(1));
        currentTime.setTime(QTime(0, 0, 0, 0));
    }
    else
    {
        currentTime = myProject.getCurrentTime().addSecs(HOUR_SECONDS);
    }

    myProject.processes.initialize();
    myProject.processes.setComputeSnow(true);

    startModels(currentTime, currentTime);
}


void MainWindow::on_actionSnow_settings_triggered()
{
    DialogSnowSettings dialogSnowSetting;
    dialogSnowSetting.setRainfallThresholdValue(myProject.snowModel.snowParameters.tempMaxWithSnow);
    dialogSnowSetting.setSnowThresholdValue(myProject.snowModel.snowParameters.tempMinWithRain);
    dialogSnowSetting.setWaterHoldingValue(myProject.snowModel.snowParameters.snowWaterHoldingCapacity);
    dialogSnowSetting.setSurfaceThickValue(myProject.snowModel.snowParameters.skinThickness);
    dialogSnowSetting.setVegetationHeightValue(myProject.snowModel.snowParameters.snowVegetationHeight);
    dialogSnowSetting.setSoilAlbedoValue(myProject.snowModel.snowParameters.soilAlbedo);
    dialogSnowSetting.setSnowDampingDepthValue(myProject.snowModel.snowParameters.snowSurfaceDampingDepth);


    dialogSnowSetting.exec();
    if (dialogSnowSetting.result() != QDialog::Accepted)
    {
        return;
    }
    else
    {   
        myProject.snowModel.snowParameters.tempMinWithRain = dialogSnowSetting.getSnowThresholdValue();
        myProject.snowModel.snowParameters.tempMaxWithSnow = dialogSnowSetting.getRainfallThresholdValue();
        myProject.snowModel.snowParameters.snowWaterHoldingCapacity = dialogSnowSetting.getWaterHoldingValue();
        myProject.snowModel.snowParameters.skinThickness = dialogSnowSetting.getSurfaceThickValue();
        myProject.snowModel.snowParameters.snowVegetationHeight = dialogSnowSetting.getVegetationHeightValue();
        myProject.snowModel.snowParameters.soilAlbedo = dialogSnowSetting.getSoilAlbedoValue();
        myProject.snowModel.snowParameters.snowSurfaceDampingDepth = dialogSnowSetting.getSnowDampingDepthValue();

        if (!myProject.writeCriteria3DParameters())
        {
            myProject.logError("Error writing snow parameters");
        }
    }
    return;
}


//--------------------- MENU WATER FLUXES  -----------------------

void MainWindow::on_actionWaterFluxes_settings_triggered()
{
    DialogWaterFluxesSettings dialogWaterFluxes;
    dialogWaterFluxes.setInitialWaterPotential(myProject.waterFluxesParameters.initialWaterPotential);
    dialogWaterFluxes.setImposedComputationDepth(myProject.waterFluxesParameters.imposedComputationDepth);

    dialogWaterFluxes.snowProcess->setChecked(myProject.processes.computeSnow);
    dialogWaterFluxes.cropProcess->setChecked(myProject.processes.computeCrop);
    dialogWaterFluxes.waterFluxesProcess->setChecked(myProject.processes.computeWater);

    if (myProject.waterFluxesParameters.computeOnlySurface)
        dialogWaterFluxes.onlySurface->setChecked(true);
    else if (myProject.waterFluxesParameters.computeAllSoilDepth)
        dialogWaterFluxes.allSoilDepth->setChecked(true);
    else
        dialogWaterFluxes.imposedDepth->setChecked(true);

    dialogWaterFluxes.useWaterRetentionFitting->setChecked(myProject.fittingOptions.useWaterRetentionData);

    dialogWaterFluxes.exec();
    if (dialogWaterFluxes.result() != QDialog::Accepted)
    {
        return;
    }
    else
    {
        myProject.waterFluxesParameters.initialWaterPotential = dialogWaterFluxes.getInitialWaterPotential();
        myProject.waterFluxesParameters.imposedComputationDepth = dialogWaterFluxes.getImposedComputationDepth();
        myProject.waterFluxesParameters.computeOnlySurface = dialogWaterFluxes.onlySurface->isChecked();
        myProject.waterFluxesParameters.computeAllSoilDepth = dialogWaterFluxes.allSoilDepth->isChecked();
        myProject.fittingOptions.useWaterRetentionData = dialogWaterFluxes.useWaterRetentionFitting->isChecked();

        myProject.processes.setComputeSnow(dialogWaterFluxes.snowProcess->isChecked());
        myProject.processes.setComputeCrop(dialogWaterFluxes.cropProcess->isChecked());
        myProject.processes.setComputeWater(dialogWaterFluxes.waterFluxesProcess->isChecked());

        /*if (! myProject.writeCriteria3DParameters())
        {
            myProject.logError("Error writing soil fluxes parameters");
        }*/
    }

    // layer thickness
    // boundary (lateral free drainage, bottom free drainage)
    // lateral conductivity ratio
}


void MainWindow::initializeCriteria3DInterface()
{
    if (myProject.isCriteria3DInitialized)
    {
        ui->groupBoxModel->setEnabled(true);

        if (myProject.nrLayers <= 1)
        {
            ui->layerNrEdit->setMinimum(0);
            ui->layerNrEdit->setMaximum(0);
            ui->layerNrEdit->setValue(0);
            ui->layerDepthEdit->setText("No soil");
        }
        else
        {
            ui->layerNrEdit->setMinimum(1);
            ui->layerNrEdit->setMaximum(myProject.nrLayers - 1);
            ui->layerNrEdit->setValue(1);

            QString depthStr = QString::number(myProject.layerDepth[1],'f',2);
            ui->layerDepthEdit->setText(depthStr + " m");
        }

        myProject.currentSeconds = 0;
        updateModelTime();
    }
}


void MainWindow::on_actionCriteria3D_Initialize_triggered()
{
    myProject.initializeCrop();
    if (myProject.processes.computeCrop)
    {
        if (! myProject.initializeCropWithClimateData())
        {
            myProject.logError();
            return;
        }
    }

    if (myProject.processes.computeWater)
        if (! myProject.initializeCriteria3DModel())
        {
            myProject.logError();
            return;
        }

    if (myProject.processes.computeCrop || myProject.processes.computeWater)
    {
        initializeCriteria3DInterface();
        myProject.isCriteria3DInitialized = true;
        myProject.logInfoGUI("Criteria3D model initialized");
    }
}


void MainWindow::on_actionCriteria3D_compute_next_hour_triggered()
{
    if (! myProject.isCriteria3DInitialized)
    {
        myProject.logError("Initialize 3D water fluxes before");
        return;
    }

    QDateTime currentTime;
    if (myProject.getCurrentHour() == 23)
    {
        currentTime.setDate(myProject.getCurrentDate().addDays(1));
        currentTime.setTime(QTime(0, 0, 0, 0));
    }
    else
    {
        currentTime = myProject.getCurrentTime().addSecs(HOUR_SECONDS);
    }

    startModels(currentTime, currentTime);
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

    startModels(firstTime, lastTime);
}


void MainWindow::showCriteria3DVariable(criteria3DVariable var, int layerIndex, bool isFixedRange, float minimum, float maximum)
{
    if (! myProject.isCriteria3DInitialized)
    {
        myProject.logError("Initialize water fluxes before.");
        return;
    }

    // compute map
    bool isOk;
    if (var == minimumFactorOfSafety)
    {
        isOk = myProject.computeMinimumFoS(myProject.criteria3DMap);
    }
    else
    {
        isOk = myProject.getCriteria3DMap(myProject.criteria3DMap, var, layerIndex);
    }

    if (! isOk)
    {
        myProject.logError();
        return;
    }

    current3DVariable = var;
    current3DlayerIndex = layerIndex;

    if (current3DVariable == volumetricWaterContent)
    {
        if (layerIndex == 0)
        {
            // SURFACE
            setSurfaceWaterScale(myProject.criteria3DMap.colorScale);
            ui->labelOutputRaster->setText("Surface water content [mm]");
        }
        else
        {
            // SUB-SURFACE
            setTemperatureScale(myProject.criteria3DMap.colorScale);
            reverseColorScale(myProject.criteria3DMap.colorScale);
            ui->labelOutputRaster->setText("Volumetric water content [m3 m-3]");
        }
    }
    else if (current3DVariable == degreeOfSaturation)
    {
        setTemperatureScale(myProject.criteria3DMap.colorScale);
        reverseColorScale(myProject.criteria3DMap.colorScale);
        ui->labelOutputRaster->setText("Degree of saturation [-]");
    }
    else if (current3DVariable == factorOfSafety || current3DVariable == minimumFactorOfSafety)
    {
        setSlopeStabilityScale(myProject.criteria3DMap.colorScale);
        ui->labelOutputRaster->setText("Factor of safety [-]");
    }

    // set range
    if (isFixedRange)
    {
        myProject.criteria3DMap.colorScale->setRange(minimum, maximum);
        myProject.criteria3DMap.colorScale->setRangeBlocked(true);
    }
    else
    {
        myProject.criteria3DMap.colorScale->setRangeBlocked(false);
    }

    setCurrentRasterOutput(&(myProject.criteria3DMap));
}


//------------------- STATES ----------------------

void MainWindow::on_flagSave_state_daily_step_toggled(bool isChecked)
{
    myProject.setSaveDailyState(isChecked);
}


void MainWindow::on_flagSave_state_endRun_triggered(bool isChecked)
{
     myProject.setSaveEndOfRunState(isChecked);
}


void MainWindow::on_actionSave_state_triggered()
{
    if (myProject.isProjectLoaded)
    {
        if (myProject.saveModelsState())
        {
            myProject.logInfoGUI("State model successfully saved: " + myProject.getCurrentDate().toString()
                                 + " H:" + QString::number(myProject.getCurrentHour()));
        }
    }
    else
    {
        myProject.logError(ERROR_STR_MISSING_PROJECT);
    }
}


void MainWindow::on_actionLoad_external_state_triggered()
{
    if (! myProject.isProjectLoaded)
    {
        myProject.logError(ERROR_STR_MISSING_PROJECT);
        return;
    }

    QString stateDirectory = QFileDialog::getExistingDirectory(this, tr("Open Directory"), "",
                                                               QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (myProject.loadModelState(stateDirectory))
    {
        updateDateTime();
        loadMeteoPointsDataSingleDay(myProject.getCurrentDate(), true);
        redrawMeteoPoints(currentPointsVisualization, true);
    }
    else
    {
        myProject.logError();
    }
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

    QString stateDirectory = myProject.getProjectPath() + PATH_STATES + dialogLoadState.getSelectedState();
    if (myProject.loadModelState(stateDirectory))
    {
        updateDateTime();
        loadMeteoPointsDataSingleDay(myProject.getCurrentDate(), true);
        redrawMeteoPoints(currentPointsVisualization, true);
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

    loadMeteoPointsDataSingleDay(myProject.getCurrentDate(), true);
    redrawMeteoPoints(currentPointsVisualization, true);
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

    loadMeteoPointsDataSingleDay(myProject.getCurrentDate(), true);
    redrawMeteoPoints(currentPointsVisualization, true);
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

    if (!myProject.writeOutputPointList(myProject.outputPointsFileName))
    {
        return;
    }

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

    if (!myProject.writeOutputPointList(myProject.outputPointsFileName))
    {
        return;
    }

    myProject.clearSelectedOutputPoints();
    redrawOutputPoints();
}

void MainWindow::on_actionOutputPoints_activate_all_triggered()
{
    for (unsigned int i = 0; i < myProject.outputPoints.size(); i++)
    {
        myProject.outputPoints[i].active = true;
    }

    if (!myProject.writeOutputPointList(myProject.outputPointsFileName))
    {
        return;
    }

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

    if (!myProject.writeOutputPointList(myProject.outputPointsFileName))
    {
        return;
    }

    myProject.clearSelectedOutputPoints();
    redrawOutputPoints();
}

void MainWindow::on_actionOutputPoints_delete_selected_triggered()
{
    unsigned int n = 0;
    for (unsigned int i = 0; i < myProject.outputPoints.size(); i++)
    {
        if (myProject.outputPoints[i].selected)
        {
            n = n + 1;
        }
    }

    if (n == 0)
    {
        myProject.logError("No meteo point selected.");
        return;
    }

    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, "Are you sure?" ,
                                  QString::number(n) + " selected points will be deleted",
                                  QMessageBox::Yes|QMessageBox::No);

    if (reply == QMessageBox::Yes)
    {
        for (int i = 0; i < int(myProject.outputPoints.size()); i++)
        {
            if (myProject.outputPoints[unsigned(i)].selected)
            {
                myProject.outputPoints.erase(myProject.outputPoints.begin()+i);
                mapView->scene()->removeObject(outputPointList[i]);
                delete outputPointList[i];
                outputPointList.removeAt(i);
                i = i-1;
            }
        }

        if (!myProject.writeOutputPointList(myProject.outputPointsFileName))
        {
            return;
        }

        myProject.clearSelectedOutputPoints();
        redrawOutputPoints();
    }
    return;
}


void MainWindow::on_actionOutputPoints_newFile_triggered()
{
    if (!myProject.outputPoints.empty())
    {
        QMessageBox::StandardButton closeBox;
        closeBox = QMessageBox::question(this, "close output points" ,
                                      "existing output points will be closed",
                                      QMessageBox::Yes|QMessageBox::No);
        if (closeBox == QMessageBox::Yes)
        {
            resetOutputPointMarkers();

        }
        else
        {
            return;
        }
    }

    QString csvName = QFileDialog::getSaveFileName(this, tr("Save as"), myProject.getProjectPath() + PATH_OUTPUT, tr("csv files (*.csv)"));
    if (csvName == "")
    {
        return;
    }

    QFile csvFile(csvName);
    if (csvFile.exists())
    {
        if (!csvFile.remove())
        {
            myProject.logError("Failed to remove existing csv file.");
            return;
        }
    }

    if (csvFile.open(QIODevice::ReadWrite))
    {
        QTextStream outStream(&csvFile);
        outStream << "id, latitude, longitude, height, active" << "\n";
        csvFile.close();
    }
    else
    {
        myProject.logError("Failed to open csv file.");
        return;
    }

    myProject.loadOutputPointList(csvName);
}


void MainWindow::on_actionOutputDB_new_triggered()
{
    QString dbName = QFileDialog::getSaveFileName(this, tr("Save as"), myProject.getProjectPath() + PATH_OUTPUT, tr("DB files (*.db)"));
    if (dbName == "") return;

    myProject.newOutputPointsDB(dbName);
}

void MainWindow::on_actionOutputDB_open_triggered()
{
    QString dbName = QFileDialog::getOpenFileName(this, tr("Open output db"), myProject.getProjectPath() + PATH_OUTPUT, tr("DB files (*.db)"));

    if (dbName == "") return;

    myProject.loadOutputPointsDB(dbName);
}


void MainWindow::on_flagOutputPoints_save_output_toggled(bool isChecked)
{
    if (isChecked && myProject.outputPointsDbHandler == nullptr)
    {
        myProject.logError("Open or create a new output DB before.");
        isChecked = false;
    }
    myProject.setSaveOutputPoints(isChecked);
    ui->flagOutputPoints_save_output->setChecked(isChecked);
}


void MainWindow::on_flagCompute_only_points_toggled(bool isChecked)
{
    myProject.setComputeOnlyPoints(isChecked);
}


void MainWindow::on_actionLoad_OutputPoints_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open output point list"), myProject.getProjectPath() + PATH_OUTPUT, tr("csv files (*.csv)"));
    if (fileName == "") return;

    if (! myProject.loadOutputPointList(fileName))
    {
        resetOutputPointMarkers();
        return;
    }

    addOutputPointsGUI();
}


void MainWindow::on_actionOutputPoints_add_triggered()
{
    if (myProject.outputPointsFileName.isEmpty())
    {
        myProject.logError("Load an output point list before");
        return;
    }

    QList<QString> idPoints;
    for (unsigned int i = 0; i < myProject.outputPoints.size(); i++)
    {
        idPoints.append(QString::fromStdString(myProject.outputPoints[i].id));
    }

    gis::Crit3DRasterGrid myGrid;
    DialogNewPoint newPointDialog(idPoints, myProject.gisSettings, &(myProject.DEM));
    if (newPointDialog.result() == QDialog::Accepted)
    {
        gis::Crit3DOutputPoint newPoint;
        newPoint.initialize(newPointDialog.getId().toStdString(), true, newPointDialog.getLat(), newPointDialog.getLon(), newPointDialog.getHeight(), myProject.gisSettings.utmZone);
        myProject.outputPoints.push_back(newPoint);
        if (!myProject.writeOutputPointList(myProject.outputPointsFileName))
        {
            return;
        }
        addOutputPointsGUI();
    }
}


void MainWindow::on_flagView_values_toggled(bool isChecked)
{
    for (int i = 0; i < meteoPointList.size(); i++)
    {
        meteoPointList[i]->showText(isChecked);
    }
}


void MainWindow::on_actionTopographicDistanceMapWrite_triggered()
{
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, "Create topographic distance maps", "Only for stations with data?",
            QMessageBox::Yes|QMessageBox::No);

    bool onlyWithData = (reply == QMessageBox::Yes);

    myProject.writeTopographicDistanceMaps(onlyWithData, true);
}


void MainWindow::on_actionTopographicDistanceMapLoad_triggered()
{
    QMessageBox::StandardButton reply;
    reply = QMessageBox::question(this, "Load topographic distance maps", "Only for stations with data?",
            QMessageBox::Yes|QMessageBox::No);

    bool onlyWithData = (reply == QMessageBox::Yes);

    myProject.loadTopographicDistanceMaps(onlyWithData, true);
}


void MainWindow::on_viewer3DClosed()
{
    myProject.clearGeometry();
    viewer3D = nullptr;
}


void MainWindow::on_slopeChanged()
{
    myProject.openGlGeometry->setArtifactSlope(int(viewer3D->getSlope()));
    myProject.update3DColors();
    viewer3D->glWidget->update();
}


void MainWindow::on_actionShow_3D_viewer_triggered()
{
    if (viewer3D != nullptr)
    {
        if (! viewer3D->isVisible())
            viewer3D->setVisible(true);
        return;
    }

    if (! myProject.initializeGeometry())
    {
        myProject.logError();
        return;
    }

    viewer3D = new Viewer3D(myProject.openGlGeometry);
    viewer3D->show();

    connect (viewer3D, SIGNAL(destroyed()), this, SLOT(on_viewer3DClosed()));
    connect (viewer3D, SIGNAL(slopeChanged()), this, SLOT(on_slopeChanged()));
}


void MainWindow::on_actionLoad_land_use_map_triggered()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open land use map"), "",
                                                    tr("ESRI float (*.flt);; ENVI image (*.img)"));
    if (fileName == "") return;

    if (myProject.loadLandUseMap(fileName))
    {
        showLandUseMap();

        if (! myProject.DEM.isLoaded)
        {
            // resize map
            double size = double(this->rasterOutput->getRasterMaxSize());
            size = log2(1000 / size);
            mapView->setZoomLevel(quint8(size));

            // center map
            gis::Crit3DGeoPoint center = myProject.landUseMap.getCenterLatLon(myProject.gisSettings);
            mapView->centerOn(center.longitude, center.latitude);

            showLandUseMap();
        }
    }
}


void MainWindow::on_actionView_LandUseMap_triggered()
{
    showLandUseMap();
}

void MainWindow::on_actionHide_LandUseMap_triggered()
{
    if (ui->labelOutputRaster->text() == "Land use")
    {
        setOutputRasterVisible(false);
    }
}


void MainWindow::on_actionHide_Geomap_triggered()
{
    setOutputRasterVisible(false);
}


//------------------- MENU VIEW SOIL FLUXES OUTPUT ------------------

void MainWindow::on_actionView_surfaceWaterContent_automatic_range_triggered()
{
    showCriteria3DVariable(volumetricWaterContent, 0, false, NODATA, NODATA);
}


void MainWindow::on_actionView_surfaceWaterContent_fixed_range_triggered()
{
    // choose minimum
    float minimum = 0;
    QString valueStr = editValue("Choose minimum value [mm]", QString::number(minimum));
    if (valueStr == "") return;
    minimum = valueStr.toFloat();

    // choose maximum
    float maximum = 100;
    valueStr = editValue("Choose maximum value [mm]", QString::number(maximum));
    if (valueStr == "") return;
    maximum = valueStr.toFloat();

    showCriteria3DVariable(volumetricWaterContent, 0, true, minimum, maximum);
}


void MainWindow::on_actionView_SoilMoisture_triggered()
{
    int layerIndex = std::max(1, ui->layerNrEdit->value());
    showCriteria3DVariable(volumetricWaterContent, layerIndex, false, NODATA, NODATA);
}


void MainWindow::on_actionView_degreeOfSaturation_automatic_range_triggered()
{
    int layerIndex = ui->layerNrEdit->value();
    showCriteria3DVariable(degreeOfSaturation, layerIndex, false, NODATA, NODATA);
}


void MainWindow::on_actionView_degreeOfSaturation_fixed_range_triggered()
{
    int layerIndex = ui->layerNrEdit->value();
    showCriteria3DVariable(degreeOfSaturation, layerIndex, true, 0.0, 1.0);
}


void MainWindow::on_actionView_factor_of_safety_triggered()
{
    int layerIndex = std::max(1, ui->layerNrEdit->value());
    showCriteria3DVariable(factorOfSafety, layerIndex, true, 0, 10);
}


void MainWindow::on_actionView_factor_of_safety_minimum_triggered()
{
    showCriteria3DVariable(minimumFactorOfSafety, NODATA, true, 0, 10);
}


void MainWindow::on_actionCriteria3D_update_subHourly_triggered(bool isChecked)
{
    myProject.showEachTimeStep = isChecked;
}


void MainWindow::on_flag_double_slope_triggered(bool isChecked)
{
    myProject.useDoubleSlope = isChecked;
}


//------------------- OTHER FUNCTIONS ---------------------

void MainWindow::on_layerNrEdit_valueChanged(int layerIndex)
{
    if (myProject.nrLayers <= 1)
    {
        ui->layerNrEdit->setValue(1);
        ui->layerDepthEdit->setText("No soil");
        return;
    }

    if (unsigned(layerIndex) >= myProject.nrLayers)
    {
        layerIndex = myProject.nrLayers - 1;
        ui->layerNrEdit->setValue(layerIndex);
    }

    QString depthStr = QString::number(myProject.layerDepth[layerIndex],'f',2);
    ui->layerDepthEdit->setText(depthStr + " m");

    if (view3DVariable && current3DlayerIndex != 0)
    {
        if (myProject.criteria3DMap.colorScale->isRangeBlocked())
        {
            showCriteria3DVariable(current3DVariable, layerIndex, true,
                                   myProject.criteria3DMap.colorScale->minimum(),
                                   myProject.criteria3DMap.colorScale->maximum());
        }
        else
        {
            showCriteria3DVariable(current3DVariable, layerIndex, false, NODATA, NODATA);
        }
    }
}


void MainWindow::on_actionCriteria3D_load_state_triggered()
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

    QString stateDirectory = myProject.getProjectPath() + PATH_STATES + dialogLoadState.getSelectedState();
    if (! myProject.loadModelState(stateDirectory))
    {
        myProject.logError();
        return;
    }

    updateDateTime();
    initializeCriteria3DInterface();
    loadMeteoPointsDataSingleDay(myProject.getCurrentDate(), true);
    redrawMeteoPoints(currentPointsVisualization, true);
}


void MainWindow::on_actionCriteria3D_load_external_state_triggered()
{
    if (! myProject.isProjectLoaded)
    {
        myProject.logError(ERROR_STR_MISSING_PROJECT);
        return;
    }

    QString stateDirectory = QFileDialog::getExistingDirectory(this, tr("Open Directory"), "",
                                                               QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (! myProject.loadModelState(stateDirectory))
    {
        myProject.logError();
        return;
    }

    updateDateTime();
    initializeCriteria3DInterface();
    loadMeteoPointsDataSingleDay(myProject.getCurrentDate(), true);
    redrawMeteoPoints(currentPointsVisualization, true);
}


void MainWindow::on_actionCriteria3D_save_state_triggered()
{
    if (myProject.isProjectLoaded)
    {
        if (myProject.saveModelsState())
        {
            myProject.logInfoGUI("State model successfully saved: " + myProject.getCurrentDate().toString()
                                 + " H:" + QString::number(myProject.getCurrentHour()));
        }
    }
    else
    {
        myProject.logError(ERROR_STR_MISSING_PROJECT);
    }
}


