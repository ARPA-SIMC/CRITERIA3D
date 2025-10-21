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
#include "dialogModelProcesses.h"
#include "utilities.h"
#include "formText.h"
#include "soilFluxes3D.h"

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

    myProject.setSaveYearlyState(false);
    myProject.setSaveMonthlyState(false);

    myProject.setSaveOutputPoints(false);
    myProject.setComputeOnlyPoints(false);
    ui->flagOutputPoints_save_output->setChecked(myProject.isSaveOutputPoints());
    ui->flagCompute_only_points->setChecked(myProject.getComputeOnlyPoints());
    ui->actionCriteria3D_parallel_computing->setChecked(myProject.isParallelComputing());
    ui->actionCriteria3D_update_subHourly->setChecked(myProject.showEachTimeStep);

    this->setMouseTracking(true);
    this->setTitle();

    connect(&myProject, &Crit3DProject::updateOutputSignal, this, &MainWindow::updateOutputMap);
}


void MainWindow::resizeEvent(QResizeEvent * event)
{
    Q_UNUSED(event)

    const int INFOHEIGHT = 42;

    int stepY = (this->height() - INFOHEIGHT) / 40;
    int x1 = this->width() - TOOLSWIDTH - MAPBORDER;
    int dy = ui->groupBoxModel->height() + ui->groupBoxMeteoPoints->height() + ui->groupBoxDEM->height() + ui->groupBoxVariableMap->height() + stepY*3;
    int y1 = (this->height() - INFOHEIGHT - dy) / 2;

    ui->widgetMap->setGeometry(0, 0, x1, this->height() - INFOHEIGHT);
    mapView->resize(ui->widgetMap->size());

    ui->groupBoxModel->move(x1, y1);
    ui->groupBoxModel->resize(TOOLSWIDTH, ui->groupBoxModel->height());
    y1 += ui->groupBoxModel->height() + stepY;

    ui->groupBoxDEM->move(x1, y1);
    ui->groupBoxDEM->resize(TOOLSWIDTH, ui->groupBoxDEM->height());
    y1 += ui->groupBoxDEM->height() + stepY;

    ui->groupBoxMeteoPoints->move(x1, y1);
    ui->groupBoxMeteoPoints->resize(TOOLSWIDTH, ui->groupBoxMeteoPoints->height());
    y1 += ui->groupBoxMeteoPoints->height() + stepY;

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
        myProject.computeCriteria3DMap(myProject.criteria3DMap, current3DVariable, current3DlayerIndex);
    }

    emit rasterOutput->redrawRequested();
    outputRasterColorLegend->update();

    refreshViewer3D();

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

    QString infoStr = "Lat:" + QString::number(pos.latitude(), 'g', 7) + " Lon:" + QString::number(pos.longitude(), 'g', 7);

    if (rasterOutput->visible())
    {
        float value = rasterOutput->getValue(pos);
        if (! isEqual(value, NODATA))
            infoStr += "  Value:" + QString::number(double(value));
    }
    else if (rasterDEM->visible())
    {
        float value = rasterDEM->getValue(pos);
        if (! isEqual(value, NODATA))
            infoStr += "  DEM value:" + QString::number(double(value));
    }

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

    if (myProject.DEM.isLoaded)
    {
        if (isInsideDEM(mapPos))
        {
            contextMenu.addSeparator();
            contextMenu.addAction("Extract the basin from this point");
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

        if (selection->text().contains("Extract the basin"))
        {
            double x, y;
            Position geoPos = mapView->mapToScene(mapPos);
            gis::latLonToUtmForceZone(myProject.gisSettings.utmZone, geoPos.latitude(), geoPos.longitude(), &x, &y);

            // extract basin
            gis::Crit3DRasterGrid basinRaster;
            if (! gis::extractBasin(myProject.DEM, basinRaster, x, y))
            {
                myProject.logWarning("Wrong closure point.");
                return false;
            }

            // choose fileName
            QString completeFileName = QFileDialog::getSaveFileName(this, tr("Save basin raster"), "", tr("ESRI float (*.flt)"));
            if (completeFileName.isEmpty())
                return false;
            std::string fileName = completeFileName.left(completeFileName.size() - 4).toStdString();

            // save map
            std::string errorStr;
            if (! gis::writeEsriGrid(fileName, &basinRaster, errorStr))
            {
                myProject.logError(QString::fromStdString(errorStr));
                return false;
            }
        }

        if (selection->text().contains("View land use"))
        {
            Position geoPos = mapView->mapToScene(mapPos);
            int id = myProject.getLandUnitIdGeo(geoPos.latitude(), geoPos.longitude());
            if (id != NODATA)
            {
                int index = myProject.getLandUnitListIndex(id);
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
                int index = myProject.getLandUnitListIndex(id);
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
        point->setCallerSoftware(CRITERIA3D_caller);

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
        myProject.showMeteoWidgetGrid(id, dataset, isAppend);
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
        myProject.showMeteoWidgetGrid(id, dataset, isAppend);
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
    QString currentTileMap = myProject.getCurrentTileMap().toUpper();
    if (! currentTileMap.isEmpty())
    {
        if (currentTileMap == "ESRI")
        {
            this->setTileMapSource(WebTileSource::ESRI_WorldImagery);
        }
        else if (currentTileMap == "TERRAIN")
        {
            this->setTileMapSource(WebTileSource::GOOGLE_Terrain);
        }
        else if (currentTileMap == "GOOGLE")
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


void MainWindow::setTitle()
{
    QString title = "CRITERIA3D  " + QString(CRITERIA3D_VERSION);
    QString projectName = myProject.getProjectName();
    if (! projectName.isEmpty())
    {
        title += " - " + projectName;
    }

    this->setWindowTitle(title);
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

    this->addOutputPointsGUI();

    this->setTitle();
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
    if (! myProject.DEM.isLoaded)
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
    int hour = myProject.getCurrentHour() - 1;
    if (hour == -1)
    {
        date = date.addDays(-1);
        hour = 23;
    }
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


void MainWindow::on_actionLoad_DEM__triggered()
{
    QString demPath = myProject.getDefaultPath() + PATH_DEM;
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open Digital Elevation Model"), demPath,
                                    tr("ESRI float (*.flt);; ENVI image (*.img)"));
    if (fileName == "") return;

    clearRaster_GUI();

    if (! myProject.loadDEM(fileName)) return;

    renderDEM();
}


void MainWindow::on_actionExtract_sub_basin_triggered()
{
    if (! myProject.DEM.isLoaded)
    {
        myProject.logWarning(ERROR_STR_MISSING_DEM);
        return;
    }

    //myProject.logInfoGUI("Select the basin closing point.");
    myProject.logWarning("This feature will be available soon.");

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
        myProject.logError("*** Error opening the project! ***\n" + myProject.errorString);
        myProject.loadCriteria3DProject(myProject.getApplicationPath() + "default.ini");
    }

    ui->flagOutputPoints_save_output->setChecked(myProject.isSaveOutputPoints());
    ui->flagCompute_only_points->setChecked(myProject.getComputeOnlyPoints());

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
                      myProject.qualityInterpolationSettings, myProject.meteoSettings,
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

void MainWindow::setInputRasterVisible(bool isVisible)
{
    inputRasterColorLegend->setVisible(isVisible);
    ui->labelInputRaster->setVisible(isVisible);
    rasterDEM->setVisible(isVisible);
}

void MainWindow::setOutputRasterVisible(bool isVisible)
{
    outputRasterColorLegend->setVisible(isVisible);
    ui->labelOutputRaster->setVisible(isVisible);
    rasterOutput->setVisible(isVisible);
}

void MainWindow::setCurrentRasterInput(gis::Crit3DRasterGrid *myRaster)
{
    setInputRasterVisible(true);

    rasterDEM->initialize(myRaster, myProject.gisSettings);
    inputRasterColorLegend->colorScale = myRaster->colorScale;

    inputRasterColorLegend->repaint();
    emit rasterDEM->redrawRequested();
}


void MainWindow::refreshViewer3D()
{
    if (viewer3D != nullptr)
    {
        if (rasterOutput->visible())
        {
            myProject.update3DColors(rasterOutput->getRasterPointer());
        }
        else
        {
            myProject.update3DColors();
        }

        viewer3D->glWidget->update();
    }
}


void MainWindow::setCurrentRasterOutput(gis::Crit3DRasterGrid *rasterPointer)
{
    setOutputRasterVisible(true);

    rasterOutput->initialize(rasterPointer, myProject.gisSettings);
    outputRasterColorLegend->colorScale = rasterPointer->colorScale;

    emit rasterOutput->redrawRequested();
    outputRasterColorLegend->update();
    rasterOutput->updateCenter();

    refreshViewer3D();

    view3DVariable = (rasterPointer == &(myProject.criteria3DMap));
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
        myProject.logWarning(ERROR_STR_MISSING_DEM);
        return;
    }
}

void MainWindow::on_actionView_Aspect_triggered()
{
    if (myProject.DEM.isLoaded)
    {
        myProject.radiationMaps->aspectMap->colorScale->setMinimum(0);
        myProject.radiationMaps->aspectMap->colorScale->setMaximum(360);
        myProject.radiationMaps->aspectMap->colorScale->setFixedRange(true);
        setCircolarScale(myProject.radiationMaps->aspectMap->colorScale);
        setCurrentRasterOutput(myProject.radiationMaps->aspectMap);
        ui->labelOutputRaster->setText("Aspect °");
    }
    else
    {
        myProject.logWarning(ERROR_STR_MISSING_DEM);
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
        myProject.logError(ERROR_STR_INITIALIZE_3D);
        return;
    }
}

void MainWindow::on_actionView_SoilMap_triggered()
{
    if (myProject.soilMap.isLoaded)
    {
        setColorScale(noMeteoVar, myProject.soilMap.colorScale);
        setCurrentRasterOutput(&(myProject.soilMap));
        ui->labelOutputRaster->setText("Soil");
    }
    else
    {
        myProject.logError("Load a soil map before.");
    }
}


void MainWindow::on_actionHide_Soil_map_triggered()
{
    if (ui->labelOutputRaster->text() == "Soil")
    {
        setOutputRasterVisible(false);
        refreshViewer3D();
    }
}

// -------------------- METEO VARIABLES -------------------------

bool MainWindow::checkMapVariable(bool isComputed)
{
    if (! myProject.DEM.isLoaded)
    {
        myProject.logWarning(ERROR_STR_MISSING_DEM);
        return false;
    }

    if (! isComputed)
    {
        myProject.logWarning("Compute meteo variables before.");
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

void MainWindow::on_actionView_Radiation_None_triggered()
{
    setOutputRasterVisible(false);
    refreshViewer3D();
}

void MainWindow::on_actionView_MeteoVariable_None_triggered()
{
    setOutputRasterVisible(false);
    refreshViewer3D();
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

void MainWindow::on_actionView_Crop_degreeDays_triggered()
{
    if (! myProject.isCropInitialized)
    {
        myProject.logWarning("Initialize crop model before.");
        return;
    }

    setOutputMeteoVariable(dailyHeatingDegreeDays, &(myProject.degreeDaysMap));
}


void MainWindow::on_actionView_Crop_LAI_triggered()
{
    if (! myProject.isCropInitialized)
    {
        myProject.logWarning("Initialize crop model before.");
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

bool MainWindow::isInsideDEM(QPoint mapPos)
{
    if (! myProject.DEM.isLoaded)
        return false;

    double x, y;
    Position geoPos = mapView->mapToScene(mapPos);
    gis::latLonToUtmForceZone(myProject.gisSettings.utmZone, geoPos.latitude(), geoPos.longitude(), &x, &y);

    float value = myProject.DEM.getValueFromXY(x, y);
    return (value != myProject.DEM.header->flag);
}


bool MainWindow::isSoil(QPoint mapPos)
{
    if (! myProject.soilMap.isLoaded)
        return false;

    double x, y;
    Position geoPos = mapView->mapToScene(mapPos);
    gis::latLonToUtmForceZone(myProject.gisSettings.utmZone, geoPos.latitude(), geoPos.longitude(), &x, &y);

    int idSoil = myProject.getSoilMapId(x, y);
    return (idSoil != NODATA);
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
        setColorScale(noMeteoVar, myProject.landUseMap.colorScale);
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

        QString imgPath = myProject.getApplicationPath() + "/DOC/img/";
        soilWidget = new Crit3DSoilWidget(imgPath);
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
    QString meteoPointsPath = myProject.getDefaultPath() + PATH_METEOPOINT;
    QString dbName = QFileDialog::getOpenFileName(this, tr("Open meteo points DB"), meteoPointsPath, tr("DB files (*.db)"));
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
    QString soilPath = myProject.getDefaultPath() + PATH_SOIL;
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open soil map"), soilPath, tr("ESRI float (*.flt);; ENVI image (*.img)"));
    if (fileName == "") return;

    if (myProject.loadSoilMap(fileName))
    {
        on_actionView_SoilMap_triggered();
    }
}

void MainWindow::on_actionLoad_soil_data_triggered()
{
    QString soilPath = myProject.getDefaultPath() + PATH_SOIL;
    QString fileName = QFileDialog::getOpenFileName(this, tr("Load soil data"), soilPath, tr("SQLite files (*.db)"));
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

    return myProject.showProxyGraph(NODATA);
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

    initializeGroupBoxModel();
    myProject.startModels(firstTime, lastTime);
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


void MainWindow::initializeGroupBoxModel()
{
    ui->groupBoxModel->setEnabled(true);
    ui->buttonModelStart->setDisabled(true);
    ui->buttonModel1hour->setDisabled(true);
    ui->buttonModelPause->setEnabled(true);
    ui->buttonModelStop->setEnabled(true);
}


void MainWindow::on_buttonModelPause_clicked()
{
    myProject.isModelPaused = true;

    ui->buttonModelPause->setDisabled(true);
    ui->buttonModel1hour->setEnabled(true);
    ui->buttonModelStart->setEnabled(true);
    ui->buttonModelStop->setEnabled(true);

    qApp->processEvents();
}


void MainWindow::on_buttonModelStop_clicked()
{
    myProject.isModelStopped = true;
    myProject.isModelRunning = false;

    ui->buttonModelPause->setDisabled(true);
    ui->buttonModelStart->setDisabled(true);
    ui->buttonModel1hour->setDisabled(true);
    ui->buttonModelStop->setDisabled(true);
}


void MainWindow::on_buttonModel1hour_clicked()
{
    ui->buttonModelPause->setEnabled(true);
    ui->buttonModel1hour->setDisabled(true);
    ui->buttonModelStart->setDisabled(true);
    ui->buttonModelStop->setEnabled(true);

    QDateTime firstTime = QDateTime(myProject.getCurrentDate(), QTime(myProject.getCurrentHour(), 0, 0), Qt::UTC);
    QDateTime lastTime = firstTime.addSecs(3600);
    firstTime = firstTime.addSecs(myProject.currentSeconds);

    myProject.isModelPaused = false;
    bool isRestart = true;
    myProject.runModels(firstTime, lastTime, isRestart);

    on_buttonModelPause_clicked();
}


void MainWindow::on_buttonModelStart_clicked()
{
    if (myProject.isModelPaused)
    {
        ui->buttonModelPause->setEnabled(true);
        ui->buttonModel1hour->setDisabled(true);
        ui->buttonModelStart->setDisabled(true);
        ui->buttonModelStop->setEnabled(true);

        QDateTime newFirstTime = QDateTime(myProject.getCurrentDate(), QTime(myProject.getCurrentHour(), 0, 0), Qt::UTC);
        newFirstTime = newFirstTime.addSecs(myProject.currentSeconds);

        myProject.isModelPaused = false;
        bool isRestart = true;
        myProject.runModels(newFirstTime, myProject.modelLastTime, isRestart);

        // computation finished
        if (myProject.getCurrentTime() == myProject.modelLastTime)
        {
            on_buttonModelStop_clicked();
        }
    }
    else
    {
        myProject.logWarning("Choose the computation period in the 'Run model' menu.");
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

    initializeGroupBoxModel();
    myProject.startModels(firstTime, lastTime);
}


//------------------------------- MENU 3D MODEL  ------------------------------------

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
        return;

    myProject.snowModel.snowParameters.tempMinWithRain = dialogSnowSetting.getSnowThresholdValue();
    myProject.snowModel.snowParameters.tempMaxWithSnow = dialogSnowSetting.getRainfallThresholdValue();
    myProject.snowModel.snowParameters.snowWaterHoldingCapacity = dialogSnowSetting.getWaterHoldingValue();
    myProject.snowModel.snowParameters.skinThickness = dialogSnowSetting.getSurfaceThickValue();
    myProject.snowModel.snowParameters.snowVegetationHeight = dialogSnowSetting.getVegetationHeightValue();
    myProject.snowModel.snowParameters.soilAlbedo = dialogSnowSetting.getSoilAlbedoValue();
    myProject.snowModel.snowParameters.snowSurfaceDampingDepth = dialogSnowSetting.getSnowDampingDepthValue();

    bool isSnow = true;
    bool isWater = false;
    bool isSoilCrack = false;
    if (! myProject.writeCriteria3DParameters(isSnow, isWater, isSoilCrack))
    {
        myProject.logError("Error writing snow parameters");
    }
}


void MainWindow::on_actionCriteria3D_set_processes_triggered()
{
    DialogModelProcesses dialogProcesses;

    dialogProcesses.snowProcess->setChecked(myProject.processes.computeSnow);
    dialogProcesses.cropProcess->setChecked(myProject.processes.computeCrop);
    dialogProcesses.waterFluxesProcess->setChecked(myProject.processes.computeWater);
    dialogProcesses.hydrallProcess->setChecked(myProject.processes.computeHydrall);
    dialogProcesses.rothCProcess->setChecked(myProject.processes.computeRothC);

    dialogProcesses.exec();

    if (dialogProcesses.result() == QDialog::Accepted)
    {
        myProject.processes.setComputeSnow(dialogProcesses.snowProcess->isChecked());
        myProject.processes.setComputeCrop(dialogProcesses.cropProcess->isChecked());
        myProject.processes.setComputeWater(dialogProcesses.waterFluxesProcess->isChecked());

        if (dialogProcesses.hydrallProcess->isChecked() && (! dialogProcesses.cropProcess->isChecked() || ! dialogProcesses.waterFluxesProcess->isChecked()))
            myProject.logWarning("Crop and water processes will be activated in order to compute Hydrall model.");
        myProject.processes.setComputeHydrall(dialogProcesses.hydrallProcess->isChecked());

        /*if (dialogProcesses.rothCProcess->isChecked() && (! dialogProcesses.hydrallProcess->isChecked() || ! dialogProcesses.cropProcess->isChecked()
                                                          || ! dialogProcesses.waterFluxesProcess->isChecked()))
            myProject.logWarning("Hydrall, crop and water processes will be activated in order to compute RothC model.");*/
        myProject.processes.setComputeRothC(dialogProcesses.rothCProcess->isChecked());
    }
}


void MainWindow::on_actionCriteria3D_waterFluxes_settings_triggered()
{
    DialogWaterFluxesSettings dialogWaterFluxes;
    dialogWaterFluxes.setInitialWaterPotential(myProject.waterFluxesParameters.initialWaterPotential);
    dialogWaterFluxes.setInitialDegreeOfSaturation(myProject.waterFluxesParameters.initialDegreeOfSaturation);

    dialogWaterFluxes.useInitialWaterPotential->setChecked(myProject.waterFluxesParameters.isInitialWaterPotential);
    dialogWaterFluxes.useInitialDegreeOfSaturation->setChecked(! myProject.waterFluxesParameters.isInitialWaterPotential);

    dialogWaterFluxes.setImposedComputationDepth(myProject.waterFluxesParameters.imposedComputationDepth);

    dialogWaterFluxes.accuracySlider->setValue(myProject.waterFluxesParameters.modelAccuracy);
    dialogWaterFluxes.setThreadsNumber(myProject.waterFluxesParameters.numberOfThreads);

    if (myProject.waterFluxesParameters.computeOnlySurface)
        dialogWaterFluxes.onlySurface->setChecked(true);
    else if (myProject.waterFluxesParameters.computeAllSoilDepth)
        dialogWaterFluxes.allSoilDepth->setChecked(true);
    else
        dialogWaterFluxes.imposedDepth->setChecked(true);

    dialogWaterFluxes.useWaterRetentionFitting->setChecked(myProject.fittingOptions.useWaterRetentionData);
    dialogWaterFluxes.setConductivityHVRatio(myProject.waterFluxesParameters.conductivityHorizVertRatio);

    dialogWaterFluxes.exec();

    if (dialogWaterFluxes.isUpdateAccuracy())
    {
        myProject.waterFluxesParameters.modelAccuracy = dialogWaterFluxes.accuracySlider->value();
        int nrThread = dialogWaterFluxes.getThreadsNumber();
        nrThread = soilFluxes3D::setThreadsNumber(nrThread);                  // check
        myProject.waterFluxesParameters.numberOfThreads = nrThread;

        if (myProject.isCriteria3DInitialized)
        {
            myProject.setAccuracy();
        }
    }

    if (dialogWaterFluxes.result() == QDialog::Accepted)
    {
        myProject.waterFluxesParameters.initialWaterPotential = dialogWaterFluxes.getInitialWaterPotential();
        myProject.waterFluxesParameters.initialDegreeOfSaturation = dialogWaterFluxes.getInitialDegreeOfSaturation();
        myProject.waterFluxesParameters.isInitialWaterPotential = dialogWaterFluxes.useInitialWaterPotential->isChecked();

        myProject.waterFluxesParameters.conductivityHorizVertRatio = dialogWaterFluxes.getConductivityHVRatio();

        myProject.waterFluxesParameters.imposedComputationDepth = dialogWaterFluxes.getImposedComputationDepth();
        myProject.waterFluxesParameters.computeOnlySurface = dialogWaterFluxes.onlySurface->isChecked();
        myProject.waterFluxesParameters.computeAllSoilDepth = dialogWaterFluxes.allSoilDepth->isChecked();

        myProject.waterFluxesParameters.modelAccuracy = dialogWaterFluxes.accuracySlider->value();

        // check nr of threads
        int threadNumber = dialogWaterFluxes.getThreadsNumber();
        threadNumber = soilFluxes3D::setThreadsNumber(threadNumber);
        myProject.waterFluxesParameters.numberOfThreads = threadNumber;

        if (myProject.isCriteria3DInitialized)
        {
            myProject.setAccuracy();
        }

        myProject.fittingOptions.useWaterRetentionData = dialogWaterFluxes.useWaterRetentionFitting->isChecked();

        bool isWater = true;
        bool isSnow = false;
        bool isSoilCrack = false;
        if (! myProject.writeCriteria3DParameters(isSnow, isWater, isSoilCrack))
        {
            myProject.logError("Error writing soil fluxes parameters");
        }

        // TODO layer thickness
        // TODO soil crack
    }
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

        myProject.currentSeconds = 3600;
        updateModelTime();
    }
}


void MainWindow::on_actionCriteria3D_Initialize_triggered()
{
    if (! (myProject.processes.computeSnow || myProject.processes.computeCrop || myProject.processes.computeWater ||
           myProject.processes.computeHydrall || myProject.processes.computeRothC))
    {
        myProject.logWarning("Set active processes before.");
        return;
    }

    if (myProject.isModelRunning)
    {
        myProject.logWarning("The model is running, stop it before reinitializing.");
        return;
    }

    if (myProject.processes.computeSnow)
    {
        if (! myProject.initializeSnowModel())
        {
            myProject.logError();
            return;
        }
    }
    else
    {
        myProject.snowMaps.clear();
        myProject.isSnowInitialized = false;
    }

    if (myProject.processes.computeCrop)
    {
        if (! myProject.initializeCropWithClimateData())
        {
            myProject.logError();
            return;
        }
    }
    else
    {
        myProject.clearCropMaps();
    }

    if (myProject.processes.computeWater)
    {
        if (! myProject.processes.computeCrop)
        {
            if (! myProject.initializeCropMaps())
            {
                myProject.logError();
                return;
            }
        }

        if (! myProject.initializeCriteria3DModel())
        {
            myProject.logError();
            return;
        }
    }
    else
    {
        myProject.clearWaterBalance3D();
    }

    if (myProject.processes.computeHydrall)
    {
        if (! myProject.processes.computeCrop || ! myProject.processes.computeWater)
        {
            myProject.logError("Active water and crop processes before.");
        }

        myProject.hydrallMaps.initialize(myProject.DEM);

        if (! myProject.initializeHydrall())
        {
            myProject.isHydrallInitialized = false;
            myProject.logError("Couldn't initialize Hydrall model:\n" + myProject.errorString);
            return;
        }
    }
    else
    {
        myProject.clearHydrallMaps();
    }

    if (myProject.processes.computeRothC)
    {
        if (! myProject.processes.computeWater)
        {
            QString defaultPath = myProject.getDefaultPath() + PATH_GEO;
            myProject.rothCModel.BICMapFolderName = QFileDialog::getExistingDirectory(this, tr("Open folder with monthly average BIC files"), defaultPath).toStdString();

            if (myProject.rothCModel.BICMapFolderName.empty())
                return;
        }

        if (! myProject.initializeRothC())
        {
            myProject.isRothCInitialized = false;
            myProject.logError("Couldn't initialize RothC model.");
            return;
        }
    }
    else
    {
        myProject.clearRothCMaps();
    }

    initializeCriteria3DInterface();
    myProject.logInfoGUI("The model is initialized.");
}


void MainWindow::on_actionCriteria3D_compute_next_hour_triggered()
{
    if (! myProject.checkProcesses())
    {
        myProject.logWarning();
        return;
    }

    if (myProject.isModelRunning)
    {
        myProject.logWarning("The model is running, stop it before restart.");
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

    initializeGroupBoxModel();
    myProject.startModels(currentTime, currentTime);
}


void MainWindow::on_actionCriteria3D_run_models_triggered()
{
    if (! myProject.checkProcesses())
    {
        myProject.logWarning();
        return;
    }

    if (myProject.isModelRunning)
    {
        myProject.logWarning("The model is running, stop it before restart.");
        return;
    }

    QDateTime firstTime, lastTime;
    if (! selectDates(firstTime, lastTime))
        return;

    initializeGroupBoxModel();
    myProject.startModels(firstTime, lastTime);
}


void MainWindow::on_actionCriteria3D_Water_content_summary_triggered()
{
    double voxelArea = myProject.DEM.header->cellSize * myProject.DEM.header->cellSize;     // [m2]

    // SURFACE
    double surfaceWaterContent = 0;                                                         // [m3]
    long nrSurfaceVoxels = 0;
    if (! myProject.getTotalSurfaceWaterContent(surfaceWaterContent, nrSurfaceVoxels))
    {
        myProject.logError();
        return;
    }
    double surfaceArea = voxelArea * nrSurfaceVoxels;                                       // [m2]
    double surfaceAvgLevel = surfaceWaterContent / surfaceArea * 1000;                      // [mm]

    // SOIL
    double soilWaterContent = 0;                                                            // [m3]
    long nrSoilVoxels = 0;
    if (! myProject.getTotalSoilWaterContent(soilWaterContent, nrSoilVoxels))
    {
        myProject.logError();
        return;
    }
    double soilArea = voxelArea * nrSoilVoxels;                                             // [m2]
    double soilAvgWC = soilWaterContent / soilArea * 1000;                                  // [mm]

    double totalWaterContent = soilFluxes3D::getTotalWaterContent();                        // [m3]

    QString summaryStr = "WATER CONTENT SUMMARY\n\n";

    summaryStr += "Total water content: " + QString::number(totalWaterContent, 'f', 1) + " [m3]\n";
    summaryStr += "-------------------------------------------\n";
    summaryStr += "Surface area: " + QString::number(surfaceArea / 10000, 'f', 1) + " [hectares]\n";
    summaryStr += "Surface water content: " + QString::number(surfaceWaterContent, 'f', 1) + " [m3]\n";
    summaryStr += "Surface average water level: " + QString::number(surfaceAvgLevel, 'f', 1) + " [mm]\n";
    summaryStr += "-------------------------------------------\n";
    summaryStr += "Soil area: " + QString::number(soilArea / 10000, 'f', 1) + " [hectares]\n";
    summaryStr += "Soil water content: " + QString::number(soilWaterContent, 'f', 1) + " [m3]\n";
    summaryStr += "Soil average water content: " + QString::number(soilAvgWC, 'f', 1) + " [mm]\n";

    myProject.logInfoGUI(summaryStr);
}


void MainWindow::on_actionDEM_summary_triggered()
{
    if (! myProject.DEM.isLoaded)
    {
        myProject.logWarning(ERROR_STR_MISSING_DEM);
        return;
    }

    long nrVoxels = 0;
    double elevationSum = 0;
    float maxElevation = NODATA;
    float minElevation = NODATA;
    bool isFirst = true;
    for (int row = 0; row < myProject.DEM.header->nrRows; row++)
    {
        for (int col = 0; col < myProject.DEM.header->nrCols; col++)
        {
            float elevation = myProject.DEM.value[row][col];
            if (! isEqual(elevation, myProject.DEM.header->flag))
            {
                if (isFirst)
                {
                    maxElevation = elevation;
                    minElevation = elevation;
                    isFirst = false;
                }
                else
                {
                    maxElevation = std::max(maxElevation, elevation);
                    minElevation = std::min(minElevation, elevation);
                }

                elevationSum += elevation;
                nrVoxels++;
            }
        }
    }

    double voxelArea = myProject.DEM.header->cellSize * myProject.DEM.header->cellSize;     // [m2]
    double area = voxelArea * nrVoxels;                                                     // [m2]

    QString summaryStr = "DIGITAL ELEVATION MODEL SUMMARY\n\n";

    summaryStr += "Number of pixels:  " + QString::number(nrVoxels) + "\n";
    summaryStr += "Area:  " + QString::number(area, 'f', 0) + " [m2]\n";
    summaryStr += "Hectares:  " + QString::number(area / 10000., 'f', 2) + " [ha]\n";
    summaryStr += "Area (km2):  " + QString::number(area / 1000000, 'f', 3) + " [km2]\n";
    summaryStr += "Max. elevation:  " + QString::number(maxElevation, 'f', 1) + " [m]\n";
    summaryStr += "Min. elevation:  " + QString::number(minElevation, 'f', 1) + " [m]\n";
    summaryStr += "Avg. elevation:  " + QString::number(elevationSum / nrVoxels, 'f', 1) + " [m]\n";

    myProject.logInfoGUI(summaryStr);
}


void MainWindow::showCriteria3DVariable(criteria3DVariable var, int layerIndex, bool isFixedRange,
                                        bool isHideMinimum, double minimum, double maximum)
{
    if (! myProject.isCriteria3DInitialized)
    {
        myProject.logWarning("Initialize water fluxes before.");
        return;
    }

    // compute map
    if (! myProject.computeCriteria3DMap(myProject.criteria3DMap, var, layerIndex))
    {
        myProject.logWarning();
        return;
    }

    current3DVariable = var;
    current3DlayerIndex = layerIndex;

    myProject.criteria3DMap.colorScale->setFixedRange(false);
    myProject.criteria3DMap.colorScale->setHideMinimum(false);
    myProject.criteria3DMap.colorScale->setTransparent(false);

    if (current3DVariable == volumetricWaterContent)
    {
        if (layerIndex == 0)
        {
            // SURFACE
            setSurfaceWaterScale(myProject.criteria3DMap.colorScale);
            myProject.criteria3DMap.colorScale->setHideMinimum(true);
            myProject.criteria3DMap.colorScale->setTransparent(true);
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
    else if (current3DVariable == waterMatricPotential)
    {
        setTemperatureScale(myProject.criteria3DMap.colorScale);
        reverseColorScale(myProject.criteria3DMap.colorScale);
        ui->labelOutputRaster->setText("Water matric potential [m]");
    }
    else if (current3DVariable == factorOfSafety || current3DVariable == minimumFactorOfSafety)
    {
        setSlopeStabilityScale(myProject.criteria3DMap.colorScale);
        ui->labelOutputRaster->setText("Factor of safety [-]");
    }
    else if (current3DVariable == surfacePond)
    {
        setSurfaceWaterScale(myProject.criteria3DMap.colorScale);
        ui->labelOutputRaster->setText("Surface maximum pond [mm]");
    }

    // range fixed
    if (isFixedRange)
    {
        myProject.criteria3DMap.colorScale->setRange(minimum, maximum);
        myProject.criteria3DMap.colorScale->setFixedRange(true);
    }

    if (isHideMinimum)
    {
        myProject.criteria3DMap.colorScale->setRange(minimum, maximum);
        myProject.criteria3DMap.colorScale->setHideMinimum(true);
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
            bool existData = myProject.meteoPointsDbHandler->existData(myProject.meteoPoints[i], daily) || myProject.meteoPointsDbHandler->existData(myProject.meteoPoints[i], hourly);
            if (! existData)
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
    refreshViewer3D();
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
    setOutputRasterVisible(false);
    refreshViewer3D();
}


void MainWindow::on_actionHide_Geomap_triggered()
{
    setOutputRasterVisible(false);
    refreshViewer3D();
}


//------------------- MENU VIEW SOIL FLUXES OUTPUT ------------------

void MainWindow::on_actionView_SurfacePond_triggered()
{
    showCriteria3DVariable(surfacePond, 0, false, false, NODATA, NODATA);
}


void MainWindow::on_actionView_SurfaceWaterContent_automatic_range_triggered()
{
    showCriteria3DVariable(volumetricWaterContent, 0, false, false, NODATA, NODATA);
}


void MainWindow::on_actionView_SurfaceWaterContent_fixed_range_triggered()
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

    showCriteria3DVariable(volumetricWaterContent, 0, true, false, minimum, maximum);
}


void MainWindow::on_actionView_SoilMoisture_triggered()
{
    int layerIndex = std::max(1, ui->layerNrEdit->value());
    showCriteria3DVariable(volumetricWaterContent, layerIndex, false, false, NODATA, NODATA);
}



void MainWindow::on_actionView_Water_potential_triggered()
{
    int layerIndex = std::max(1, ui->layerNrEdit->value());
    showCriteria3DVariable(waterMatricPotential, layerIndex, false, false, NODATA, NODATA);
}


void MainWindow::on_actionView_DegreeOfSaturation_automatic_range_triggered()
{
    int layerIndex = ui->layerNrEdit->value();
    showCriteria3DVariable(degreeOfSaturation, layerIndex, false, false, NODATA, NODATA);
}


void MainWindow::on_actionView_DegreeOfSaturation_fixed_range_triggered()
{
    int layerIndex = ui->layerNrEdit->value();
    showCriteria3DVariable(degreeOfSaturation, layerIndex, true, false, 0.2, 1.0);
}


void MainWindow::on_actionView_Factor_of_safety_triggered()
{
    int layerIndex = std::max(1, ui->layerNrEdit->value());
    showCriteria3DVariable(factorOfSafety, layerIndex, true, false, 0, 2);
}


void MainWindow::on_actionView_Factor_of_safety_minimum_triggered()
{
    showCriteria3DVariable(minimumFactorOfSafety, NODATA, true, false, 0, 2);
}


void MainWindow::on_actionCriteria3D_update_subHourly_triggered(bool isChecked)
{
    myProject.showEachTimeStep = isChecked;
}


void MainWindow::on_actionCriteria3D_parallel_computing_triggered(bool isChecked)
{
    myProject.setParallelComputing(isChecked);
}


void MainWindow::on_flag_increase_slope_triggered(bool isChecked)
{
    myProject.increaseSlope = isChecked;
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

    bool isRangeFixed = myProject.criteria3DMap.colorScale->isFixedRange();
    bool isHideMinimum = myProject.criteria3DMap.colorScale->isHideMinimum();

    showCriteria3DVariable(current3DVariable, layerIndex, isRangeFixed, isHideMinimum,
                           myProject.criteria3DMap.colorScale->minimum(),
                           myProject.criteria3DMap.colorScale->maximum());

}


void MainWindow::on_actionCriteria3D_load_state_triggered()
{
    if (! myProject.isProjectLoaded())
    {
        myProject.logWarning(ERROR_STR_MISSING_PROJECT);
        return;
    }

    if (myProject.isModelRunning)
    {
        myProject.logWarning("The model is running, stop it before loading a state.");
        return;
    }

    QString statesPath = myProject.getProjectPath() + PATH_STATES;
    QList<QString> stateList;
    if (! myProject.getAllSavedState(stateList))
    {
        myProject.logError();
        return;
    }

    DialogLoadState dialogLoadState(stateList);
    if (dialogLoadState.result() != QDialog::Accepted)
        return;

    QString stateDirectory = statesPath + dialogLoadState.getSelectedState();
    if (! myProject.loadModelState(stateDirectory))
    {
        myProject.logError();
        return;
    }

    initializeCriteria3DInterface();
    updateOutputMap();

    loadMeteoPointsDataSingleDay(myProject.getCurrentDate(), true);
    redrawMeteoPoints(currentPointsVisualization, true);
}


void MainWindow::on_actionCriteria3D_load_external_state_triggered()
{
    if (! myProject.isProjectLoaded())
    {
        myProject.logWarning(ERROR_STR_MISSING_PROJECT);
        return;
    }

    if (myProject.isModelRunning)
    {
        myProject.logWarning("The model is running, stop it before loading a state.");
        return;
    }

    QString stateDirectory = QFileDialog::getExistingDirectory(this, tr("Open Directory"), myProject.getProjectPath(),
                                                               QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
    if (stateDirectory.isEmpty())
        return;

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
    QString dirName;
    if (myProject.saveModelsState(dirName))
    {
        myProject.logInfoGUI("State successfully saved: " + dirName);
    }
    else
    {
        myProject.logError();
    }
}


void MainWindow::on_actionCreate_new_land_use_map_triggered()
{
    if (! myProject.DEM.isLoaded)
    {
        myProject.logWarning(ERROR_STR_MISSING_DEM);
        return;
    }

    // set default value
    float defaultValue = 1;
    bool isOk = false;
    while (! isOk)
    {
        QString valueStr = editValue("Enter the default value for land use:", QString::number(defaultValue));
        if (valueStr.isEmpty())
            return;

        defaultValue = valueStr.toFloat(&isOk);
        if (! isOk)
        {
            myProject.logWarning("Wrong value: only numeric values are accepted.");
        }
    }

    // initialize land use map with DEM
    gis::Crit3DRasterGrid landUseMap;
    landUseMap.initializeGrid(myProject.DEM);

    for (int row = 0; row < myProject.DEM.header->nrRows; row++)
    {
        for (int col = 0; col < myProject.DEM.header->nrCols; col++)
        {
            if (! isEqual(myProject.DEM.value[row][col], myProject.DEM.header->flag))
            {
                landUseMap.value[row][col] = defaultValue;
            }
        }
    }

    // set fileName
    QString completeFileName = QFileDialog::getSaveFileName(this, tr("Save land use map"), "", tr("ESRI float (*.flt)"));
    if (completeFileName.isEmpty())
        return;

    std::string fileName = completeFileName.left(completeFileName.size() - 4).toStdString();

    // save map
    std::string errorStr;
    if (! gis::writeEsriGrid(fileName, &landUseMap, errorStr))
    {
        myProject.logError(QString::fromStdString(errorStr));
        return;
    }

    myProject.loadLandUseMap(completeFileName);
}


void MainWindow::on_actionSave_outputRaster_triggered()
{
    if (! rasterOutput->visible())
    {
        myProject.logWarning("No current output.");
        return;
    }

    // set fileName
    QString outputFileName = QFileDialog::getSaveFileName(this, tr("Save output raster"), "", tr("ESRI float (*.flt)"));
    if (outputFileName.isEmpty())
        return;
    std::string fileName = outputFileName.left(outputFileName.size() - 4).toStdString();

    // write raster
    std::string errorStr;
    if (! gis::writeEsriGrid(fileName, rasterOutput->getRasterPointer(), errorStr))
    {
        myProject.logError(QString::fromStdString(errorStr));
    }
}


void MainWindow::on_actionDecomposable_plant_matter_triggered()
{
    if (myProject.isRothCInitialized)
    {
        if (myProject.rothCModel.map.decomposablePlantMaterial->isLoaded)
        {
            setColorScale(noMeteoTerrain, myProject.rothCModel.map.decomposablePlantMaterial->colorScale);
            setCurrentRasterOutput((myProject.rothCModel.map.decomposablePlantMaterial));
            ui->labelOutputRaster->setText("Decomposable plant matter");
        }
        else
        {
            myProject.logError("Error while loading decomposable plant matter.");
        }
    }
    else
    {
        myProject.logWarning("Initialize RothC model before.");
    }
}


void MainWindow::on_actionResistant_plant_matter_triggered()
{
    if (myProject.isRothCInitialized)
    {
        if (myProject.rothCModel.map.resistantPlantMaterial->isLoaded)
        {
            setColorScale(noMeteoTerrain, myProject.rothCModel.map.resistantPlantMaterial->colorScale);
            setCurrentRasterOutput((myProject.rothCModel.map.resistantPlantMaterial));
            ui->labelOutputRaster->setText("Resistant plant matter");
        }
        else
        {
            myProject.logError("Error while loading resistant plant matter.");
        }
    }
    else
    {
        myProject.logWarning("Initialize RothC model before.");
    }
}


void MainWindow::on_actionMicrobial_biomass_triggered()
{
    if (myProject.isRothCInitialized)
    {
        if (myProject.rothCModel.map.microbialBiomass->isLoaded)
        {
            setColorScale(noMeteoTerrain, myProject.rothCModel.map.microbialBiomass->colorScale);
            setCurrentRasterOutput((myProject.rothCModel.map.microbialBiomass));
            ui->labelOutputRaster->setText("Microbial biomass");
        }
        else
        {
            myProject.logError("Error while loading microbial biomass.");
        }
    }
    else
    {
        myProject.logWarning("Initialize RothC model before.");
    }
}


void MainWindow::on_actionHumified_organic_matter_triggered()
{
    if (myProject.isRothCInitialized)
    {
        if (myProject.rothCModel.map.humifiedOrganicMatter->isLoaded)
        {
            setColorScale(noMeteoTerrain, myProject.rothCModel.map.humifiedOrganicMatter->colorScale);
            setCurrentRasterOutput((myProject.rothCModel.map.humifiedOrganicMatter));
            ui->labelOutputRaster->setText("Humified organic matter");
        }
        else
        {
            myProject.logError("Error while loading humified organic matter.");
        }
    }
    else
    {
        myProject.logWarning("Initialize RothC model before.");
    }
}


void MainWindow::on_actionSoil_organic_matter_triggered()
{
    if (myProject.isRothCInitialized)
    {
        if (myProject.rothCModel.map.soilOrganicMatter->isLoaded)
        {
            setColorScale(noMeteoTerrain, myProject.rothCModel.map.soilOrganicMatter->colorScale);
            setCurrentRasterOutput((myProject.rothCModel.map.soilOrganicMatter));
            ui->labelOutputRaster->setText("Soil organic matter");
        }
        else
        {
            myProject.logError("Error while loading soil organic matter.");
        }
    }
    else
    {
        myProject.logWarning("Initialize RothC model before.");
    }
}


void MainWindow::on_actionAutomatic_state_saving_end_of_year_triggered(bool isChecked)
{
    myProject.setSaveYearlyState(isChecked);
}


void MainWindow::on_actionAutomatic_state_saving_end_of_month_toggled(bool isChecked)
{
    myProject.setSaveMonthlyState(isChecked);
}


void MainWindow::on_actionHide_TreeCover_map_triggered()
{
    setOutputRasterVisible(false);
    refreshViewer3D();
}


void MainWindow::on_actionViewTree_cover_map_triggered()
{
    if (myProject.treeCoverMap.isLoaded)
    {
        setColorScale(noMeteoVar, myProject.treeCoverMap.colorScale);
        setCurrentRasterOutput(&(myProject.treeCoverMap));
        ui->labelOutputRaster->setText("Tree cover");
    }
    else
    {
        myProject.logWarning("Load a tree cover map before.");
    }
}


void MainWindow::on_actiontree_NPP_triggered()
{
    if (myProject.isHydrallInitialized)
    {
        if (myProject.hydrallMaps.treeNetPrimaryProduction->isLoaded)
        {
            setColorScale(noMeteoTerrain, myProject.hydrallMaps.treeNetPrimaryProduction->colorScale);
            setCurrentRasterOutput((myProject.hydrallMaps.treeNetPrimaryProduction));
            ui->labelOutputRaster->setText("Tree net primary production");
        }
        else
        {
            myProject.logError("Error while loading tree net primary production.");
        }
    }
    else
    {
        myProject.logWarning("Initialize Hydrall model before.");
    }
}


void MainWindow::on_actionunderstorey_NPP_triggered()
{
    if (myProject.isHydrallInitialized)
    {
        if (myProject.hydrallMaps.understoreyNetPrimaryProduction->isLoaded)
        {
            setColorScale(noMeteoTerrain, myProject.hydrallMaps.understoreyNetPrimaryProduction->colorScale);
            setCurrentRasterOutput((myProject.hydrallMaps.understoreyNetPrimaryProduction));
            ui->labelOutputRaster->setText("Understorey net primary production");
        }
        else
        {
            myProject.logError("Error while loading understorey net primary production.");
        }
    }
    else
    {
        myProject.logWarning("Initialize Hydrall model before.");
    }
}


void MainWindow::on_actiontree_foliage_biomass_triggered()
{
    if (myProject.isHydrallInitialized)
    {
        if (myProject.hydrallMaps.treeBiomassFoliage->isLoaded)
        {
            setColorScale(noMeteoTerrain, myProject.hydrallMaps.treeBiomassFoliage->colorScale);
            setCurrentRasterOutput((myProject.hydrallMaps.treeBiomassFoliage));
            ui->labelOutputRaster->setText("Tree foliage biomass");
        }
        else
        {
            myProject.logError("Error while loading tree foliage biomass.");
        }
    }
    else
    {
        myProject.logWarning("Initialize Hydrall model before.");
    }
}


void MainWindow::on_actiontree_root_biomass_triggered()
{
    if (myProject.isHydrallInitialized)
    {
        if (myProject.hydrallMaps.treeBiomassRoot->isLoaded)
        {
            setColorScale(noMeteoTerrain, myProject.hydrallMaps.treeBiomassRoot->colorScale);
            setCurrentRasterOutput((myProject.hydrallMaps.treeBiomassRoot));
            ui->labelOutputRaster->setText("Tree root biomass");
        }
        else
        {
            myProject.logError("Error while loading tree root biomass.");
        }
    }
    else
    {
        myProject.logWarning("Initialize Hydrall model before.");
    }
}


void MainWindow::on_actiontree_sapwood_biomass_triggered()
{
    if (myProject.isHydrallInitialized)
    {
        if (myProject.hydrallMaps.treeBiomassSapwood->isLoaded)
        {
            setColorScale(noMeteoTerrain, myProject.hydrallMaps.treeBiomassSapwood->colorScale);
            setCurrentRasterOutput((myProject.hydrallMaps.treeBiomassSapwood));
            ui->labelOutputRaster->setText("Tree sapwood biomass");
        }
        else
        {
            myProject.logError("Error while loading tree sapwood biomass.");
        }
    }
    else
    {
        myProject.logWarning("Initialize Hydrall model before.");
    }
}


void MainWindow::on_actionunderstorey_foliage_biomass_triggered()
{
    if (myProject.isHydrallInitialized)
    {
        if (myProject.hydrallMaps.understoreyBiomassFoliage->isLoaded)
        {
            setColorScale(noMeteoTerrain, myProject.hydrallMaps.understoreyBiomassFoliage->colorScale);
            setCurrentRasterOutput((myProject.hydrallMaps.understoreyBiomassFoliage));
            ui->labelOutputRaster->setText("Understorey foliage biomass");
        }
        else
        {
            myProject.logError("Error while loading understorey foliage biomass.");
        }
    }
    else
    {
        myProject.logWarning("Initialize Hydrall model before.");
    }
}


void MainWindow::on_actionunderstorey_root_biomass_triggered()
{
    if (myProject.isHydrallInitialized)
    {
        if (myProject.hydrallMaps.understoreyBiomassRoot->isLoaded)
        {
            setColorScale(noMeteoTerrain, myProject.hydrallMaps.understoreyBiomassRoot->colorScale);
            setCurrentRasterOutput((myProject.hydrallMaps.understoreyBiomassRoot));
            ui->labelOutputRaster->setText("Understorey root biomass");
        }
        else
        {
            myProject.logError("Error while loading understorey root biomass.");
        }
    }
    else
    {
        myProject.logWarning("Initialize Hydrall model before.");
    }
}


void MainWindow::on_actionoutput_carbon_triggered()
{
    if (myProject.isHydrallInitialized)
    {
        if (myProject.hydrallMaps.outputC->isLoaded)
        {
            setColorScale(noMeteoTerrain, myProject.hydrallMaps.outputC->colorScale);
            setCurrentRasterOutput((myProject.hydrallMaps.outputC));
            ui->labelOutputRaster->setText("Carbon output");
        }
        else
        {
            myProject.logError("Error while loading carbon output.");
        }
    }
    else
    {
        myProject.logWarning("Initialize Hydrall model before.");
    }
}


void MainWindow::on_actioncumulated_yearly_ET0_triggered()
{
    if (myProject.isHydrallInitialized)
    {
        if (myProject.hydrallMaps.yearlyET0->isLoaded)
        {
            setColorScale(noMeteoTerrain, myProject.hydrallMaps.yearlyET0->colorScale);
            setCurrentRasterOutput((myProject.hydrallMaps.yearlyET0));
            ui->labelOutputRaster->setText("Cumulated yearly ET0");
        }
        else
        {
            myProject.logError("Error while loading cumulated yearly ET0.");
        }
    }
    else
    {
        myProject.logWarning("Initialize Hydrall model before.");
    }
}


void MainWindow::on_actioncumulated_yearly_precipitation_triggered()
{
    if (myProject.isHydrallInitialized)
    {
        if (myProject.hydrallMaps.yearlyPrec->isLoaded)
        {
            setColorScale(noMeteoTerrain, myProject.hydrallMaps.yearlyPrec->colorScale);
            setCurrentRasterOutput((myProject.hydrallMaps.yearlyPrec));
            ui->labelOutputRaster->setText("Cumulated yearly prec");
        }
        else
        {
            myProject.logError("Error while loading understorey net primary production.");
        }
    }
    else
    {
        myProject.logWarning("Initialize Hydrall model before.");
    }
}

