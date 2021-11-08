#include <QGridLayout>
#include <QFileDialog>
#include <QtDebug>
#include <QMessageBox>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QListWidget>
#include <QRadioButton>
#include <QTextBrowser>
#include <QLineEdit>
#include <QLabel>

#include <sstream>
#include <iostream>
#include <fstream>

#include "tileSources/WebTileSource.h"

#include "commonConstants.h"
#include "gis.h"
#include "waterBalance.h"
#include "vine3DProject.h"
#include "utilities.h"
#include "Position.h"
#include "spatialControl.h"
#include "dialogInterpolation.h"
#include "dialogRadiation.h"
#include "dialogSettings.h"
#include "dialogSelection.h"
#include "formPeriod.h"
#include "mainWindow.h"
#include "ui_mainWindow.h"


extern Vine3DProject myProject;

#define MAPBORDER 8
#define TOOLSWIDTH 260

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    this->showPoints = true;

    this->myRubberBand = nullptr;

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
    this->setMapSource(WebTileSource::OPEN_STREET_MAP);

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

    connect(this->ui->dateEdit, SIGNAL(editingFinished()), this, SLOT(on_dateChanged()));

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


void MainWindow::mouseReleaseEvent(QMouseEvent *event){
    updateMaps();

    if (myRubberBand != nullptr && myRubberBand->isVisible())
    {
        QPoint lastCornerOffset = getMapPos(event->pos());
        QPoint firstCornerOffset = myRubberBand->getOrigin() - QPoint(MAPBORDER, MAPBORDER);
        QPoint pixelTopLeft;
        QPoint pixelBottomRight;

        if (firstCornerOffset.y() > lastCornerOffset.y())
        {
            if (firstCornerOffset.x() > lastCornerOffset.x())
            {
                // bottom to left
                pixelTopLeft = lastCornerOffset;
                pixelBottomRight = firstCornerOffset;
            }
            else
            {
                // bottom to right
                pixelTopLeft = QPoint(firstCornerOffset.x(), lastCornerOffset.y());
                pixelBottomRight = QPoint(lastCornerOffset.x(), firstCornerOffset.y());
            }
        }
        else
        {
            if (firstCornerOffset.x() > lastCornerOffset.x())
            {
                // top to left
                pixelTopLeft = QPoint(lastCornerOffset.x(), firstCornerOffset.y());
                pixelBottomRight = QPoint(firstCornerOffset.x(), lastCornerOffset.y());
            }
            else
            {
                // top to right
                pixelTopLeft = firstCornerOffset;
                pixelBottomRight = lastCornerOffset;
            }
        }

        QPointF topLeft = this->mapView->mapToScene(pixelTopLeft);
        QPointF bottomRight = this->mapView->mapToScene(pixelBottomRight);
        QRectF rectF(topLeft, bottomRight);
        gis::Crit3DGeoPoint pointSelected;

        foreach (StationMarker* marker, pointList)
        {
            if (rectF.contains(marker->longitude(), marker->latitude()))
            {
                if ( marker->color() ==  Qt::white )
                {
                    marker->setFillColor(QColor((Qt::red)));
                    pointSelected.latitude = marker->latitude();
                    pointSelected.longitude = marker->longitude();
                    myProject.meteoPointsSelected << pointSelected;
                }
            }
        }

        myRubberBand->hide();
    }
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

    if (myRubberBand != nullptr && myRubberBand->isActive)
    {
        QPoint widgetPos = mapPos + QPoint(MAPBORDER, MAPBORDER);
        myRubberBand->setGeometry(QRect(myRubberBand->getOrigin(), widgetPos).normalized());
    }
}


void MainWindow::mousePressEvent(QMouseEvent *event)
{
    QPoint mapPos = getMapPos(event->pos());
    if (! isInsideMap(mapPos)) return;

    if (event->button() == Qt::RightButton)
    {
        if (myRubberBand != nullptr)
        {
            QPoint widgetPos = mapPos + QPoint(MAPBORDER, MAPBORDER);
            myRubberBand->setOrigin(widgetPos);
            myRubberBand->setGeometry(QRect(widgetPos, QSize()));
            myRubberBand->isActive = true;
            myRubberBand->show();
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
    resetMeteoPoints();
    addMeteoPoints();
    updateDateTime();
    myProject.loadObsDataAllPoints(myProject.getCurrentDate(), myProject.getCurrentDate(), true);
    showPointsGroup->setEnabled(true);
    currentPointsVisualization = showLocation;
    redrawMeteoPoints(currentPointsVisualization, true);
}


void MainWindow::on_actionOpen_project_triggered()
{
    QString myFileName = QFileDialog::getOpenFileName(this,tr("Open Project"), "", tr("Project files (*.ini)"));
    if (myFileName == "") return;

    if (! myProject.loadVine3DProject(myFileName)) return;

    if (myProject.DEM.isLoaded)
        renderDEM();

    drawMeteoPoints();
}

void MainWindow::on_actionRun_models_triggered()
{
    if (! myProject.isProjectLoaded) return;

    QDateTime timeIni(QDateTime::currentDateTime());
    QDateTime timeFin(QDateTime::currentDateTime());

    FormPeriod formPeriod(&timeIni, &timeFin);
    formPeriod.show();
    int myReturn = formPeriod.exec();
    if (myReturn == QDialog::Rejected) return;

    myProject.runModels(timeIni, timeFin, true, true, myProject.idArea);
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


void MainWindow::resetMeteoPoints()
{
    for (int i = 0; i < this->pointList.size(); i++)
        this->mapView->scene()->removeObject(this->pointList[i]);

    this->pointList.clear();

    this->myRubberBand = nullptr;
}


void MainWindow::on_actionVariableQualitySpatial_triggered()
{
    myProject.checkSpatialQuality = ui->actionVariableQualitySpatial->isChecked();
    updateVariable();
}


void MainWindow::interpolateDemGUI()
{
    meteoVariable myVar = myProject.getCurrentVariable();

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


void MainWindow::on_dateChanged()
{

    QDate date = this->ui->dateEdit->date();

    if (date != myProject.getCurrentDate())
    {
        myProject.setCurrentDate(date);
        myProject.loadObsDataAllPoints(date, date, true);
    }

    redrawMeteoPoints(currentPointsVisualization, true);
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
        pointList[i]->setVisible(false);

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
                    pointList[i]->setFillColor(QColor(Qt::white));
                    pointList[i]->setRadius(5);
                    pointList[i]->setToolTip();
                    pointList[i]->setVisible(true);
            }

            myProject.meteoPointsColorScale->setRange(NODATA, NODATA);
            meteoPointsLegend->update();
            break;
        }
        case showCurrentVariable:
        {
            this->ui->actionShowPointsVariable->setChecked(true);

            // quality control
            checkData(myProject.quality, myProject.getCurrentVariable(),
                      myProject.meteoPoints, myProject.nrMeteoPoints, myProject.getCrit3DCurrentTime(),
                      &myProject.qualityInterpolationSettings, myProject.meteoSettings,
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

                    pointList[i]->setToolTip();
                    pointList[i]->setVisible(true);
                }
            }

            meteoPointsLegend->update();
            break;
        }
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
    meteoVariable myVar = chooseMeteoVariable(&myProject);
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
    if (this->rasterObj->getRaster() == nullptr)
    {
        QMessageBox::information(nullptr, "No Raster", "Load raster before");
        return;
    }

    setDefaultDEMScale(myProject.DEM.colorScale);
    this->setCurrentRaster(&(myProject.DEM));
    ui->labelRasterScale->setText(QString::fromStdString(getVariableString(noMeteoTerrain)));
}

void MainWindow::setCurrentRaster(gis::Crit3DRasterGrid *myRaster)
{
    this->rasterObj->initializeUTM(myRaster, myProject.gisSettings, false);
    this->rasterLegend->colorScale = myRaster->colorScale;
    this->rasterObj->redrawRequested();
}

void MainWindow::on_dateEdit_dateChanged(const QDate &date)
{
    Q_UNUSED(date)
    this->on_dateChanged();
}

void MainWindow::on_actionInterpolation_to_DEM_triggered()
{
    myProject.logInfoGUI("Interpolation...");

    interpolateDemGUI();

    myProject.closeLogInfo();
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
        myProject.logInfoGUI("Load a Digital Elevation Model.");
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
            myProject.logError("Initialize model");
            return;
        }
}



void MainWindow::on_actionVine3D_InitializeWaterBalance_triggered()
{
    if (! myProject.setVine3DSoilIndexMap()) return;

    if (myProject.initializeWaterBalance3D())
    {
        myProject.outputWaterBalanceMaps = new Crit3DWaterBalanceMaps(myProject.DEM);
        QMessageBox::information(nullptr, "", "Criteria3D initialized.");
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
