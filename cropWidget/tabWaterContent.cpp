#include "tabWaterContent.h"
#include "commonConstants.h"


TabWaterContent::TabWaterContent()
{
    QVBoxLayout *mainLayout = new QVBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;
    graphic = new QCustomPlot();

    // configure axis rect:
    graphic->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom); // this will also allow rescaling the color scale by dragging/zooming
    graphic->axisRect()->setupFullAxesBox(true);
    graphic->xAxis->setLabel("Date");

    QSharedPointer<QCPAxisTickerDateTime> dateTicker(new QCPAxisTickerDateTime);
    dateTicker->setDateTimeFormat("MMM d");
    dateTicker->setTickCount(13);
    graphic->xAxis->setTicker(dateTicker);
    QDateTime first(QDate(QDate::currentDate().year(), 1, 1), QTime(0, 0, 0));
    QDateTime last(QDate(QDate::currentDate().year(), 12, 31), QTime(23, 0, 0));
    double firstDouble = first.toTime_t();
    double lastDouble = last.toTime_t();
    graphic->xAxis->setRange(firstDouble, lastDouble);
    graphic->xAxis->setVisible(true);
    graphic->yAxis->setLabel("Water Content");
    graphic->yAxis->setRangeReversed(true);


    // set up the QCPColorMap:
    colorMap = new QCPColorMap(graphic->xAxis2, graphic->yAxis);
    nx = 365;
    ny = 20;
    colorMap->data()->setSize(nx, ny); // we want the color map to have nx * ny data points
    colorMap->data()->setRange(QCPRange(0, 364), QCPRange(2, 0)); // and span the coordinate range in both key (x) and value (y) dimensions

    // add a color scale:
    colorScale = new QCPColorScale(graphic);
    colorScale->setDataRange(QCPRange(0,1));
    graphic->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
    colorScale->setType(QCPAxis::atRight); // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
    colorMap->setColorScale(colorScale); // associate the color map with the color scale

    // set the color gradient of the color map to one of the presets:
    //colorMap->setGradient(QCPColorGradient::gpPolar);
    // create a QCPColorGradient instance and added own colors to the gradient
    gradient.clearColorStops();
    gradient.setColorStopAt(0, QColor(128, 0, 128));
    gradient.setColorStopAt(0.25, QColor(255, 0, 0));
    gradient.setColorStopAt(0.5, QColor(255, 255, 0));
    gradient.setColorStopAt(0.75, QColor(64, 196, 64));
    gradient.setColorStopAt(1, QColor(0, 0, 255));
    colorMap->setGradient(gradient);

    // rescale the data dimension (color) such that all data points lie in the span visualized by the color gradient:
    colorMap->rescaleDataRange();

    // make sure the axis rect and color scale synchronize their bottom and top margins (so they line up):
    QCPMarginGroup *marginGroup = new QCPMarginGroup(graphic);
    graphic->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
    colorScale->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);

    // rescale the key (x) and value (y) axes so the whole color map is visible:
    graphic->rescaleAxes();

    plotLayout->addWidget(graphic);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}

void TabWaterContent::computeWaterContent(Crit1DCase myCase, int currentYear, bool isVolumetricWaterContent)
{
    if (isVolumetricWaterContent)
    {
        title = "volumetric water content [m3 m-3]";
    }
    else
    {
        title = "degree of saturation [-]";
    }

    unsigned int nrLayers = unsigned(myCase.soilLayers.size());
    double totalSoilDepth = 0;
    if (nrLayers > 0)
    {
        totalSoilDepth = myCase.soilLayers[nrLayers-1].depth + myCase.soilLayers[nrLayers-1].thickness / 2;
    }

    int currentDoy = 1;
    year = currentYear;
    int prevYear = currentYear - 1;

    Crit3DDate firstDate = Crit3DDate(1, 1, prevYear);
    Crit3DDate lastDate = Crit3DDate(31, 12, year);

    // update axes and colorMap size
    nx = getDoyFromDate(lastDate);
    ny = nrLayers-1;
    colorMap->data()->setSize(nx, ny);
    colorMap->data()->setRange(QCPRange(0, nx), QCPRange(totalSoilDepth,0));
    colorMap->rescaleDataRange(true);
    graphic->rescaleAxes();

    int doy;
    myCase.myCrop.initialize(myCase.meteoPoint.latitude, nrLayers, totalSoilDepth, currentDoy);

    std::string errorString;
    double waterContent = 0.0;
    double maxWaterContent = 0.0;
    for (Crit3DDate myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        if (! myCase.computeDailyModel(myDate, errorString))
        {
            QMessageBox::critical(nullptr, "Error!", QString::fromStdString(errorString));
            return;
        }
        // display only current year
        if (myDate.year == year)
        {
            doy = getDoyFromDate(myDate);
            for (unsigned int i = 1; i < nrLayers; i++)
            {
                waterContent = myCase.soilLayers[i].waterContent;
                if (waterContent != NODATA)
                {
                    if (isVolumetricWaterContent)
                    {
                        waterContent = waterContent/(myCase.soilLayers[i].thickness*1000);
                        if (waterContent > maxWaterContent)
                        {
                            maxWaterContent = waterContent;
                        }
                    }
                    else
                    {
                        waterContent = waterContent/myCase.soilLayers[i].SAT;
                    }
                    /*
                    qDebug() << " doy " << QString::number(doy);
                    qDebug() << " myCase.soilLayers[i].depth " << QString::number(myCase.soilLayers[i].depth);
                    qDebug() << " waterContent " << QString::number(waterContent);
                    */
                    colorMap->data()->setCell(doy-1, i-1, waterContent);
                }
            }
        }
    }
    double step;
    if(isVolumetricWaterContent)
    {
        step = maxWaterContent/4;
        colorScale->setDataRange(QCPRange(0, maxWaterContent));
    }
    else
    {
        maxWaterContent = 1;
        step = maxWaterContent/4;
        colorScale->setDataRange(QCPRange(0,1));
    }
    gradient.clearColorStops();
    gradient.setColorStopAt(0, QColor(128, 0, 128));
    gradient.setColorStopAt(step, QColor(255, 0, 0));
    gradient.setColorStopAt(step*2, QColor(255, 255, 0));
    gradient.setColorStopAt(step*3, QColor(64, 196, 64));
    gradient.setColorStopAt(maxWaterContent, QColor(0, 0, 255));
    colorMap->setGradient(gradient);
    colorMap->setColorScale(colorScale);

    colorScale->axis()->setLabel(title);
    graphic->replot();
}

