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
    graphic->yAxis->setLabel("Water Content");
    graphic->yAxis->setRangeReversed(true);

    // set up the QCPColorMap:
    colorMap = new QCPColorMap(graphic->xAxis, graphic->yAxis);
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
    colorMap->setGradient(QCPColorGradient::gpPolar);
    // we could have also created a QCPColorGradient instance and added own colors to
    // the gradient, see the documentation of QCPColorGradient for what's possible.

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

    /*
    depthLayers.clear();

    double n = totalSoilDepth/0.02;
    double value;
    for (int i = n; i>=0; i--)
    {
        value = i*0.02;
        depthLayers.append(value);
        qDebug() << "value " << value;
    }
qDebug() << "n " << n;
qDebug() << "totalSoilDepth " << totalSoilDepth;
*/
    int currentDoy = 1;
    year = currentYear;
    int prevYear = currentYear - 1;

    Crit3DDate firstDate = Crit3DDate(1, 1, prevYear);
    Crit3DDate lastDate = Crit3DDate(31, 12, year);

    // update axes and colorMap size
    nx = getDoyFromDate(lastDate);
    ny = nrLayers;
    colorMap->data()->setSize(nx, ny);
    colorMap->data()->setRange(QCPRange(0, nx), QCPRange(totalSoilDepth,0));
    colorMap->rescaleDataRange();
    graphic->rescaleAxes();

    int doy;
    myCase.myCrop.initialize(myCase.meteoPoint.latitude, nrLayers, totalSoilDepth, currentDoy);

    std::string errorString;
    double waterContent = 0.0;
    double maxWaterContent = -NODATA;
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
            /*
            for (int i = 0; i<depthLayers.size(); i++)
            {
                int layerIndex;
                if (depthLayers[i] <= 2)
                {
                    layerIndex = getSoilLayerIndex(myCase.soilLayers, depthLayers[i]);
                    if (layerIndex != NODATA)
                    {
                        waterContent = myCase.soilLayers[layerIndex].waterContent;
                        if (isVolumetricWaterContent)
                        {
                            waterContent = waterContent/myCase.soilLayers[layerIndex].thickness;
                            if (waterContent > maxWaterContent)
                            {
                                maxWaterContent = waterContent;
                            }
                        }
                        else
                        {
                            waterContent = waterContent/myCase.soilLayers[layerIndex].SAT;
                        }

                    }
                    colorMap->data()->setCell(doy, depthLayers[i], waterContent);
                }
            }
            */
            for (int i = 0; i<nrLayers; i++)
            {
                waterContent = myCase.soilLayers[i].waterContent;
                if (waterContent != NODATA)
                {
                    if (isVolumetricWaterContent)
                    {
                        waterContent = waterContent*myCase.soilLayers[i].thickness;
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
                    colorMap->data()->setCell(doy-1, i, waterContent);
                }
            }
        }
    }

    colorScale->axis()->setLabel(title);
    graphic->replot();
}

