#include "tabWaterContent.h"
#include "commonConstants.h"
#include "formInfo.h"
#include "criteria1DCase.h"


TabWaterContent::TabWaterContent()
{
    QVBoxLayout *mainLayout = new QVBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;
    graphic = new QCustomPlot();

    // configure axis rect:
    graphic->axisRect()->setupFullAxesBox(true);
    graphic->xAxis->setLabel("Date");

    QSharedPointer<QCPAxisTickerDateTime> dateTicker(new QCPAxisTickerDateTime);
    dateTicker->setDateTimeFormat("MMM d \n yyyy");
    dateTicker->setTickCount(13);
    graphic->xAxis->setTicker(dateTicker);
    QDateTime first(QDate(QDate::currentDate().year(), 1, 1), QTime(0, 0, 0));
    QDateTime last(QDate(QDate::currentDate().year(), 12, 31), QTime(23, 0, 0));
    double firstDouble = first.toSecsSinceEpoch();
    double lastDouble = last.toSecsSinceEpoch();
    graphic->xAxis->setRange(firstDouble, lastDouble);
    graphic->xAxis->setVisible(true);
    graphic->yAxis->setLabelFont(QFont("Noto Sans", 8, QFont::Bold));
    graphic->yAxis->setLabel("Depth [m]");
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

void TabWaterContent::computeWaterContent(Crit1DCase myCase, int firstYear, int lastYear, QDate lastDBMeteoDate, bool isVolumetricWaterContent)
{

    FormInfo formInfo;
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
    int prevYear = firstYear - 1;

    Crit3DDate firstDate = Crit3DDate(1, 1, prevYear);
    Crit3DDate lastDate;
    if (lastYear != lastDBMeteoDate.year())
    {
        lastDate = Crit3DDate(31, 12, lastYear);
    }
    else
    {
        lastDate = Crit3DDate(lastDBMeteoDate.day(), lastDBMeteoDate.month(), lastYear);
    }

    // update axes and colorMap size
    QDateTime first(QDate(firstYear, 1, 1), QTime(0, 0, 0));
    QDateTime last(QDate(lastDate.year, lastDate.month, lastDate.day), QTime(23, 0, 0));
    double firstDouble = first.toSecsSinceEpoch();
    double lastDouble = last.toSecsSinceEpoch();
    graphic->xAxis->setRange(firstDouble, lastDouble);

    nx = first.date().daysTo(last.date())+1;
    ny = (nrLayers-1);

    colorMap->data()->setSize(nx, ny);
    colorMap->data()->setRange(QCPRange(0, nx), QCPRange(totalSoilDepth,0));

    myCase.crop.initialize(myCase.meteoPoint.latitude, nrLayers, totalSoilDepth, currentDoy);
    myCase.initializeWaterContent(firstDate);

    std::string errorString;
    double waterContent = 0.0;
    double maxWaterContent = 0.0;

    int step = formInfo.start("Compute model...", (lastYear-firstYear+2)*365);
    int cont = 0;
    int doy = 0;
    for (Crit3DDate myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        if (! myCase.computeDailyModel(myDate, errorString))
        {
            QMessageBox::critical(nullptr, "Error!", QString::fromStdString(errorString));
            return;
        }
        if ( (cont % step) == 0) formInfo.setValue(cont);
        // display only interval firstYear lastYear
        if (myDate.year >= firstYear)
        {
            doy = doy+1; // if display 1 year this is the day Of year, otherwise count all days in that period
            for (unsigned int i = 1; i < nrLayers; i++)
            {
                if (isVolumetricWaterContent)
                {
                    waterContent = myCase.soilLayers[i].getVolumetricWaterContent() * myCase.soilLayers[i].soilFraction;
                    maxWaterContent = MAXVALUE(waterContent, maxWaterContent);
                }
                else
                {
                    waterContent = myCase.soilLayers[i].getDegreeOfSaturation();
                }
                colorMap->data()->setCell(doy-1, i-1, waterContent);
            }
        }
        cont++; // formInfo update
    }
    formInfo.close();

    if(isVolumetricWaterContent)
    {
        colorScale->setDataRange(QCPRange(0, maxWaterContent));
    }
    else
    {
        colorScale->setDataRange(QCPRange(0, 1));
    }

    gradient.clearColorStops();
    gradient.setColorStopAt(0, QColor(128, 0, 128));
    gradient.setColorStopAt(0.25, QColor(255, 0, 0));
    gradient.setColorStopAt(0.5, QColor(255, 255, 0));
    gradient.setColorStopAt(0.75, QColor(64, 196, 64));
    gradient.setColorStopAt(1, QColor(0, 0, 255));
    colorMap->setGradient(gradient);
    colorMap->setColorScale(colorScale);

    graphic->rescaleAxes();

    colorScale->axis()->setLabel(title);
    colorScale->axis()->setLabelFont(QFont("Noto Sans", 8, QFont::Bold));
    graphic->replot();
}

