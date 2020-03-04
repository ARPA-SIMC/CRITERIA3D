#include "tabRootDensity.h"
#include "commonConstants.h"

TabRootDensity::TabRootDensity()
{
    QVBoxLayout *mainLayout = new QVBoxLayout;
    QVBoxLayout *sliderLayout = new QVBoxLayout;
    QVBoxLayout *dateLayout = new QVBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;

    dateLayout->setAlignment(Qt::AlignHCenter);
    currentDate = new QDateEdit;
    slider = new QSlider(Qt::Horizontal);
    slider->setMinimum(1);
    QDate middleDate(currentDate->date().year(),06,30);
    slider->setMaximum(QDate(middleDate.year(),12,31).dayOfYear());
    slider->setValue(middleDate.dayOfYear());
    currentDate->setDate(middleDate);
    currentDate->setDisplayFormat("MMM dd");
    currentDate->setMaximumWidth(this->width()/5);
    chart = new QChart();
    chartView = new QChartView(chart);
    chartView->setChart(chart);

    seriesRootDensity = new QHorizontalBarSeries();
    seriesRootDensity->setName("Rooth density");
    set = new QBarSet("");
    seriesRootDensity->append(set);
    chart->addSeries(seriesRootDensity);

    axisX = new QValueAxis();
    axisY = new QBarCategoryAxis();

    axisX->setTitleText("Rooth density [%]");
    axisX->setRange(0,2.2);
    axisX->setTickCount(12);
    axisX->setLabelFormat("%.2f");
    chart->addAxis(axisX, Qt::AlignBottom);
    seriesRootDensity->attachAxis(axisX);

    axisY->setTitleText("Depth [m]");
    //categories << "2.0" << "1.8" << "1.6" << "1.4" << "1.2" << "1.0" << "0.8" << "0.6" << "0.4" << "0.2" << "0.0";
    categories << "2.00" << "1.80" << "1.60" << "1.40" << "1.20" << "1.00" << "0.80" << "0.60" << "0.40" << "0.20" << "0.00";
    axisY->append(categories);
    chart->addAxis(axisY, Qt::AlignLeft);
    seriesRootDensity->attachAxis(axisY);

    chart->legend()->setVisible(false);
    nrLayers = 0;

    connect(currentDate, &QDateEdit::dateChanged, this, &TabRootDensity::updateRootDensity);
    connect(slider, &QSlider::valueChanged, this, &TabRootDensity::updateDate);
    plotLayout->addWidget(chartView);
    sliderLayout->addWidget(slider);
    dateLayout->addWidget(currentDate);
    mainLayout->addLayout(sliderLayout);
    mainLayout->addLayout(dateLayout);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}

void TabRootDensity::computeRootDensity(Crit3DCrop* myCrop, Crit3DMeteoPoint *meteoPoint, int currentYear, const std::vector<soil::Crit3DLayer> &soilLayers)
{

    crop = myCrop;
    mp = meteoPoint;
    layers = soilLayers;
    nrLayers = unsigned(soilLayers.size());
    year = currentYear;

    QDate lastDate(year,12,31);
    slider->setMaximum(lastDate.dayOfYear());

    double totalSoilDepth = 0;
    if (nrLayers > 0) totalSoilDepth = soilLayers[nrLayers-1].depth + soilLayers[nrLayers-1].thickness / 2;

    axisY->clear();
    categories.clear();
    for (int i = (nrLayers-1); i>0; i--)
    {
        if (soilLayers[i].depth <= 2)
        {
            double layerDepth = soilLayers[i].depth + soilLayers[i].thickness/2;
            QString depthStr = QString::number(layerDepth, 'f', 1);
            if (!categories.contains(depthStr))
            {
                categories << depthStr;
            }
        }
    }

    axisY->append(categories);
    int currentDoy = 1;
    myCrop->initialize(meteoPoint->latitude, nrLayers, totalSoilDepth, currentDoy);
    updateRootDensity();
}

void TabRootDensity::updateDate()
{
    int doy = slider->value();
    QDate newDate = QDate(year, 1, 1).addDays(doy - 1);
    if (newDate != currentDate->date())
    {
        currentDate->setDate(newDate);
    }

}

void TabRootDensity::updateRootDensity()
{

    QDate newDate(year,currentDate->date().month(),currentDate->date().day());
    slider->blockSignals(true);
    slider->setValue(newDate.dayOfYear());
    slider->blockSignals(false);
    if (crop == nullptr || mp == nullptr || nrLayers == 0)
    {
        return;
    }
    if (set != nullptr)
    {
        seriesRootDensity->remove(set);
        chart->removeSeries(seriesRootDensity);
    }
    set = new QBarSet("");

    std::string error;
    int prevYear = year - 1;
    Crit3DDate firstDate = Crit3DDate(1, 1, prevYear);
    Crit3DDate lastDate = Crit3DDate(currentDate->date().day(), currentDate->date().month(), year);
    double waterTableDepth = NODATA;
    double tmin;
    double tmax;
    double maxRootDensity = 0;
    for (Crit3DDate myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        tmin = mp->getMeteoPointValueD(myDate, dailyAirTemperatureMin);
        tmax = mp->getMeteoPointValueD(myDate, dailyAirTemperatureMax);

        if (!crop->dailyUpdate(myDate, mp->latitude, layers, tmin, tmax, waterTableDepth, error))
        {
            QMessageBox::critical(nullptr, "Error!", QString::fromStdString(error));
            return;
        }

        // display only current doy
        if (myDate == lastDate)
        {
            for (int i = (nrLayers-1); i>0; i--)
            {
                if (layers[i].depth <= 2)
                {
                    *set << crop->roots.rootDensity[i]*100;
                    if (crop->roots.rootDensity[i]*100 > maxRootDensity)
                    {
                        maxRootDensity = crop->roots.rootDensity[i]*100;
                    }
                }
            }
        }
    }

    maxRootDensity = ceil(maxRootDensity);
    axisX->setRange(0, maxRootDensity);
    seriesRootDensity->append(set);
    chart->addSeries(seriesRootDensity);
}

