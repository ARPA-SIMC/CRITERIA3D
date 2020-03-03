#include "tabRootDensity.h"
#include "commonConstants.h"

TabRootDensity::TabRootDensity()
{
    QVBoxLayout *mainLayout = new QVBoxLayout;
    QHBoxLayout *dateLayout = new QHBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;
    dateLayout->setAlignment(Qt::AlignHCenter);
    QLabel *dateLabel = new QLabel(tr("select day and month: "));
    currentDate = new QDateEdit;
    QDate defaultDate(currentDate->date().year(), 06, 30);
    currentDate->setDate(defaultDate);
    currentDate->setDisplayFormat("MMM dd");
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
    axisX->setLabelFormat("%.1f");
    chart->addAxis(axisX, Qt::AlignBottom);
    seriesRootDensity->attachAxis(axisX);

    axisY->setTitleText("Depth  [m]");
    categories << "2.0" << "1.8" << "1.6" << "1.4" << "1.2" << "1.0" << "0.8" << "0.6" << "0.4" << "0.2" << "0.0";
    axisY->append(categories);
    chart->addAxis(axisY, Qt::AlignLeft);
    seriesRootDensity->attachAxis(axisY);

    chart->legend()->setVisible(false);
    nrLayers = 0;

    connect(currentDate, &QDateEdit::dateChanged, this, &TabRootDensity::updateRootDensity);
    plotLayout->addWidget(chartView);
    dateLayout->addWidget(dateLabel);
    dateLayout->addWidget(currentDate);
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

    double totalSoilDepth = 0;
    if (nrLayers > 0) totalSoilDepth = soilLayers[nrLayers-1].depth + soilLayers[nrLayers-1].thickness / 2;

    axisY->clear();
    categories.clear();
    for (int i = (nrLayers-1); i>0; i--)
    {
        if (soilLayers[i].depth <= 2)
        {
            if (!categories.contains(QString::number(soilLayers[i].depth, 'f', 1)))
            {
                categories << QString::number(soilLayers[i].depth, 'f', 1);
            }

        }
    }
    axisY->append(categories);
    int currentDoy = 1;
    myCrop->initialize(meteoPoint->latitude, nrLayers, totalSoilDepth, currentDoy);
    updateRootDensity();
}

void TabRootDensity::updateRootDensity()
{

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
    double maxDepth = 0;
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
                    if (crop->roots.rootDensity[i]*100 > maxDepth)
                    {
                        maxDepth = crop->roots.rootDensity[i]*100;
                    }
                }

            }

        }
    }
    maxDepth = qRound(maxDepth);
    axisX->setRange(0,maxDepth);
    seriesRootDensity->append(set);
    chart->addSeries(seriesRootDensity);
}

