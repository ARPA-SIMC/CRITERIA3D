#include "tabRootDensity.h"
#include "commonConstants.h"
#include "criteria1DProject.h"
#include "meteoPoint.h"

TabRootDensity::TabRootDensity()
{
    QVBoxLayout *mainLayout = new QVBoxLayout;
    QVBoxLayout *sliderLayout = new QVBoxLayout;
    QHBoxLayout *dateLayout = new QHBoxLayout;
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
    yearComboBox.addItem(QString::number(QDate::currentDate().year()));
    yearComboBox.setMaximumWidth(this->width()/5);
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

    axisY->setTitleText("Depth [m]");

    double i = 1.95;
    while (i > 0)
    {
        categories.append(QString::number(i, 'f', 2));
        i = i-0.1;
    }
    axisY->append(categories);
    chart->addAxis(axisY, Qt::AlignLeft);
    seriesRootDensity->attachAxis(axisY);

    chart->legend()->setVisible(false);
    nrLayers = 0;

    m_tooltip = new Callout(chart);
    m_tooltip->hide();

    connect(currentDate, &QDateEdit::dateChanged, this, &TabRootDensity::updateRootDensity);
    connect(slider, &QSlider::valueChanged, this, &TabRootDensity::updateDate);
    connect(seriesRootDensity, &QHorizontalBarSeries::hovered, this, &TabRootDensity::tooltip);
    connect(&yearComboBox, &QComboBox::currentTextChanged, this, &TabRootDensity::on_actionChooseYear);
    plotLayout->addWidget(chartView);
    sliderLayout->addWidget(slider);
    dateLayout->addWidget(&yearComboBox);
    dateLayout->addWidget(currentDate);
    mainLayout->addLayout(sliderLayout);
    mainLayout->addLayout(dateLayout);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}


void TabRootDensity::computeRootDensity(const Crit1DProject &myProject, int firstYear, int lastYear)
{
    mp =  myProject.myCase.meteoPoint;
    crop = myProject.myCase.crop;
    layers = myProject.myCase.soilLayers;
    nrLayers = unsigned(layers.size());
    lastMeteoDate = myProject.lastSimulationDate;

    yearComboBox.blockSignals(true);
    yearComboBox.clear();
    for (int i = firstYear; i<=lastYear; i++)
    {
        yearComboBox.addItem(QString::number(i));
    }
    year = yearComboBox.currentText().toInt();
    yearComboBox.blockSignals(false);

    if (year == myProject.lastSimulationDate.year())
    {
        slider->setMaximum(myProject.lastSimulationDate.dayOfYear());
        currentDate->setDate(myProject.lastSimulationDate);
    }
    else
    {
        QDate middleDate(currentDate->date().year(), 06, 30);
        QDate lastDate(year,12,31);
        slider->setMaximum(lastDate.dayOfYear());
        currentDate->setDate(middleDate);
    }

    double totalSoilDepth = 0;
    if (nrLayers > 0) totalSoilDepth = myProject.myCase.soilLayers[nrLayers-1].depth
                                        + myProject.myCase.soilLayers[nrLayers-1].thickness / 2;

    axisY->clear();
    categories.clear();
    depthLayers.clear();
    double i = totalSoilDepth-0.05;
    while (i >= 0)
    {
        categories.append(QString::number(i, 'f', 2));
        i = i-0.1;
    }

    double n = totalSoilDepth/0.02;
    double value;
    for (int i = n; i>0; i--)
    {
        value = (i-1)*0.02 + 0.01;
        depthLayers.append(value);
    }

    axisY->append(categories);

    int currentDoy = 1;
    crop.initialize(myProject.myCase.meteoPoint.latitude, nrLayers, totalSoilDepth, currentDoy);

    updateRootDensity();
}


void TabRootDensity::on_actionChooseYear(QString myYear)
{
    this->year = myYear.toInt();
    if (year == lastMeteoDate.year())
    {
        slider->setMaximum(lastMeteoDate.dayOfYear());
        currentDate->setDate(lastMeteoDate);
    }
    else
    {
        QDate middleDate(currentDate->date().year(),06,30);
        QDate lastDate(this->year,12,31);
        slider->setMaximum(lastDate.dayOfYear());
        currentDate->setDate(middleDate);
    }

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
    if (newDate > lastMeteoDate)
    {
        currentDate->setDate(lastMeteoDate);
        return;
    }
    slider->blockSignals(true);
    slider->setValue(newDate.dayOfYear());
    slider->blockSignals(false);

    if (nrLayers == 0) return;

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
        tmin = mp.getMeteoPointValueD(myDate, dailyAirTemperatureMin);
        tmax = mp.getMeteoPointValueD(myDate, dailyAirTemperatureMax);

        if (!crop.dailyUpdate(myDate, mp.latitude, layers, tmin, tmax, waterTableDepth, error))
        {
            QMessageBox::critical(nullptr, "Error!", QString::fromStdString(error));
            return;
        }

        // display only current doy
        if (myDate == lastDate)
        {
            for (int i = 0; i<depthLayers.size(); i++)
            {
                int layerIndex;
                double rootDensity;
                double rootDensityAdj;
                if (depthLayers[i] <= 2)
                {
                    layerIndex = getSoilLayerIndex(layers, depthLayers[i]);
                    if (layerIndex != NODATA)
                    {
                        rootDensity = crop.roots.rootDensity[layerIndex]*100;
                        rootDensityAdj = rootDensity/layers[layerIndex].thickness*0.02;
                        *set << rootDensityAdj;

                        if (rootDensityAdj > maxRootDensity)
                        {
                            maxRootDensity = rootDensityAdj;
                        }
                    }
                }
            }
        }
    }

    axisX->setRange(0, maxRootDensity);
    seriesRootDensity->append(set);
    chart->addSeries(seriesRootDensity);
}


void TabRootDensity::tooltip(bool state, int index, QBarSet *barset)
{
    if (state && barset!=nullptr && index < barset->count())
    {
        QString valueStr = QString::number(barset->at(index));
        m_tooltip->setText(valueStr);

        QPoint point = QCursor::pos();
        QPoint mapPoint = chartView->mapFromGlobal(point);
        QPointF pointF = chart->mapToValue(mapPoint,seriesRootDensity);

        m_tooltip->setAnchor(pointF);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();
    }
    else
    {
        m_tooltip->hide();
    }
}
