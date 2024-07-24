#include <QMessageBox>
#include <QLegendMarker>

#include "tabLAI.h"
#include "commonConstants.h"
#include "formInfo.h"
#include "meteoPoint.h"
//#include "qdebug.h"
#include "crop.h"


TabLAI::TabLAI()
{
    QHBoxLayout *mainLayout = new QHBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;
    chart = new QChart();   
    chartView = new QChartView(chart);
    chartView->setChart(chart);

    seriesLAI = new QLineSeries();
    seriesETP = new QLineSeries();
    seriesMaxTransp = new QLineSeries();
    seriesMaxEvap = new QLineSeries();

    seriesLAI->setName("Leaf Area Index [m2 m-2] ");
    seriesETP->setName("Potential evapotranspiration [mm] ");
    seriesMaxTransp->setName("Maximum transpiration [mm] ");
    seriesMaxEvap->setName("Maximum evaporation [mm] ");

    QPen pen;
    pen.setWidth(2);
    seriesLAI->setPen(pen);

    seriesLAI->setColor(QColor(0, 200, 0, 255));
    seriesMaxTransp->setColor(QColor(Qt::red));
    seriesMaxEvap->setColor(QColor(Qt::blue));

    // bug with black
    seriesETP->setColor(QColor(0, 0, 16, 255));

    axisX = new QDateTimeAxis();
    axisY = new QValueAxis();
    axisYdx = new QValueAxis();

    chart->addSeries(seriesLAI);
    chart->addSeries(seriesETP);
    chart->addSeries(seriesMaxTransp);
    chart->addSeries(seriesMaxEvap);

    QDate first(QDate::currentDate().year(), 1, 1);
    QDate last(QDate::currentDate().year(), 12, 31);
    axisX->setTitleText("Date");
    axisX->setFormat("MMM dd <br> yyyy");
    axisX->setMin(QDateTime(first, QTime(0,0,0)));
    axisX->setMax(QDateTime(last, QTime(0,0,0)));
    axisX->setTickCount(13);
    chart->addAxis(axisX, Qt::AlignBottom);
    seriesLAI->attachAxis(axisX);
    seriesETP->attachAxis(axisX);
    seriesMaxEvap->attachAxis(axisX);
    seriesMaxTransp->attachAxis(axisX);

    QFont font = axisX->titleFont();

    qreal maximum = 8;
    axisY->setTitleText("Leaf Area Index [m2 m-2]");
    axisY->setTitleFont(font);
    axisY->setRange(0, maximum);
    axisY->setTickCount(maximum+1);

    axisYdx->setTitleText("Evapotranspiration [mm]");
    axisYdx->setTitleFont(font);
    axisYdx->setRange(0, maximum);
    axisYdx->setTickCount(maximum+1);

    chart->addAxis(axisY, Qt::AlignLeft);
    chart->addAxis(axisYdx, Qt::AlignRight);
    seriesLAI->attachAxis(axisY);
    seriesETP->attachAxis(axisYdx);
    seriesMaxEvap->attachAxis(axisYdx);
    seriesMaxTransp->attachAxis(axisYdx);

    chart->legend()->setVisible(true);
    QFont legendFont = chart->legend()->font();
    legendFont.setPointSize(8);
    legendFont.setBold(true);
    chart->legend()->setFont(legendFont);
    chart->legend()->setAlignment(Qt::AlignBottom);
    chart->setAcceptHoverEvents(true);

    m_tooltip = new Callout(chart);
    m_tooltip->hide();

    connect(seriesLAI, &QLineSeries::hovered, this, &TabLAI::tooltipLAI);
    connect(seriesETP, &QLineSeries::hovered, this, &TabLAI::tooltipPE);
    connect(seriesMaxEvap, &QLineSeries::hovered, this, &TabLAI::tooltipME);
    connect(seriesMaxTransp, &QLineSeries::hovered, this, &TabLAI::tooltipMaxTranspiration);

    foreach(QLegendMarker* marker, chart->legend()->markers())
    {
        QObject::connect(marker, &QLegendMarker::clicked, this, &TabLAI::handleMarkerClicked);
    }

    plotLayout->addWidget(chartView);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}


void TabLAI::computeLAI(Crit3DCrop* myCrop, Crit3DMeteoPoint *meteoPoint, int firstYear, int lastYear,
                        const QDate &lastDBMeteoDate, const std::vector<soil::Crit1DLayer> &soilLayers)
{
    unsigned int nrLayers = unsigned(soilLayers.size());
    double totalSoilDepth = 0;
    if (nrLayers > 0) totalSoilDepth = soilLayers[nrLayers-1].depth + soilLayers[nrLayers-1].thickness / 2;

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

    double tmin, tmax, waterTableDepth;
    QDateTime x;
    double dailyEt0;
    int doy;
    std::string errorStr;

    chart->removeSeries(seriesLAI);
    chart->removeSeries(seriesETP);
    chart->removeSeries(seriesMaxEvap);
    chart->removeSeries(seriesMaxTransp);

    seriesLAI->clear();
    seriesETP->clear();
    seriesMaxEvap->clear();
    seriesMaxTransp->clear();

    int currentDoy = 1;
    myCrop->initialize(meteoPoint->latitude, nrLayers, totalSoilDepth, currentDoy);

    for (Crit3DDate myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        tmin = meteoPoint->getMeteoPointValueD(myDate, dailyAirTemperatureMin);
        tmax = meteoPoint->getMeteoPointValueD(myDate, dailyAirTemperatureMax);
        waterTableDepth = meteoPoint->getMeteoPointValueD(myDate, dailyWaterTableDepth);

        if (!myCrop->dailyUpdate(myDate, meteoPoint->latitude, soilLayers, tmin, tmax, waterTableDepth, errorStr))
        {
            QMessageBox::critical(nullptr, "Error!", QString::fromStdString(errorStr));
            return;
        }

        // display only interval firstYear lastYear
        if (myDate.year >= firstYear)
        {         
            x.setDate(QDate(myDate.year, myDate.month, myDate.day));
            doy = getDoyFromDate(myDate);

            // ET0
            dailyEt0 = ET0_Hargreaves(TRANSMISSIVITY_SAMANI_COEFF_DEFAULT, meteoPoint->latitude, doy, tmax, tmin);
            seriesETP->append(x.toMSecsSinceEpoch(), dailyEt0);
            seriesMaxEvap->append(x.toMSecsSinceEpoch(), myCrop->getMaxEvaporation(dailyEt0));
            seriesMaxTransp->append(x.toMSecsSinceEpoch(), myCrop->getMaxTranspiration(dailyEt0));
            seriesLAI->append(x.toMSecsSinceEpoch(), myCrop->LAI);
        }
    }

    // update x axis
    QDate first(firstYear, 1, 1);
    QDate last(lastDate.year, lastDate.month, lastDate.day);
    axisX->setMin(QDateTime(first, QTime(0,0,0)));
    axisX->setMax(QDateTime(last, QTime(0,0,0)));

    chart->addSeries(seriesLAI);
    chart->addSeries(seriesETP);
    chart->addSeries(seriesMaxEvap);
    chart->addSeries(seriesMaxTransp);

    seriesLAI->attachAxis(axisY);
    seriesETP->attachAxis(axisYdx);
    seriesMaxEvap->attachAxis(axisYdx);
    seriesMaxTransp->attachAxis(axisYdx);

    foreach(QLegendMarker* marker, chart->legend()->markers())
    {
        QObject::connect(marker, &QLegendMarker::clicked, this, &TabLAI::handleMarkerClicked);
    }
}


void TabLAI::tooltipLAI(QPointF point, bool state)
{
    if (state)
    {
        QDateTime xDate;
        xDate.setMSecsSinceEpoch(point.x());
        m_tooltip->setText(QString("%1\nLAI: %2 [m2 m-2]").arg(xDate.date().toString("yyyy-MM-dd")).arg(point.y(), 0, 'f', 1));
        m_tooltip->setAnchor(point);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();
    } else {
        m_tooltip->hide();
    }
}

void TabLAI::tooltipPE(QPointF point, bool state)
{
    if (state)
    {
        QDateTime xDate;
        xDate.setMSecsSinceEpoch(point.x());
        m_tooltip->setText(QString("%1\nETP: %2 mm").arg(xDate.date().toString("yyyy-MM-dd")).arg(point.y(), 0, 'f', 1));
        m_tooltip->setAnchor(point);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();
    } else {
        m_tooltip->hide();
    }
}

void TabLAI::tooltipME(QPointF point, bool state)
{
    if (state)
    {
        QDateTime xDate;
        xDate.setMSecsSinceEpoch(point.x());
        m_tooltip->setText(QString("%1\nmax. Evaporation: %2 mm").arg(xDate.date().toString("yyyy-MM-dd")).arg(point.y(), 0, 'f', 1));
        m_tooltip->setAnchor(point);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();
    } else {
        m_tooltip->hide();
    }
}

void TabLAI::tooltipMaxTranspiration(QPointF point, bool state)
{

    if (state)
    {
        QDateTime xDate;
        xDate.setMSecsSinceEpoch(point.x());
        m_tooltip->setText(QString("%1\nmax. Transpiration: %2 mm").arg(xDate.date().toString("yyyy-MM-dd")).arg(point.y(), 0, 'f', 1));
        m_tooltip->setAnchor(point);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();
    } else {
        m_tooltip->hide();
    }
}

void TabLAI::handleMarkerClicked()
{
    QLegendMarker* marker = qobject_cast<QLegendMarker*> (sender());

    if(marker->type() == QLegendMarker::LegendMarkerTypeXY)
    {
        // Toggle visibility of series
        marker->series()->setVisible(!marker->series()->isVisible());

        // Turn legend marker back to visible, since otherwise hiding series also hides the marker
        marker->setVisible(true);

        // change marker alpha, if series is not visible
        qreal alpha = 1.0;

        if (!marker->series()->isVisible()) {
            alpha = 0.5;
        }

        QColor color;
        QBrush brush = marker->labelBrush();
        color = brush.color();
        color.setAlphaF(alpha);
        brush.setColor(color);
        marker->setLabelBrush(brush);

        brush = marker->brush();
        color = brush.color();
        color.setAlphaF(alpha);
        brush.setColor(color);
        marker->setBrush(brush);

        QPen pen = marker->pen();
        color = pen.color();
        color.setAlphaF(alpha);
        pen.setColor(color);
        marker->setPen(pen);
    }

}


