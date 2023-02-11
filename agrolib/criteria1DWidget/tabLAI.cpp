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
    seriesPotentialEvap = new QLineSeries();
    seriesMaxEvap = new QLineSeries();
    seriesMaxTransp = new QLineSeries();
    seriesLAI->setName("Leaf Area Index [m2 m-2]");
    seriesPotentialEvap->setName("Potential evapotranspiration [mm]");
    seriesPotentialEvap->setColor(QColor(Qt::darkGray));
    seriesMaxEvap->setName("Evaporation max [mm]");
    seriesMaxEvap->setColor(QColor(Qt::blue));
    seriesMaxTransp->setName("Transpiration max [mm]");
    seriesMaxTransp->setColor(QColor(Qt::red));

    axisX = new QDateTimeAxis();
    axisY = new QValueAxis();
    axisYdx = new QValueAxis();

    chart->addSeries(seriesLAI);
    chart->addSeries(seriesPotentialEvap);
    chart->addSeries(seriesMaxEvap);
    chart->addSeries(seriesMaxTransp);
    QDate first(QDate::currentDate().year(), 1, 1);
    QDate last(QDate::currentDate().year(), 12, 31);
    axisX->setTitleText("Date");
    axisX->setFormat("MMM dd <br> yyyy");
    axisX->setMin(QDateTime(first, QTime(0,0,0)));
    axisX->setMax(QDateTime(last, QTime(0,0,0)));
    axisX->setTickCount(13);
    chart->addAxis(axisX, Qt::AlignBottom);
    seriesLAI->attachAxis(axisX);
    seriesPotentialEvap->attachAxis(axisX);
    seriesMaxEvap->attachAxis(axisX);
    seriesMaxTransp->attachAxis(axisX);

    QFont font = axisX->titleFont();

    axisY->setTitleText("Leaf Area Index [m2 m-2]");
    axisY->setTitleFont(font);
    axisY->setRange(0,7);
    axisY->setTickCount(8);

    QPen pen;
    pen.setWidth(3);
    pen.setBrush(Qt::green);

    axisYdx->setTitleText("Evapotranspiration [mm]");
    axisYdx->setRange(0,7);
    axisYdx->setTickCount(8);
    axisYdx->setTitleFont(font);

    seriesLAI->setPen(pen);

    chart->addAxis(axisY, Qt::AlignLeft);
    chart->addAxis(axisYdx, Qt::AlignRight);
    seriesLAI->attachAxis(axisY);
    seriesPotentialEvap->attachAxis(axisYdx);
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
    connect(seriesPotentialEvap, &QLineSeries::hovered, this, &TabLAI::tooltipPE);
    connect(seriesMaxEvap, &QLineSeries::hovered, this, &TabLAI::tooltipME);
    connect(seriesMaxTransp, &QLineSeries::hovered, this, &TabLAI::tooltipMT);

    foreach(QLegendMarker* marker, chart->legend()->markers())
    {
        QObject::connect(marker, &QLegendMarker::clicked, this, &TabLAI::handleMarkerClicked);
    }

    plotLayout->addWidget(chartView);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}


void TabLAI::computeLAI(Crit3DCrop* myCrop, Crit3DMeteoPoint *meteoPoint, int firstYear, int lastYear,
                        const QDate &lastDBMeteoDate, const std::vector<soil::Crit3DLayer> &soilLayers)
{
    unsigned int nrLayers = unsigned(soilLayers.size());
    double totalSoilDepth = 0;
    if (nrLayers > 0) totalSoilDepth = soilLayers[nrLayers-1].depth + soilLayers[nrLayers-1].thickness / 2;

    int prevYear = firstYear - 1;

    double waterTableDepth = NODATA;
    std::string error;

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
    double tmin;
    double tmax;
    QDateTime x;
    double dailyEt0;
    int doy;

    chart->removeSeries(seriesLAI);
    chart->removeSeries(seriesPotentialEvap);
    chart->removeSeries(seriesMaxEvap);
    chart->removeSeries(seriesMaxTransp);

    seriesLAI->clear();
    seriesPotentialEvap->clear();
    seriesMaxEvap->clear();
    seriesMaxTransp->clear();

    int currentDoy = 1;
    myCrop->initialize(meteoPoint->latitude, nrLayers, totalSoilDepth, currentDoy);

    for (Crit3DDate myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        tmin = meteoPoint->getMeteoPointValueD(myDate, dailyAirTemperatureMin);
        tmax = meteoPoint->getMeteoPointValueD(myDate, dailyAirTemperatureMax);

        if (!myCrop->dailyUpdate(myDate, meteoPoint->latitude, soilLayers, tmin, tmax, waterTableDepth, error))
        {
            QMessageBox::critical(nullptr, "Error!", QString::fromStdString(error));
            return;
        }

        // display only interval firstYear lastYear
        if (myDate.year >= firstYear)
        {         
            x.setDate(QDate(myDate.year, myDate.month, myDate.day));
            doy = getDoyFromDate(myDate);

            // ET0
            dailyEt0 = ET0_Hargreaves(TRANSMISSIVITY_SAMANI_COEFF_DEFAULT, meteoPoint->latitude, doy, tmax, tmin);
            seriesPotentialEvap->append(x.toMSecsSinceEpoch(), dailyEt0);
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
    chart->addSeries(seriesPotentialEvap);
    chart->addSeries(seriesMaxEvap);
    chart->addSeries(seriesMaxTransp);

    seriesLAI->attachAxis(axisY);
    seriesPotentialEvap->attachAxis(axisYdx);
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
        m_tooltip->setText(QString("%1 \nLAI: %2 ").arg(xDate.date().toString("yyyy-MM-dd")).arg(point.y(), 0, 'f', 1));
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
        m_tooltip->setText(QString("%1 \nPot. ET: %2 ").arg(xDate.date().toString("yyyy-MM-dd")).arg(point.y(), 0, 'f', 1));
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
        m_tooltip->setText(QString("%1 \nEvap. max: %2 ").arg(xDate.date().toString("yyyy-MM-dd")).arg(point.y(), 0, 'f', 1));
        m_tooltip->setAnchor(point);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();
    } else {
        m_tooltip->hide();
    }
}

void TabLAI::tooltipMT(QPointF point, bool state)
{

    if (state)
    {
        QDateTime xDate;
        xDate.setMSecsSinceEpoch(point.x());
        m_tooltip->setText(QString("%1 \nTransp. max: %2 ").arg(xDate.date().toString("yyyy-MM-dd")).arg(point.y(), 0, 'f', 1));
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


