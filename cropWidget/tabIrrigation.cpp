#include "tabIrrigation.h"
#include "commonConstants.h"
#include "formInfo.h"

TabIrrigation::TabIrrigation()
{
    QHBoxLayout *mainLayout = new QHBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;
    chart = new QChart();
    chartView = new QChartView(chart);
    chartView->setChart(chart);

    seriesLAI = new QLineSeries();
    seriesMaxTransp = new QLineSeries();
    seriesRealTransp = new QLineSeries();

    seriesLAI->setName("LAI [m2 m-2]");
    seriesMaxTransp->setName("Transpiration max [mm]");
    seriesRealTransp->setName("Transpiration real [mm]");

    seriesLAI->setColor(QColor(Qt::darkGreen));
    seriesMaxTransp->setColor(QColor(0,0,1,255));
    seriesRealTransp->setColor(QColor(Qt::red));

    seriesPrecIrr = new QBarSeries();

    setPrec = new QBarSet("Precipitation");
    setIrrigation = new QBarSet("Irrigation");
    setPrec->setColor(QColor(Qt::blue));
    setPrec->setBorderColor(QColor(Qt::blue));
    setIrrigation->setColor(QColor(Qt::cyan));
    setIrrigation->setBorderColor(QColor(Qt::cyan));

    seriesPrecIrr->append(setPrec);
    seriesPrecIrr->append(setIrrigation);

    axisX = new QBarCategoryAxis();
    axisXvirtual = new QDateTimeAxis();
    axisY = new QValueAxis();
    axisYdx = new QValueAxis();

    QDate first(QDate::currentDate().year(), 1, 1);
    QDate last(QDate::currentDate().year(), 12, 31);
    axisX->setTitleText("Date");
    axisXvirtual->setTitleText("Date");
    axisXvirtual->setFormat("MMM dd");
    axisXvirtual->setMin(QDateTime(first, QTime(0,0,0)));
    axisXvirtual->setMax(QDateTime(last, QTime(0,0,0)));
    axisXvirtual->setTickCount(13);

    QFont font = axisY->titleFont();
    font.setPointSize(9);
    font.setBold(true);
    axisY->setTitleText("LAI [m2 m-2] - Crop transpiration [mm]");
    axisY->setTitleFont(font);
    axisY->setRange(0,8);
    axisY->setTickCount(9);

    axisYdx->setTitleText("Prec - Irrigation [mm]");
    axisYdx->setRange(0,40);
    axisYdx->setTickCount(9);

    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisXvirtual, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);
    chart->addAxis(axisYdx, Qt::AlignRight);

    chart->addSeries(seriesLAI);
    chart->addSeries(seriesMaxTransp);
    chart->addSeries(seriesRealTransp);
    chart->addSeries(seriesPrecIrr);
    seriesLAI->attachAxis(axisX);
    seriesLAI->attachAxis(axisY);

    seriesMaxTransp->attachAxis(axisX);
    seriesMaxTransp->attachAxis(axisY);
    seriesRealTransp->attachAxis(axisX);
    seriesRealTransp->attachAxis(axisY);

    seriesPrecIrr->attachAxis(axisX);
    seriesPrecIrr->attachAxis(axisYdx);

    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignBottom);
    chartView->setRenderHint(QPainter::Antialiasing);
    axisX->hide();

    m_tooltip = new Callout(chart);
    m_tooltip->hide();

    connect(seriesLAI, &QLineSeries::hovered, this, &TabIrrigation::tooltipLAI);
    connect(seriesMaxTransp, &QLineSeries::hovered, this, &TabIrrigation::tooltipMT);
    connect(seriesRealTransp, &QLineSeries::hovered, this, &TabIrrigation::tooltipRT);
    connect(seriesPrecIrr, &QHorizontalBarSeries::hovered, this, &TabIrrigation::tooltipPrecIrr);

    plotLayout->addWidget(chartView);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}


void TabIrrigation::computeIrrigation(Crit1DCase myCase, int currentYear)
{
    FormInfo formInfo;

    unsigned int nrLayers = unsigned(myCase.soilLayers.size());
    double totalSoilDepth = 0;
    if (nrLayers > 0) totalSoilDepth = myCase.soilLayers[nrLayers-1].depth + myCase.soilLayers[nrLayers-1].thickness / 2;

    year = currentYear;
    int prevYear = currentYear - 1;

    std::string error;

    Crit3DDate firstDate = Crit3DDate(1, 1, prevYear);
    Crit3DDate lastDate = Crit3DDate(31, 12, year);

    int doy;

    axisX->clear();
    seriesLAI->clear();
    seriesMaxTransp->clear();
    seriesRealTransp->clear();
    categories.clear();

    if (setPrec!= nullptr)
    {
        seriesPrecIrr->remove(setPrec);
        setPrec = new QBarSet("Precipitation");
        setPrec->setColor(QColor(Qt::blue));
        setPrec->setBorderColor(QColor(Qt::blue));
    }
    if (setIrrigation!= nullptr)
    {
        seriesPrecIrr->remove(setIrrigation);
        setIrrigation = new QBarSet("Irrigation");
        setIrrigation->setColor(QColor(Qt::cyan));
        setIrrigation->setBorderColor(QColor(Qt::cyan));
    }

    int currentDoy = 1;
    myCase.myCrop.initialize(myCase.meteoPoint.latitude, nrLayers, totalSoilDepth, currentDoy);

    std::string errorString;
    int step = formInfo.start("Compute model...", 730);
    int cont = 0;
    for (Crit3DDate myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        if ( (cont % step) == 0) formInfo.setValue(cont);

        if (! myCase.computeDailyModel(myDate, errorString))
        {
            QMessageBox::critical(nullptr, "Error!", QString::fromStdString(error));
            return;
        }
        // display only current year
        if (myDate.year == year)
        {
            doy = getDoyFromDate(myDate);
            categories.append(QString::number(doy));
            seriesLAI->append(doy, myCase.myCrop.LAI);
            seriesMaxTransp->append(doy, myCase.output.dailyMaxTranspiration);
            seriesRealTransp->append(doy, myCase.output.dailyTranspiration);
            *setPrec << myCase.output.dailyPrec;
            *setIrrigation << myCase.output.dailyIrrigation;
        }

        cont++; // formInfo update
    }

    formInfo.close();

    seriesPrecIrr->append(setPrec);
    seriesPrecIrr->append(setIrrigation);
    axisX->append(categories);
    axisX->setGridLineVisible(false);

    // update virtual x axis
    QDate first(year, 1, 1);
    QDate last(year, 12, 31);
    axisXvirtual->setMin(QDateTime(first, QTime(0,0,0)));
    axisXvirtual->setMax(QDateTime(last, QTime(0,0,0)));

    foreach(QLegendMarker* marker, chart->legend()->markers())
    {
        QObject::connect(marker, &QLegendMarker::clicked, this, &TabIrrigation::handleMarkerClicked);
    }
}

void TabIrrigation::tooltipLAI(QPointF point, bool state)
{
    if (state)
    {
        QDate xDate(year, 1, 1);
        int doy = point.x();
        xDate = xDate.addDays(doy-1);
        m_tooltip->setText(QString("%1 \nLAI %2 ").arg(xDate.toString("MMM dd")).arg(point.y(), 0, 'f', 1));
        m_tooltip->setAnchor(point);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();
    }
    else
    {
        m_tooltip->hide();
    }
}

void TabIrrigation::tooltipMT(QPointF point, bool state)
{
    if (state)
    {
        QDate xDate(year, 1, 1);
        int doy = point.x();
        xDate = xDate.addDays(doy-1);
        m_tooltip->setText(QString("%1 \nTransp max %2 ").arg(xDate.toString("MMM dd")).arg(point.y(), 0, 'f', 1));
        m_tooltip->setAnchor(point);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();
    }
    else
    {
        m_tooltip->hide();
    }

}

void TabIrrigation::tooltipRT(QPointF point, bool state)
{
    if (state)
    {
        QDate xDate(year, 1, 1);
        int doy = point.x();
        xDate = xDate.addDays(doy-1);
        m_tooltip->setText(QString("%1 \nTransp real %2 ").arg(xDate.toString("MMM dd")).arg(point.y(), 0, 'f', 1));
        m_tooltip->setAnchor(point);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();
    }
    else
    {
        m_tooltip->hide();
    }
}

void TabIrrigation::tooltipPrecIrr(bool state, int index, QBarSet *barset)
{

    if (state && barset!=nullptr && index < barset->count())
    {
        QString valueStr;
        if (barset->label() == "Precipitation")
        {
            valueStr = "Precipitation\n" + QString::number(barset->at(index));
        }
        else if (barset->label() == "Irrigation")
        {
            valueStr = "Irrigation\n" + QString::number(barset->at(index));
        }

        m_tooltip->setText(valueStr);

        QPoint point = QCursor::pos();
        QPoint mapPoint = chartView->mapFromGlobal(point);
        QPointF pointF = chart->mapToValue(mapPoint,seriesPrecIrr);
        int ratio = axisYdx->max()/axisY->max();
        pointF.setY(pointF.y()/ratio);

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

void TabIrrigation::handleMarkerClicked()
{

    QLegendMarker* marker = qobject_cast<QLegendMarker*> (sender());
    if (marker->type() == QLegendMarker::LegendMarkerTypeXY)
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
    else if (marker->type() == QLegendMarker::LegendMarkerTypeBar)
    {
        // Toggle visibility of series
        marker->series()->setVisible(!marker->series()->isVisible());

        // change marker alpha, if series is not visible
        qreal alpha = 1.0;

        // Turn legend marker back to visible, since otherwise hiding series also hides the marker
        foreach(QLegendMarker* marker, chart->legend()->markers())
        {
            if (marker->type() == QLegendMarker::LegendMarkerTypeBar)
            {
                marker->setVisible(true);
            }
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

}
