#include "tabRootDensity.h"

TabRootDensity::TabRootDensity()
{
    QHBoxLayout *mainLayout = new QHBoxLayout;
    QVBoxLayout *plotLayout = new QVBoxLayout;
    chart = new QChart();
    chartView = new QChartView(chart);
    chart->setTitle("Root Density");
    chartView->setChart(chart);
    seriesRootDensity = new QLineSeries();
    seriesRootDensity->setName("rooth density");
    seriesRootDensity->setColor(QColor(Qt::red));

    axisX = new QDateTimeAxis();
    axisY = new QValueAxis();

    chart->addSeries(seriesRootDensity);
    QDate first(QDate::currentDate().year(), 1, 1);
    QDate last(QDate::currentDate().year(), 12, 31);
    axisX->setTitleText("Date");
    axisX->setFormat("MMM dd");
    axisX->setMin(QDateTime(first, QTime(0,0,0)));
    axisX->setMax(QDateTime(last, QTime(0,0,0)));
    axisX->setTickCount(13);
    chart->addAxis(axisX, Qt::AlignBottom);
    seriesRootDensity->attachAxis(axisX);

    axisY->setTitleText("Depth  [m]");
    axisY->setReverse(true);
    axisY->setRange(0,2);
    axisY->setTickCount(5);
    chart->addAxis(axisY, Qt::AlignLeft);
    seriesRootDensity->attachAxis(axisY);

    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignBottom);

    chart->setAcceptHoverEvents(true);
    m_tooltip = new Callout(chart);
    connect(seriesRootDensity, &QLineSeries::hovered, this, &TabRootDensity::tooltip);

    plotLayout->addWidget(chartView);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}

void TabRootDensity::tooltip(QPointF point, bool state)
{
    /*
    if (m_tooltip == nullptr)
        m_tooltip = new Callout(chart);

    if (state)
    {
        QDateTime xDate;
        xDate.setMSecsSinceEpoch(point.x());
        m_tooltip->setText(QString("%1 \nroot ini %2 ").arg(xDate.date().toString("MMM dd")).arg(point.y()));
        m_tooltip->setAnchor(point);
        m_tooltip->setZValue(11);
        m_tooltip->updateGeometry();
        m_tooltip->show();
    } else {
        m_tooltip->hide();
    }
    */
}
