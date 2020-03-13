#include "tabIrrigation.h"

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
    seriesMaxTransp->setColor(QColor(Qt::darkGray));
    seriesRealTransp->setColor(QColor(Qt::red));

    seriesPrec = new QBarSeries();
    seriesIrrigation = new QBarSeries();
    seriesPrec->setName("Precipitation");
    seriesIrrigation->setName("Irrigation");
    setPrec = new QBarSet("Precipitation");
    setIrrigation = new QBarSet("Irrigation");
    setPrec->setColor(QColor(Qt::blue));
    setIrrigation->setColor(QColor(Qt::cyan));
    seriesPrec->append(setPrec);
    seriesIrrigation->append(setIrrigation);

    axisX = new QDateTimeAxis();
    axisY = new QValueAxis();
    axisYdx = new QBarCategoryAxis();

    QDate first(QDate::currentDate().year(), 1, 1);
    QDate last(QDate::currentDate().year(), 12, 31);
    axisX->setTitleText("Date");
    axisX->setFormat("MMM dd");
    axisX->setMin(QDateTime(first, QTime(0,0,0)));
    axisX->setMax(QDateTime(last, QTime(0,0,0)));
    axisX->setTickCount(13);

    QFont font = axisY->titleFont();
    font.setPointSize(9);
    font.setBold(true);
    axisY->setTitleText("LAI [m2 m-2] - Crop transpiration [mm]");
    axisY->setTitleFont(font);
    axisY->setRange(0,9);   // LC se lo faccio arrivare ad 8 i 2 reticolati di asse dx e sx non coincidono come invece così
    axisY->setTickCount(10);

    axisYdx->setTitleText("Prec - Irrigation [mm]");

    int i = 0;
    while (i<=40)
    {
        categories.append(QString::number(i));
        i = i+5;
    }
    axisYdx->append(categories);

    chart->addAxis(axisX, Qt::AlignBottom);
    chart->addAxis(axisY, Qt::AlignLeft);
    chart->addAxis(axisYdx, Qt::AlignRight);

    seriesLAI->attachAxis(axisX);
    seriesLAI->attachAxis(axisY);
    seriesMaxTransp->attachAxis(axisX);
    seriesMaxTransp->attachAxis(axisY);
    seriesRealTransp->attachAxis(axisX);
    seriesRealTransp->attachAxis(axisY);
    seriesPrec->attachAxis(axisX);
    seriesPrec->attachAxis(axisYdx);
    seriesIrrigation->attachAxis(axisX);
    seriesIrrigation->attachAxis(axisYdx);

    chart->addSeries(seriesLAI);
    chart->addSeries(seriesMaxTransp);
    chart->addSeries(seriesRealTransp);
    chart->addSeries(seriesPrec);
    chart->addSeries(seriesIrrigation);

    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignBottom);

    plotLayout->addWidget(chartView);
    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);
}
