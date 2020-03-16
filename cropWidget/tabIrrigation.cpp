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
    seriesMaxTransp->setColor(QColor(Qt::darkGray));
    seriesRealTransp->setColor(QColor(Qt::red));

    seriesPrecIrr = new QBarSeries();

    setPrec = new QBarSet("Precipitation");
    setIrrigation = new QBarSet("Irrigation");
    setPrec->setColor(QColor(Qt::blue));
    setIrrigation->setColor(QColor(Qt::cyan));
    seriesPrecIrr->append(setPrec);
    seriesPrecIrr->append(setIrrigation);

    axisX = new QBarCategoryAxis();
    axisY = new QValueAxis();
    axisYdx = new QValueAxis();

    QDate first(QDate::currentDate().year(), 1, 1);
    QDate last(QDate::currentDate().year(), 12, 31);
    axisX->setTitleText("Date");
    categories << "Jan 01" << "Feb 01" << "Mar 01" << "Apr 01" << "May 01" << "Jun 01" << "Jul 01" << "Aug 01" << "Sep 01" << "Oct 01" << "Nov 01" << "Dic 01";
    axisX->append(categories);

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

    int currentDoy = 1;
    myCase.myCrop.initialize(myCase.meteoPoint.latitude, nrLayers, totalSoilDepth, currentDoy);

    std::string errorString;
    int step = formInfo.start("Compute model...", 730);
    int cont = 0;
    for (Crit3DDate myDate = firstDate; myDate <= lastDate; ++myDate)
    {
        if ( (cont % step) == 0)
        {
            formInfo.setValue(cont);
        }
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
        cont = cont + 1; // formInfo update

    }
    formInfo.close();
    axisX->append(categories);

    // add histogram series
    seriesPrecIrr->append(setPrec);
    seriesPrecIrr->append(setIrrigation);
    chart->addSeries(seriesPrecIrr);

}
