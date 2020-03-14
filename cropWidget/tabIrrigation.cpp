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

    chart->addSeries(seriesLAI);
    chart->addSeries(seriesMaxTransp);
    chart->addSeries(seriesRealTransp);
    chart->addSeries(seriesPrec);
    chart->addSeries(seriesIrrigation);


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


    chart->legend()->setVisible(true);
    chart->legend()->setAlignment(Qt::AlignBottom);

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
    QDateTime x;
    int doy;

    seriesLAI->clear();
    seriesMaxTransp->clear();
    seriesRealTransp->clear();

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
            x.setDate(QDate(myDate.year, myDate.month, myDate.day));
            doy = getDoyFromDate(myDate);
            seriesLAI->append(x.toMSecsSinceEpoch(), myCase.myCrop.LAI);
            seriesMaxTransp->append(x.toMSecsSinceEpoch(), myCase.output.dailyMaxTranspiration);
            seriesRealTransp->append(x.toMSecsSinceEpoch(), myCase.output.dailyTranspiration);

        }
        cont = cont + 1; // formInfo update

    }
    formInfo.close();

    // update x axis
    QDate first(year, 1, 1);
    QDate last(year, 12, 31);
    axisX->setMin(QDateTime(first, QTime(0,0,0)));
    axisX->setMax(QDateTime(last, QTime(0,0,0)));
}
