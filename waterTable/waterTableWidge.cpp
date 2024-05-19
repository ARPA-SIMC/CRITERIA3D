#include "waterTableWidget.h"

WaterTableWidget::WaterTableWidget(QString id, std::vector<QDate> myDates, std::vector<float> myValues, std::vector<float> myHindcastSeries, std::vector<float> myInterpolateSeries)
{
    this->setWindowTitle("Graph Id well: "+ id);
    this->resize(1240, 700);
    this->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    this->setAttribute(Qt::WA_DeleteOnClose);

    // layout
    QHBoxLayout *mainLayout = new QHBoxLayout();
    QVBoxLayout *plotLayout = new QVBoxLayout();

    waterTableChartView = new WaterTableChartView();
    plotLayout->addWidget(waterTableChartView);

    mainLayout->addLayout(plotLayout);
    setLayout(mainLayout);

    plot(myDates, myValues, myHindcastSeries, myInterpolateSeries);
    show();
}

void WaterTableWidget::plot(std::vector<QDate> myDates, std::vector<float> myValues, std::vector<float> myHindcastSeries, std::vector<float> myInterpolateSeries)
{
    // TO DO
}
