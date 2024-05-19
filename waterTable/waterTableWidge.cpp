#include "waterTableWidget.h"

WaterTableWidget::WaterTableWidget(WaterTable myWaterTable)
{
    this->setWindowTitle("Graph Id well: "+myWaterTable.getIdWell());
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
}
