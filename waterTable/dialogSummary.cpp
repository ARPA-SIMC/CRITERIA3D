#include "dialogSummary.h"

DialogSummary::DialogSummary(WaterTable myWaterTable)
{
    setWindowTitle("Summary");
    QVBoxLayout* mainLayout = new QVBoxLayout;
    this->resize(350, 200);


    QHBoxLayout *layoutOk = new QHBoxLayout;
    QGridLayout *infoLayout = new QGridLayout;

    QLabel labelId("ID: ");
    QLineEdit myId(myWaterTable.getIdWell());

    QLabel labelObsData("Nr of observed depth: ");
    QLineEdit myObsData(QString::number(myWaterTable.getNrObsData()));

    QLabel labelAlpha("alpha [-]: ");
    QLineEdit myAlpha(QString::number(myWaterTable.getAlpha()));

    QLabel labelH0("H0 [cm]: ");
    QLineEdit myH0(QString::number(myWaterTable.getH0()));

    QLabel labelNrDays("Nr days: ");
    QLineEdit myNrDays(QString::number(myWaterTable.getNrDaysPeriod()));

    QLabel labelR2("R2 [-]: ");
    QLineEdit myR2(QString::number(myWaterTable.getR2()));

    QLabel labelRMSE("RMSE [cm]: ");
    QLineEdit myRMSE(QString::number(myWaterTable.getRMSE()));

    QLabel labelNASH("Nash-Sutcliffe [-]: ");
    QLineEdit myNASH(QString::number(myWaterTable.getNASH()));

    QLabel labelEfIndex("Efficiency Index [-]: ");
    QLineEdit myEfIndex(QString::number(myWaterTable.getEF()));

    infoLayout->addWidget(&labelId,0,0,1,1);
    infoLayout->addWidget(&myId,0,1,1,1);

    infoLayout->addWidget(&labelObsData,1,0,1,1);
    infoLayout->addWidget(&myObsData,1,1,1,1);

    infoLayout->addWidget(&labelAlpha,2,0,1,1);
    infoLayout->addWidget(&myAlpha,2,1,1,1);

    infoLayout->addWidget(&labelH0,3,0,1,1);
    infoLayout->addWidget(&myH0,3,1,1,1);

    infoLayout->addWidget(&labelNrDays,4,0,1,1);
    infoLayout->addWidget(&myNrDays,4,1,1,1);

    infoLayout->addWidget(&labelR2,5,0,1,1);
    infoLayout->addWidget(&myR2,5,1,1,1);

    infoLayout->addWidget(&labelRMSE,6,0,1,1);
    infoLayout->addWidget(&myRMSE,6,1,1,1);

    infoLayout->addWidget(&labelNASH,7,0,1,1);
    infoLayout->addWidget(&myNASH,7,1,1,1);

    infoLayout->addWidget(&labelEfIndex,8,0,1,1);
    infoLayout->addWidget(&myEfIndex,8,1,1,1);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    layoutOk->addWidget(&buttonBox);

    mainLayout->addLayout(infoLayout);
    mainLayout->addLayout(layoutOk);

    setLayout(mainLayout);
    exec();
}
