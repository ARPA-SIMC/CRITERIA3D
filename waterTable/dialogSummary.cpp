#include "dialogSummary.h"

DialogSummary::DialogSummary(WaterTable myWaterTable)
{
    setWindowTitle("Summary");
    QVBoxLayout* mainLayout = new QVBoxLayout;
    this->resize(350, 200);

    QGridLayout *infoLayout = new QGridLayout;

    QLabel labelId("ID: ");
    QLineEdit myId(myWaterTable.getIdWell());
    myId.setReadOnly(true);

    QLabel labelObsData("Nr of observed depth: ");
    QLineEdit myObsData(QString::number(myWaterTable.getNrObsData()));
    myObsData.setReadOnly(true);

    QLabel labelAlpha("alpha [-]: ");
    QLineEdit myAlpha(QString::number(myWaterTable.getAlpha(),'f', 2));
    myAlpha.setReadOnly(true);

    QLabel labelH0("H0 [cm]: ");
    QLineEdit myH0(QString::number((int)myWaterTable.getH0()));
    myH0.setReadOnly(true);

    QLabel labelNrDays("Nr days: ");
    QLineEdit myNrDays(QString::number(myWaterTable.getNrDaysPeriod()));
    myNrDays.setReadOnly(true);

    QLabel labelR2("R2 [-]: ");
    QLineEdit myR2(QString::number(myWaterTable.getR2(),'f', 2));
    myR2.setReadOnly(true);

    QLabel labelRMSE("RMSE [cm]: ");
    QLineEdit myRMSE(QString::number(myWaterTable.getRMSE(),'f', 2));
    myRMSE.setReadOnly(true);

    QLabel labelNASH("Nash-Sutcliffe [-]: ");
    QLineEdit myNASH(QString::number(myWaterTable.getNASH(),'f', 2));
    myNASH.setReadOnly(true);

    QLabel labelEfIndex("Efficiency Index [-]: ");
    QLineEdit myEfIndex(QString::number(myWaterTable.getEF(),'f', 2));
    myEfIndex.setReadOnly(true);

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

    mainLayout->addLayout(infoLayout);

    setLayout(mainLayout);
    exec();
}
