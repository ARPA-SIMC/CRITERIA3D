#include "dialogNewCrop.h"

DialogNewCrop::DialogNewCrop()
{
    setWindowTitle("New Crop");
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QGridLayout *layoutCrop = new QGridLayout();
    QHBoxLayout *layoutOk = new QHBoxLayout();

    QLabel *idCropLabel = new QLabel(tr("Enter crop ID: "));
    idCropValue = new QLineEdit();

    QLabel *idCropName = new QLabel(tr("Enter crop name: "));
    nameCropValue = new QLineEdit();

    QLabel *typeCropLabel = new QLabel(tr("Enter crop type: "));
    typeCropValue = new QLineEdit();

    layoutCrop->addWidget(idCropLabel, 0 , 0);
    layoutCrop->addWidget(idCropValue, 0 , 1);
    layoutCrop->addWidget(idCropName, 1 , 0);
    layoutCrop->addWidget(nameCropValue, 1 , 1);
    layoutCrop->addWidget(typeCropLabel, 2 , 0);
    layoutCrop->addWidget(typeCropValue, 2 , 1);

    // TO DO quali sono le info necessarie?

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    layoutOk->addWidget(&buttonBox);


    mainLayout->addLayout(layoutCrop);
    mainLayout->addLayout(layoutOk);

    setLayout(mainLayout);

    show();
    exec();

}
