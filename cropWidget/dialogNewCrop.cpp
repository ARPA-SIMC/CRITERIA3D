#include "dialogNewCrop.h"
#include "crop.h"

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

    QLabel *typeCropLabel = new QLabel(tr("Select crop type: "));
    QComboBox* typeCropComboBox = new QComboBox();

    for (int i=0; i<numSpeciesType; i++)
    {
        speciesType type = (speciesType) i;
        typeCropComboBox->addItem(QString::fromStdString(getCropTypeString(type)));
    }

    layoutCrop->addWidget(idCropLabel, 0 , 0);
    layoutCrop->addWidget(idCropValue, 0 , 1);
    layoutCrop->addWidget(idCropName, 1 , 0);
    layoutCrop->addWidget(nameCropValue, 1 , 1);
    layoutCrop->addWidget(typeCropLabel, 2 , 0);
    layoutCrop->addWidget(typeCropComboBox, 2 , 1);

    // isPluriannal: NO sowingDoy e cycleMaxduration, SI valori di default NULL e 365

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
