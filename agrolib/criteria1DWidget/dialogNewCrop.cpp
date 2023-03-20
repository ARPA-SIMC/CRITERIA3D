#include "dialogNewCrop.h"
#include "crop.h"
#include "commonConstants.h"

DialogNewCrop::DialogNewCrop(Crit3DCrop *newCrop)
    :newCrop(newCrop)
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

    for (int i=0; i < NR_CROP_SPECIES; i++)
    {
        speciesType type = (speciesType) i;
        typeCropComboBox->addItem(QString::fromStdString(getCropTypeString(type)));
    }

    sowingDoY = new QLabel(tr("Enter sowing DOY: "));
    sowingDoYValue = new QSpinBox();
    sowingDoYValue->setMinimum(-365);
    sowingDoYValue->setMaximum(365);

    cycleMaxDuration = new QLabel(tr("Enter cycle max duration: "));
    cycleMaxDurationValue = new QSpinBox();
    cycleMaxDurationValue->setMinimum(0);
    cycleMaxDurationValue->setMaximum(365);

    layoutCrop->addWidget(idCropLabel, 0 , 0);
    layoutCrop->addWidget(idCropValue, 0 , 1);
    layoutCrop->addWidget(idCropName, 1 , 0);
    layoutCrop->addWidget(nameCropValue, 1 , 1);
    layoutCrop->addWidget(typeCropLabel, 2 , 0);
    layoutCrop->addWidget(typeCropComboBox, 2 , 1);
    layoutCrop->addWidget(sowingDoY, 3 , 0);
    layoutCrop->addWidget(sowingDoYValue, 3 , 1);
    layoutCrop->addWidget(cycleMaxDuration, 4 , 0);
    layoutCrop->addWidget(cycleMaxDurationValue, 4 , 1);

    newCrop->type = getCropType(typeCropComboBox->currentText().toStdString());
    if (newCrop->isSowingCrop())
    {
        sowingDoY->setVisible(true);
        sowingDoYValue->setVisible(true);
        cycleMaxDuration->setVisible(true);
        cycleMaxDurationValue->setVisible(true);
    }
    else
    {
        sowingDoY->setVisible(false);
        sowingDoYValue->setVisible(false);
        cycleMaxDuration->setVisible(false);
        cycleMaxDurationValue->setVisible(false);
        newCrop->sowingDoy = NODATA;
        newCrop->plantCycle = 365;
    }

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(typeCropComboBox, &QComboBox::currentTextChanged, this, &DialogNewCrop::on_actionChooseType);
    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    layoutOk->addWidget(&buttonBox);


    mainLayout->addLayout(layoutCrop);
    mainLayout->addLayout(layoutOk);

    setLayout(mainLayout);

    show();
    exec();

}

void DialogNewCrop::on_actionChooseType(QString type)
{
    newCrop->type = getCropType(type.toStdString());
    if (newCrop->isSowingCrop())
    {
        sowingDoY->setVisible(true);
        sowingDoYValue->setVisible(true);
        cycleMaxDuration->setVisible(true);
        cycleMaxDurationValue->setVisible(true);
    }
    else
    {
        sowingDoY->setVisible(false);
        sowingDoYValue->setVisible(false);
        cycleMaxDuration->setVisible(false);
        cycleMaxDurationValue->setVisible(false);
        newCrop->sowingDoy = NODATA;
        newCrop->plantCycle = 365;
    }
}


void DialogNewCrop::done(int res)
{
    if(res)  // ok was pressed
    {
        if (!checkData())
        {
            return;
        }
        newCrop->idCrop = idCropValue->text().toStdString();
        if (sowingDoY->isVisible())
        {
            newCrop->sowingDoy = sowingDoYValue->text().toInt();
            newCrop->plantCycle = cycleMaxDurationValue->text().toInt();
        }
        else
        {
            newCrop->sowingDoy = NODATA;
            newCrop->plantCycle = 365;
        }
        QDialog::done(QDialog::Accepted);
        return;

    }
    else    // cancel, close or exc was pressed
    {
        QDialog::done(QDialog::Rejected);
        return;
    }
}


bool DialogNewCrop::checkData()
{
    if (idCropValue->text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing parameter", "Insert ID CROP");
        return false;
    }
    if (nameCropValue->text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing parameter", "Insert ID NAME");
        return false;
    }
    if (sowingDoY->isVisible())
    {
        if (sowingDoYValue->text().isEmpty())
        {
            QMessageBox::information(nullptr, "Missing parameter", "Insert sowing day of year");
            return false;
        }
        if (cycleMaxDurationValue->text().isEmpty())
        {
            QMessageBox::information(nullptr, "Missing parameter", "Insert plant cycle max duration");
            return false;
        }
    }
    return true;

}

QString DialogNewCrop::getNameCrop()
{
    return nameCropValue->text();
}
