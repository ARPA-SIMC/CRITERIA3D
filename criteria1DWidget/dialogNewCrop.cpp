#include "dialogNewCrop.h"
#include "crop.h"
#include "cropDbQuery.h"
#include "commonConstants.h"

DialogNewCrop::DialogNewCrop(QSqlDatabase *dbCrop, Crit3DCrop *newCrop)
    :dbCrop(dbCrop), newCrop(newCrop)
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

    QString cropType = typeCropComboBox->currentText();
    newCrop->type = getCropType(cropType.toStdString());

    QList<QString> cropList;
    QString errorStr;
    if (! getCropListFromType(*dbCrop, cropType, cropList, errorStr))
    {
        QMessageBox::information(this, "Error in reading crop list", errorStr);
        return;
    }

    QLabel* templateCropLabel = new QLabel(tr("Copy other parameters from crop: "));
    templateCropComboBox = new QComboBox();
    for (int i=0; i < cropList.size(); i++)
    {
        templateCropComboBox->addItem(cropList[i]);
    }

    sowingDoY = new QLabel(tr("Enter sowing DOY: "));
    sowingDoYValue = new QSpinBox();
    sowingDoYValue->setMinimum(-365);
    sowingDoYValue->setMaximum(365);

    cycleMaxDuration = new QLabel(tr("Enter crop cycle max duration [days]: "));
    cycleMaxDurationValue = new QSpinBox();
    cycleMaxDurationValue->setMinimum(0);
    cycleMaxDurationValue->setMaximum(365);

    layoutCrop->addWidget(idCropLabel, 0 , 0);
    layoutCrop->addWidget(idCropValue, 0 , 1);
    layoutCrop->addWidget(idCropName, 1 , 0);
    layoutCrop->addWidget(nameCropValue, 1 , 1);
    layoutCrop->addWidget(typeCropLabel, 2 , 0);
    layoutCrop->addWidget(typeCropComboBox, 2 , 1);
    layoutCrop->addWidget(templateCropLabel, 3 , 0);
    layoutCrop->addWidget(templateCropComboBox, 3, 1);
    layoutCrop->addWidget(sowingDoY, 4, 0);
    layoutCrop->addWidget(sowingDoYValue, 4, 1);
    layoutCrop->addWidget(cycleMaxDuration, 5, 0);
    layoutCrop->addWidget(cycleMaxDurationValue, 5, 1);

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


void DialogNewCrop::on_actionChooseType(QString cropType)
{
    newCrop->type = getCropType(cropType.toStdString());

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

    templateCropComboBox->clear();

    QList<QString> cropList;
    QString errorStr;
    if (! getCropListFromType(*dbCrop, cropType, cropList, errorStr))
    {
        QMessageBox::information(this, "Error in reading crop list", errorStr);
        return;
    }

    for (int i=0; i < cropList.size(); i++)
    {
        templateCropComboBox->addItem(cropList[i]);
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
        QMessageBox::information(nullptr, "Missing parameter", "Insert crop ID");
        return false;
    }
    if (nameCropValue->text().isEmpty())
    {
        QMessageBox::information(nullptr, "Missing parameter", "Insert crop NAME");
        return false;
    }
    if (sowingDoY->isVisible())
    {
        if (sowingDoYValue->text().isEmpty() || sowingDoYValue->text().toInt() == 0)
        {
            QMessageBox::information(nullptr, "Missing parameter", "Insert sowing day of year");
            return false;
        }
        if (cycleMaxDurationValue->text().isEmpty() || cycleMaxDurationValue->text().toInt() == 0)
        {
            QMessageBox::information(nullptr, "Missing parameter", "Insert crop cycle max duration");
            return false;
        }
    }
    return true;

}

QString DialogNewCrop::getNameCrop()
{
    return nameCropValue->text();
}
