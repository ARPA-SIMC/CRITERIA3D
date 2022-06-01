#include "dialogNewSoil.h"
#include <QIntValidator>

DialogNewSoil::DialogNewSoil()
{
    setWindowTitle("New Soil");
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QGridLayout *layoutSoil = new QGridLayout();
    QHBoxLayout *layoutOk = new QHBoxLayout();

    QLabel *idSoilLabel = new QLabel(tr("Enter soil ID (only numerical): "));
    idSoilValue = new QLineEdit();

    QIntValidator* validatorID = new QIntValidator();
    idSoilValue->setValidator(validatorID);

    QLabel *codeSoilLabel = new QLabel(tr("Enter soil code (string without spaces): "));
    codeSoilValue = new QLineEdit();

    QLabel *nameSoilLabel = new QLabel(tr("Enter soil name: "));
    nameSoilValue = new QLineEdit();

    QLabel *infoSoilLabel = new QLabel(tr("Enter additional info (optional): "));
    infoSoilValue = new QLineEdit();

    layoutSoil->addWidget(idSoilLabel, 0 , 0);
    layoutSoil->addWidget(idSoilValue, 0 , 1);
    layoutSoil->addWidget(codeSoilLabel, 1 , 0);
    layoutSoil->addWidget(codeSoilValue, 1 , 1);
    layoutSoil->addWidget(nameSoilLabel, 2 , 0);
    layoutSoil->addWidget(nameSoilValue, 2 , 1);
    layoutSoil->addWidget(infoSoilLabel, 3 , 0);
    layoutSoil->addWidget(infoSoilValue, 3 , 1);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    layoutOk->addWidget(&buttonBox);


    mainLayout->addLayout(layoutSoil);
    mainLayout->addLayout(layoutOk);

    setLayout(mainLayout);

    show();
    exec();
}

int DialogNewSoil::getIdSoilValue()
{
    return idSoilValue->text().toInt();
}

QString DialogNewSoil::getCodeSoilValue()
{
    return codeSoilValue->text();
}

QString DialogNewSoil::getNameSoilValue()
{
    return nameSoilValue->text();
}

QString DialogNewSoil::getInfoSoilValue()
{
    return infoSoilValue->text();
}

void DialogNewSoil::done(bool res)
{
    if(res)  // ok was pressed
    {
        if (idSoilValue->text().isEmpty())
        {
            QMessageBox::information(nullptr, "Error!", "Enter soil ID");
            return;
        }
        if (codeSoilValue->text().isEmpty())
        {
            QMessageBox::information(nullptr, "Error!", "Enter soil Code");
            return;
        }
        if (nameSoilValue->text().isEmpty())
        {
            QMessageBox::information(nullptr, "Error!", "Enter soil Name");
            return;
        }
        else
        {
            // remove white spaces
            QString code = codeSoilValue->text();
            code = code.simplified();
            code.replace(" ","");
            codeSoilValue->setText(code);
            QDialog::done(QDialog::Accepted);
            return;
        }
    }
    else    // cancel or close was pressed
    {
        QDialog::done(QDialog::Rejected);
        return;
    }
}
