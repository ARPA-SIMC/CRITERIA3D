#include "formSelectionSource.h"
#include "commonConstants.h"

#include <QRadioButton>
#include <QMessageBox>
#include <QBoxLayout>

FormSelectionSource::FormSelectionSource()
{

    this->setWindowTitle("Select data source");
    this->resize(300, 150);

    gridButton = new QRadioButton(tr("Meteo Grid"));
    pointButton =new QRadioButton(tr("Meteo Points"));

    QHBoxLayout *sourceLayout = new QHBoxLayout;
    sourceLayout->addWidget(gridButton);
    sourceLayout->addWidget(pointButton);

    QGroupBox *sourceGroupBox = new QGroupBox("Source");
    sourceGroupBox->setLayout(sourceLayout);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(QDialog::Accepted); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(QDialog::Rejected); });

    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(sourceGroupBox);
    mainLayout->addWidget(&buttonBox);

    setLayout(mainLayout);
    exec();
}


void FormSelectionSource::done(int res)
{
    if (res == QDialog::Accepted) // ok
    {
        if (!pointButton->isChecked() && !gridButton->isChecked())
        {
            QMessageBox::information(nullptr, "Missing source selection.", "Please choose a data source.");
            return;
        }
        QDialog::done(QDialog::Accepted);
        return;
    }
    else    // cancel, close or esc was pressed
    {
        QDialog::done(QDialog::Rejected);
        return;
    }
}


int FormSelectionSource::getSourceSelectionId()
{
    if (pointButton->isChecked())
    {
       return 1;
    }
    else if (gridButton->isChecked())
    {
        return 2;
    }
    else
    {
        return NODATA;
    }
}

