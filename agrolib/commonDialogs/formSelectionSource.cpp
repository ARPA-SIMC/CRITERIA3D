#include "formSelectionSource.h"
#include "commonConstants.h"

#include <QRadioButton>
#include <QMessageBox>
#include <QBoxLayout>


FormSelectionSource::FormSelectionSource(bool pointVisible, bool gridVisible, bool interpolationVisible)
{
    this->setWindowTitle("Select data source");
    this->resize(300, 150);

    gridButton = new QRadioButton(tr("Meteo Grid"));
    pointButton =new QRadioButton(tr("Meteo Points"));
    interpolationButton =new QRadioButton(tr("Interpolation Raster"));

    QHBoxLayout *sourceLayout = new QHBoxLayout;
    if (gridVisible)
        sourceLayout->addWidget(gridButton);
    if (pointVisible)
        sourceLayout->addWidget(pointButton);
    if (interpolationVisible)
        sourceLayout->addWidget(interpolationButton);

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
        if (!pointButton->isChecked() && !gridButton->isChecked() && !interpolationButton->isChecked())
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

    if (gridButton->isChecked())
    {
        return 2;
    }

    if (interpolationButton->isChecked())
    {
        return 3;
    }

    return NODATA;
}
