#include "formSelectionSource.h"

#include <QRadioButton>
#include <QMessageBox>
#include <QBoxLayout>

FormSelectionSource::FormSelectionSource()
{

    this->setWindowTitle("Please select a source");
    QHBoxLayout* mainLayout = new QHBoxLayout;
    this->resize(200, 300);

    QHBoxLayout *horizLayout = new QHBoxLayout;

    gridButton = new QRadioButton(tr("Grid"));
    pointButton =new QRadioButton(tr("Meteo Points"));

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(QDialog::Accepted); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(QDialog::Rejected); });

    connect(pointButton, &QRadioButton::clicked, [=](){ this->getSourceSelectionId(); });
    connect(gridButton, &QRadioButton::clicked, [=](){ this->getSourceSelectionId(); });


    /* Questo sarebbe carino ma devi passargli le informazioni e non saprei da dove (da fare forse nella main):
    if (isMeteoPointLoaded)
    {
        pointsButton.setEnabled(true);
        if (!isMeteoGridLoaded)
        {
            pointsButton.setChecked(true);
        }
    }
    else
    {
        pointsButton.setEnabled(false);
    }
    if (isMeteoGridLoaded)
    {
        gridButton.setEnabled(true);
        if (!isMeteoPointLoaded)
        {
            gridButton.setChecked(true);
        }
    }
    else
    {
        gridButton.setEnabled(false);
    }
    */

    /*
    QGridLayout *SelectionLayout = new QGridLayout;

    SelectionLayout->addWidget(gridSelection, 0, 0);
    SelectionLayout->addWidget(pointSelection,0, 1);
    */

    horizLayout->addWidget(gridButton);
    horizLayout->addWidget(pointButton);
    horizLayout->addWidget(&buttonBox);

    setLayout(horizLayout);   // Forse non serve il this.
    show();
    exec();

}


void FormSelectionSource::done(int res)
{
    if (res == QDialog::Accepted) // ok
    {
        if (!pointButton->isChecked() && !gridButton->isChecked())     //(cmbStringList->currentText() == "")
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
}

