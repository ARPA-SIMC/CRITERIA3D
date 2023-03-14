#include "formSelectionSource.h"

#include <QRadioButton>
#include <QMessageBox>
#include <QBoxLayout>

FormSelectionSource::FormSelectionSource()
{
    /*
    this->setWindowTitle("Please select a source");
    QHBoxLayout* mainLayout = new QHBoxLayout;
    this->resize(200, 300);

    QHBoxLayout *horizLayout = new QHBoxLayout;

    QRadioButton *gridButton = new QRadioButton(tr("Grid"));
    QRadioButton *pointButton =new QRadioButton(tr("Meteo Points"));

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(QDialog::Accepted); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(QDialog::Rejected); });


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
    /*
    horizLayout->addWidget(gridButton);
    horizLayout->addWidget(pointButton);
    horizLayout->addWidget(&buttonBox);

    this->setLayout(horizLayout);   // Forse non serve il this.
    exec();

    //connect(gridButton, &QRadioButton::clicked, this, [=](){ sourceChange(); });
    //connect(gridButton, &QRadioButton::clicked, this, [=](){ sourceChange(); });
    */
}


/*
void formSelectionSource::sourceChange()
{
    if (pointsButton.isChecked())
    {
        isMeteoGrid = false;
        myProject.clima->setDb(myProject.meteoPointsDbHandler->getDb());    // Al posto di clima cosa devo mettere?
    }
    else if (gridButton.isChecked())
    {
        isMeteoGrid = true;
        myProject.clima->setDb(myProject.meteoGridDbHandler->db());
    }
}



FormSelection::~FormSelection()
{
    close();
}

void FormSelection::done(int res)
{
    if (res == QDialog::Accepted) // ok
    {
        if (cmbStringList->currentText() == "")
        {
            QMessageBox::information(nullptr, "Missing selection", "Select");
            return;
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

QString FormSelection::getSelection()
{
    return cmbStringList->currentText();
}

int FormSelection::getSelectionId()
{
    return cmbStringList->currentIndex();
}

*/
