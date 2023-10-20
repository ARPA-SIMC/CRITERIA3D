#include "formSelection.h"


FormSelection::FormSelection(QList<QString> stringList_, QString title)
: stringList(stringList_)
{

    this->setWindowTitle(title);
    QVBoxLayout* mainLayout = new QVBoxLayout;
    this->resize(250, 100);

    QHBoxLayout *layoutOk = new QHBoxLayout;
    QHBoxLayout *datasetLayout = new QHBoxLayout;

    cmbStringList = new QComboBox;
    cmbStringList->addItems(stringList);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(QDialog::Accepted); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(QDialog::Rejected); });

    datasetLayout->addWidget(cmbStringList);
    layoutOk->addWidget(&buttonBox);

    mainLayout->addLayout(datasetLayout);
    mainLayout->addLayout(layoutOk);

    setLayout(mainLayout);
    exec();
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

