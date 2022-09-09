#include "formSelection.h"


FormSelection::FormSelection(QStringList stringList_)
: stringList(stringList_)
{

    this->setWindowTitle("Select");
    QVBoxLayout* mainLayout = new QVBoxLayout;
    this->resize(200, 100);

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




