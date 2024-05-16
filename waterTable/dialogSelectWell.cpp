#include "dialogSelectWell.h"

DialogSelectWell::DialogSelectWell(QList<QString> wellsId)
    :wellsId(wellsId)
{
    setWindowTitle("Select well");
    QVBoxLayout* mainLayout = new QVBoxLayout;
    this->resize(250, 100);


    QHBoxLayout *layoutOk = new QHBoxLayout;
    QHBoxLayout *wellsLayout = new QHBoxLayout;

    buttonWells = new QComboBox;
    buttonWells->addItems(wellsId);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    wellsLayout->addWidget(buttonWells);
    layoutOk->addWidget(&buttonBox);

    mainLayout->addLayout(wellsLayout);
    mainLayout->addLayout(layoutOk);

    setLayout(mainLayout);
    exec();
}

void DialogSelectWell::done(bool res)
{
    if (res) // ok
    {
        if (buttonWells->currentText() == "")
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

QString DialogSelectWell::getIdSelected() const
{
    return buttonWells->currentText();
}
