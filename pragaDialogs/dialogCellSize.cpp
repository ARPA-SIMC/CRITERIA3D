#include "dialogCellSize.h"

DialogCellSize::DialogCellSize(int defaultCellSize)
{

    this->setWindowTitle("Insert Cell size");
    QVBoxLayout* mainLayout = new QVBoxLayout;
    this->resize(250, 100);

    QHBoxLayout *layoutOk = new QHBoxLayout;
    QHBoxLayout *cellSizeLayout = new QHBoxLayout;

    QLabel cellSizeLabel("Cell size:");
    cellSizeLabel.setBuddy(&cellSizeEdit);
    cellSizeEdit.setValidator(new QDoubleValidator(0.0, 9999.0,1));
    cellSizeEdit.setText(QString::number(defaultCellSize));

    cellSizeLayout->addWidget(&cellSizeLabel);
    cellSizeLayout->addWidget(&cellSizeEdit);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    layoutOk->addWidget(&buttonBox);

    mainLayout->addLayout(cellSizeLayout);
    mainLayout->addLayout(layoutOk);

    setLayout(mainLayout);
    exec();
}

DialogCellSize::~DialogCellSize()
{
    close();
}

void DialogCellSize::done(bool res)
{
    if (res) // ok
    {
        if (cellSizeEdit.text().isEmpty())
        {
            QMessageBox::information(nullptr, "Missing cell size value", "Insert cell size");
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

double DialogCellSize::getCellSize() const
{
    return cellSizeEdit.text().toFloat();
}
