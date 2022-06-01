#include "dialogChangeAxis.h"

DialogChangeAxis::DialogChangeAxis(bool isLeftAxis_)
{
    isLeftAxis = isLeftAxis_;
    QString title;
    if (isLeftAxis)
    {
        title = "Change Left Axis";
    }
    else
    {
        title = "Change Right Axis";
    }
    this->setWindowTitle(title);
    QVBoxLayout* mainLayout = new QVBoxLayout;
    this->resize(200, 100);

    QHBoxLayout *layoutOk = new QHBoxLayout;
    QHBoxLayout *layoutEdit = new QHBoxLayout;

    QLabel minValueLabel("Minimum value:");
    minValueLabel.setBuddy(&minVal);
    minVal.setValidator(new QDoubleValidator(-999.0, 999.0, 3));

    QLabel maxValueLabel("Maximum value:");
    maxValueLabel.setBuddy(&maxVal);
    maxVal.setValidator(new QDoubleValidator(-999.0, 999.0, 3));

    layoutEdit->addWidget(&minValueLabel);
    layoutEdit->addWidget(&minVal);
    layoutEdit->addWidget(&maxValueLabel);
    layoutEdit->addWidget(&maxVal);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    layoutOk->addWidget(&buttonBox);

    mainLayout->addLayout(layoutEdit);
    mainLayout->addLayout(layoutOk);

    setLayout(mainLayout);
    exec();
}

DialogChangeAxis::~DialogChangeAxis()
{
    close();
}

void DialogChangeAxis::done(bool res)
{
    if (res) // ok
    {
        if (minVal.text().isEmpty())
        {
            QMessageBox::information(nullptr, "Missing min value", "Insert min val");
            return;
        }
        if (maxVal.text().isEmpty())
        {
            QMessageBox::information(nullptr, "Missing max value", "Insert max val");
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

float DialogChangeAxis::getMinVal() const
{
    return minVal.text().toFloat();
}

float DialogChangeAxis::getMaxVal() const
{
    return maxVal.text().toFloat();
}
