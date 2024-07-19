#include "dialogChangeAxis.h"

DialogChangeAxis::DialogChangeAxis(int nrAxis, bool isDate)
{
    QString title;
    if (nrAxis == 0)
    {
        title = "Change X Axis";
    }
    else if (nrAxis == 1)
    {
        title = "Change Left Axis";
    }
    else if (nrAxis == 2)
    {
        title = "Change Right Axis";
    }
    this->setWindowTitle(title);

    QVBoxLayout* mainLayout = new QVBoxLayout;
    this->resize(200, 100);

    QHBoxLayout *layoutOk = new QHBoxLayout;
    QHBoxLayout *layoutEdit = new QHBoxLayout;

    QLabel minValueLabel("Minimum value:");
    layoutEdit->addWidget(&minValueLabel);
    if (isDate)
    {
        minValueLabel.setBuddy(&minDate);
        layoutEdit->addWidget(&minDate);
    }
    else
    {
        minValueLabel.setBuddy(&minVal);
        minVal.setValidator(new QDoubleValidator(-9999.0, 9999.0, 3));
        layoutEdit->addWidget(&minVal);
    }

    QLabel maxValueLabel("Maximum value:");
    layoutEdit->addWidget(&maxValueLabel);
    if (isDate)
    {
         minValueLabel.setBuddy(&maxDate);
        layoutEdit->addWidget(&maxDate);
    }
    else
    {
        maxValueLabel.setBuddy(&maxVal);
        maxVal.setValidator(new QDoubleValidator(-9999.0, 9999.0, 3));
        layoutEdit->addWidget(&maxVal);
    }

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

QDate DialogChangeAxis::getMinDate() const
{
    return minDate.date();
}

QDate DialogChangeAxis::getMaxDate() const
{
    return maxDate.date();
}
