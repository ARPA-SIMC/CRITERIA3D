#include "dialogShiftData.h"
#include "meteo.h"

DialogShiftData::DialogShiftData(QDate myDate)
{

    this->setWindowTitle("Shift Data");
    QVBoxLayout* mainLayout = new QVBoxLayout;
    this->resize(250, 100);

    QHBoxLayout *layoutOk = new QHBoxLayout;
    QHBoxLayout *dateLayout = new QHBoxLayout;
    QVBoxLayout *variableLayout = new QVBoxLayout;
    QHBoxLayout *shiftLayout = new QHBoxLayout;

    QLabel *subTitleLabel = new QLabel(tr("Selected points"));
    mainLayout->addWidget(subTitleLabel);
    QLabel *dateFromLabel = new QLabel(tr("From"));
    dateLayout->addWidget(dateFromLabel);
    dateLayout->addWidget(&dateFrom);
    QLabel *dateToLabel = new QLabel(tr("To"));
    dateLayout->addWidget(dateToLabel);
    dateLayout->addWidget(&dateTo);

    dateFrom.setDate(myDate);
    dateTo.setDate(myDate);

    QLabel *variableLabel = new QLabel(tr("Daily Variable: "));
    std::map<meteoVariable, std::string>::const_iterator it;
    for(it = MapDailyMeteoVarToString.begin(); it != MapDailyMeteoVarToString.end(); ++it)
    {
        variable.addItem(QString::fromStdString(it->second));
    }
    //myVar = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, variable.currentText().toStdString());
    variable.setSizeAdjustPolicy(QComboBox::AdjustToContents);
    variable.setMaximumWidth(150);
    variableLayout->addWidget(variableLabel);
    variableLayout->addWidget(&variable);

    QLabel shiftLabel("Shift:");
    shiftLabel.setBuddy(&shiftEdit);
    shiftEdit.setValidator(new QIntValidator(1.0, 100.0));
    shiftEdit.setText(QString::number(0));

    shiftLayout->addWidget(&shiftLabel);
    shiftLayout->addWidget(&shiftEdit);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    layoutOk->addWidget(&buttonBox);

    mainLayout->addLayout(dateLayout);
    mainLayout->addLayout(variableLayout);
    mainLayout->addLayout(shiftLayout);
    mainLayout->addLayout(layoutOk);

    setLayout(mainLayout);
    exec();
}

DialogShiftData::~DialogShiftData()
{
    close();
}

void DialogShiftData::done(bool res)
{
    if (res) // ok
    {
        if (shiftEdit.text().isEmpty())
        {
            QMessageBox::information(nullptr, "Missing shift value", "Insert shift value");
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

int DialogShiftData::getShift() const
{
    return shiftEdit.text().toInt();
}

