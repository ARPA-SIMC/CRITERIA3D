#include "formPeriod.h"
#include "ui_formPeriod.h"


formPeriod::formPeriod(QDateTime *timeIni, QDateTime *timeFin, QWidget * parent) :
    QDialog(parent),
    ui(new Ui::formPeriod)
{
    if (timeIni == nullptr || timeFin == nullptr) return;

    dateTimeFirst = timeIni;
    dateTimeLast = timeFin;

    ui->setupUi(this);

    ui->dateTimeEditFirst->setDateTime(*dateTimeFirst);
    ui->dateTimeEditLast->setDateTime(*dateTimeLast);
}

formPeriod::~formPeriod()
{
    delete ui;
}


void formPeriod::setMinimumDate(QDate minDate)
{
    ui->dateTimeEditFirst->setMinimumDate(minDate);
    ui->dateTimeEditLast->setMinimumDate(minDate);
}


void formPeriod::setMaximumDate(QDate maxDate)
{
    ui->dateTimeEditFirst->setMaximumDate(maxDate);
    ui->dateTimeEditLast->setMaximumDate(maxDate);
}


void formPeriod::on_buttonBox_accepted()
{
    *dateTimeFirst = ui->dateTimeEditFirst->dateTime();
    *dateTimeLast = ui->dateTimeEditLast->dateTime();
}

