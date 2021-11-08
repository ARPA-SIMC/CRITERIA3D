#include "formTimePeriod.h"
#include "ui_formTimePeriod.h"
#include <QDateTime>


FormTimePeriod::FormTimePeriod(QDateTime *timeIni, QDateTime *timeFin, QWidget * parent) :
    QDialog(parent),
    ui(new Ui::frmTimePeriod)
{
    if (timeIni == nullptr || timeFin == nullptr) return;

    dateTimeFirst = timeIni;
    dateTimeLast = timeFin;

    ui->setupUi(this);

    ui->dateTimeEditFirst->setDateTime(*dateTimeFirst);
    ui->dateTimeEditLast->setDateTime(*dateTimeLast);
}

FormTimePeriod::~FormTimePeriod()
{
    delete ui;
}


void FormTimePeriod::setMinimumDate(QDate myDate)
{
    ui->dateTimeEditFirst->setMinimumDate(myDate);
    ui->dateTimeEditLast->setMinimumDate(myDate);
}


void FormTimePeriod::setMaximumDate(QDate myDate)
{
    ui->dateTimeEditFirst->setMaximumDate(myDate);
    ui->dateTimeEditLast->setMaximumDate(myDate);
}


void FormTimePeriod::on_buttonBox_accepted()
{
    *dateTimeFirst = ui->dateTimeEditFirst->dateTime();
    *dateTimeLast = ui->dateTimeEditLast->dateTime();
}

