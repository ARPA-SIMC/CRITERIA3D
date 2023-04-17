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


void FormTimePeriod::setMinimumDate(QDate myDate)
{
    ui->dateTimeEditFirst->setMinimumDateTime(QDateTime(myDate, QTime(0,0,0), Qt::UTC));
    ui->dateTimeEditLast->setMinimumDateTime(QDateTime(myDate, QTime(0,0,0), Qt::UTC));
}


void FormTimePeriod::setMaximumDate(QDate myDate)
{
    ui->dateTimeEditFirst->setMaximumDateTime(QDateTime(myDate, QTime(23,0,0), Qt::UTC));
    ui->dateTimeEditLast->setMaximumDateTime(QDateTime(myDate, QTime(23,0,0), Qt::UTC));
}


void FormTimePeriod::on_buttonBox_accepted()
{
    *dateTimeFirst = ui->dateTimeEditFirst->dateTime();
    *dateTimeLast = ui->dateTimeEditLast->dateTime();
}

