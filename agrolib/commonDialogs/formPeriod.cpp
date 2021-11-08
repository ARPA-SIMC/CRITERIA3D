#include "formPeriod.h"
#include "ui_FormPeriod.h"
#include <QDateTime>


FormPeriod::FormPeriod(QDateTime *timeIni, QDateTime *timeFin, QWidget * parent) :
    QDialog(parent),
    ui(new Ui::FormPeriod)
{
    if (timeIni == nullptr || timeFin == nullptr) return;

    dateTimeFirst = timeIni;
    dateTimeLast = timeFin;

    ui->setupUi(this);

    ui->dateTimeEditFirst->setDateTime(*dateTimeFirst);
    ui->dateTimeEditLast->setDateTime(*dateTimeLast);
}

FormPeriod::~FormPeriod()
{
    delete ui;
}


void FormPeriod::setMinimumDate(QDate myDate)
{
    ui->dateTimeEditFirst->setMinimumDate(myDate);
    ui->dateTimeEditLast->setMinimumDate(myDate);
}


void FormPeriod::setMaximumDate(QDate myDate)
{
    ui->dateTimeEditFirst->setMaximumDate(myDate);
    ui->dateTimeEditLast->setMaximumDate(myDate);
}


void FormPeriod::on_buttonBox_accepted()
{
    *dateTimeFirst = ui->dateTimeEditFirst->dateTime();
    *dateTimeLast = ui->dateTimeEditLast->dateTime();
}

