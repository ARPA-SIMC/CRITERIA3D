#include "formInfo.h"
#include "ui_formInfo.h"


FormInfo::FormInfo(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::formInfo)
{
    ui->setupUi(this);
}

FormInfo::~FormInfo()
{
    delete ui;
}


int FormInfo::start(QString info, int nrValues)
{
    this->ui->label->setText(info);

    if (nrValues <= 0)
        this->ui->progressBar->setVisible(false);
    else
    {
        this->ui->progressBar->setMaximum(nrValues);
        this->ui->progressBar->setValue(0);
        this->ui->progressBar->setVisible(true);
    }

    this->show();
    qApp->processEvents();

    return std::max(1, int(nrValues / 50));
}

void FormInfo::setValue(int myValue)
{
    this->ui->progressBar->setValue(myValue);
    qApp->processEvents();
}

void FormInfo::setText(QString myText)
{
    this->ui->label->setText(myText);
    qApp->processEvents();
}

void FormInfo::showInfo(QString info)
{
    this->ui->label->setText(info);
    this->ui->progressBar->setVisible(false);

    this->show();
    qApp->processEvents();
}
