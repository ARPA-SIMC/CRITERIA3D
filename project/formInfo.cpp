#include "formInfo.h"

#include <QApplication>
#include <QVBoxLayout>


FormInfo::FormInfo()
{
    this->resize(500, 200);
    this->label = new(QLabel);
    this->progressBar = new(QProgressBar);

    QVBoxLayout *mainLayout = new QVBoxLayout();
    mainLayout->addWidget(this->label);
    mainLayout->addWidget(this->progressBar);

    this->setLayout(mainLayout);
}


int FormInfo::start(QString info, int nrValues)
{
    this->label->setText(info);

    if (nrValues <= 0)
        this->progressBar->setVisible(false);
    else
    {
        this->progressBar->setMaximum(nrValues);
        this->progressBar->setValue(0);
        this->progressBar->setVisible(true);
    }

    this->show();
    qApp->processEvents();

    return std::max(1, int(nrValues / 50));
}

void FormInfo::setValue(int myValue)
{
    this->progressBar->setValue(myValue);
    qApp->processEvents();
}

void FormInfo::setText(QString myText)
{
    this->label->setText(myText);
    qApp->processEvents();
}

void FormInfo::showInfo(QString info)
{
    this->label->setText(info);
    this->progressBar->setVisible(false);

    this->show();
    qApp->processEvents();
}
