#include "formText.h"

FormText::FormText(QString title)
{
    this->setWindowTitle(title);
    QVBoxLayout* mainLayout = new QVBoxLayout;
    this->resize(250, 100);

    QHBoxLayout *layoutOk = new QHBoxLayout;
    QHBoxLayout *textLayout = new QHBoxLayout;

    textLayout->addWidget(&textEdit);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ done(QDialog::Accepted); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ done(QDialog::Rejected); });

    layoutOk->addWidget(&buttonBox);

    mainLayout->addLayout(textLayout);
    mainLayout->addLayout(layoutOk);

    setLayout(mainLayout);
    exec();
}


void FormText::done(int res)
{
    if (res == QDialog::Accepted) // ok
    {
        QDialog::done(QDialog::Accepted);
        return;
    }
    else    // cancel, close or exc was pressed
    {
        QDialog::done(QDialog::Rejected);
        return;
    }
}


QString FormText::getText() const
{
    return textEdit.text();
}
