#include "formText.h"

FormText::FormText(QString title)
{
    this->setWindowTitle(title);
    this->resize(250, 100);

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(&buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    QVBoxLayout* mainLayout = new QVBoxLayout;
    mainLayout->addWidget(&textEdit);
    mainLayout->addWidget(&buttonBox);

    setLayout(mainLayout);
    exec();
}


QString FormText::getText() const
{
    return textEdit.text();
}
