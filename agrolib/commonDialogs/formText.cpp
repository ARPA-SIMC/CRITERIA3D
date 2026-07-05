#include "formText.h"
#include <QDialogButtonBox>
#include <QVBoxLayout>


FormText::FormText(const QString &title, const QString &text)
{
    setWindowTitle(title);
    resize(250, 100);

    textEdit.setText(text);
    QDialogButtonBox* buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);

    QVBoxLayout* mainLayout = new QVBoxLayout;
    mainLayout->addWidget(&textEdit);
    mainLayout->addWidget(buttonBox);
    setLayout(mainLayout);
}
