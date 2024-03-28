#include "dialogVariableToSum.h"

DialogVariableToSum::DialogVariableToSum(QList<QString> variableList, QList<QString> varAlreadyChecked)
: variableList(variableList), varAlreadyChecked(varAlreadyChecked)
{
    setWindowTitle("Choose variable to sum");
    this->resize(400, 200);
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QVBoxLayout *variableLayout = new QVBoxLayout;
    QHBoxLayout *layoutOk = new QHBoxLayout;
    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    layoutOk->addWidget(&buttonBox);
    for (int i = 0; i<variableList.size(); i++)
    {
        QCheckBox* checkbox = new QCheckBox(variableList[i], this);
        checkList.append(checkbox);
        if (varAlreadyChecked.contains(checkbox->text()))
        {
            checkbox->setChecked(true);
        }
        variableLayout->addWidget(checkbox);
    }

    mainLayout->addLayout(variableLayout);
    mainLayout->addLayout(layoutOk);
    setLayout(mainLayout);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    show();
    exec();

}

void DialogVariableToSum::done(bool res)
{
    if (res)
    {
        foreach (QCheckBox *checkBox, checkList)
        {
            if (checkBox->isChecked())
            {
                selectedVariable.append(checkBox->text());
            }
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

QList<QString> DialogVariableToSum::getSelectedVariable()
{
    return selectedVariable;
}


