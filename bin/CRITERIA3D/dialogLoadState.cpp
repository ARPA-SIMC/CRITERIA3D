#include "dialogLoadState.h"

DialogLoadState::DialogLoadState(QList<QString> allStates)
{
    setWindowTitle("Select state");
    this->setMinimumWidth(250);
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QHBoxLayout *layoutOk = new QHBoxLayout;

    for (int i = 0; i < allStates.size(); i++)
    {
        stateListComboBox.addItem(allStates[i]);
    }

    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    layoutOk->addWidget(&buttonBox);
    mainLayout->addWidget(&stateListComboBox);
    mainLayout->addLayout(layoutOk);
    setLayout(mainLayout);

    show();
    exec();
}

QString DialogLoadState::getSelectedState()
{
    return stateListComboBox.currentText();
}
