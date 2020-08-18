#include "dialogSelectVar.h"

DialogSelectVar::DialogSelectVar(QStringList allVar, QStringList selectedVar)
: allVar(allVar), selectedVar(selectedVar)
{
    setWindowTitle("Add or remove variables");
    QVBoxLayout *mainLayout = new QVBoxLayout();
    QHBoxLayout *headerLayout = new QHBoxLayout;
    QHBoxLayout *variableLayout = new QHBoxLayout;
    QVBoxLayout *arrowLayout = new QVBoxLayout();
    QHBoxLayout *layoutOk = new QHBoxLayout;
    QDialogButtonBox buttonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);

    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });

    layoutOk->addWidget(&buttonBox);
    listAllVar = new QListWidget;
    listSelectedVar = new QListWidget;

    QLabel *allHeader = new QLabel("All variables");
    QLabel *selectedHeader = new QLabel("Selected variables");
    addButton = new QPushButton(tr("➡"));
    deleteButton = new QPushButton(tr("⬅"));
    addButton->setEnabled(false);
    deleteButton->setEnabled(false);
    arrowLayout->addWidget(addButton);
    arrowLayout->addWidget(deleteButton);
    listAllVar->addItems(allVar);
    listSelectedVar->addItems(selectedVar);
    variableLayout->addWidget(listAllVar);
    variableLayout->addLayout(arrowLayout);
    variableLayout->addWidget(listSelectedVar);

    headerLayout->addWidget(allHeader);
    headerLayout->addSpacing(listAllVar->width());
    headerLayout->addWidget(selectedHeader);
    mainLayout->addLayout(headerLayout);
    mainLayout->addLayout(variableLayout);
    mainLayout->addLayout(layoutOk);
    setLayout(mainLayout);

    connect(listAllVar, &QListWidget::itemClicked, [=](QListWidgetItem* item){ this->variableAllClicked(item); });
    connect(listSelectedVar, &QListWidget::itemClicked, [=](QListWidgetItem* item){ this->variableSelClicked(item); });
    connect(addButton, &QPushButton::clicked, [=](){ addVar(); });
    connect(deleteButton, &QPushButton::clicked, [=](){ deleteVar(); });
    connect(&buttonBox, &QDialogButtonBox::accepted, [=](){ this->done(true); });
    connect(&buttonBox, &QDialogButtonBox::rejected, [=](){ this->done(false); });


    show();
    exec();

}

void DialogSelectVar::variableAllClicked(QListWidgetItem* item)
{
    Q_UNUSED(item);

    addButton->setEnabled(true);
    deleteButton->setEnabled(false);
    listSelectedVar->clearSelection();
}

void DialogSelectVar::variableSelClicked(QListWidgetItem* item)
{
    Q_UNUSED(item);

    addButton->setEnabled(false);
    deleteButton->setEnabled(true);
    listAllVar->clearSelection();
}

void DialogSelectVar::addVar()
{
    QListWidgetItem *item = listAllVar->currentItem();
    int row = listAllVar->currentRow();
    listAllVar->takeItem(row);
    listSelectedVar->addItem(item);
}

void DialogSelectVar::deleteVar()
{
    QListWidgetItem *item = listSelectedVar->currentItem();
    int row = listSelectedVar->currentRow();
    listSelectedVar->takeItem(row);
    listAllVar->addItem(item);
}

QStringList DialogSelectVar::getSelectedVariables()
{
    QStringList variableSelected;
    for(int i = 0; i < listSelectedVar->count(); ++i)
    {
        QString var = listSelectedVar->item(i)->text();
        variableSelected.append(var);
    }
    return variableSelected;
}


