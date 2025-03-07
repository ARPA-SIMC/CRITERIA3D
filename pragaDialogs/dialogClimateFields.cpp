#include "dialogClimateFields.h"
#include "climate.h"

QString DialogClimateFields::getSelected() const
{
    return climaSelected;
}

meteoVariable DialogClimateFields::getVar() const
{
    return var;
}

QString DialogClimateFields::getIndexSelected() const
{
    return indexSelected;
}

bool DialogClimateFields::getIsShowClicked() const
{
    return isShowClicked;
}

DialogClimateFields::DialogClimateFields(QList<QString> climateDbElab, QList<QString> climateDbVarList)
: climateDbElab(climateDbElab), climateDbVarList(climateDbVarList)
{
    setWindowTitle("Climate Variables");

    showButton.setText("Show");
    showButton.setEnabled(false);

    deleteButton.setText("Delete");
    deleteButton.setEnabled(false);

    buttonLayout.addWidget(&showButton);
    buttonLayout.addWidget(&deleteButton);

    listVariable.addItems(climateDbVarList);
    variableLayout.addWidget(&listVariable);
    indexLayout.addWidget(&listIndex);

    connect(&listVariable, &QListWidget::itemClicked, [=](QListWidgetItem* item){ this->variableClicked(item); });

    connect(&showButton, &QPushButton::clicked, [=](){ showClicked(); });
    connect(&deleteButton, &QPushButton::clicked, [=](){ deleteClicked(); });

    mainLayout.addLayout(&variableLayout);

    elabW.setLayout(&elabLayout);
    mainLayout.addWidget(&elabW);
    indexW.setLayout(&indexLayout);
    mainLayout.addWidget(&indexW);
    mainLayout.addLayout(&buttonLayout);

    elabW.setVisible(false);
    indexW.setVisible(false);


    setLayout(&mainLayout);
    mainLayout.setSizeConstraint(QLayout::SetFixedSize);

    show();
    exec();
}

void DialogClimateFields::variableClicked(QListWidgetItem* item)
{
    elabW.setVisible(true);
    indexW.setVisible(false);
    showButton.setEnabled(false);
    deleteButton.setEnabled(false);

    listElab.clear();
    QList<QString> elabVarSelected;

    std::string variable = item->text().toStdString();
    var = getKeyMeteoVarMeteoMap(MapDailyMeteoVarToString, variable);

    for (int i=0; i < climateDbElab.size(); i++)
    {
        QString elab = climateDbElab[i];
        QList<QString> words = elab.split('_');
        QString var = words[1];
        if (var == item->text())
        {
            elabVarSelected.append(elab);
        }
    }

    listElab.addItems(elabVarSelected);
    elabLayout.addWidget(&listElab);
    connect(&listElab, &QListWidget::itemClicked, [=](QListWidgetItem* item){ this->elabClicked(item); });

}

void DialogClimateFields::elabClicked(QListWidgetItem* item)
{
    climaSelected = item->text();
    listIndex.clear();
    showButton.setEnabled(false);
    deleteButton.setEnabled(true);
    QList<QString> listIndexSelected;

    int n = getNumberClimateIndexFromElab(climaSelected);
    if (n == 1)
    {
        indexSelected = "1";
        indexW.setVisible(false);
        showButton.setEnabled(true);
        deleteButton.setEnabled(true);
    }
    else
    {
        indexW.setVisible(true);
        for (int i=1; i <= n; i++)
        {
            if (n==4)
            {
                switch(i) {
                    case 1 : listIndexSelected.append("MAM");
                             break;
                    case 2 : listIndexSelected.append("JJA");
                             break;
                    case 3 : listIndexSelected.append("SON");
                             break;
                    case 4 : listIndexSelected.append("DJF");
                             break;
                }

            }
            else
            {
                listIndexSelected.append(QString::number(i));
            }

        }

        listIndex.addItems(listIndexSelected);
        indexLayout.addWidget(&listIndex);
        connect(&listIndex, &QListWidget::itemClicked, [=](QListWidgetItem* item){ this->indexClicked(item); });
    }

}


void DialogClimateFields::indexClicked(QListWidgetItem* item)
{
    indexSelected = item->text();
    if (indexSelected == "MAM")
    {
        indexSelected = "1";
    }
    else if (indexSelected == "JJA")
    {
        indexSelected = "2";
    }
    else if (indexSelected == "SON")
    {
        indexSelected = "3";
    }
    else if (indexSelected == "DJF")
    {
        indexSelected = "4";
    }

    showButton.setEnabled(true);
    deleteButton.setEnabled(false);
}

void DialogClimateFields::showClicked()
{
    isShowClicked = true;
    QDialog::done(QDialog::Accepted);
}

void DialogClimateFields::deleteClicked()
{
    isShowClicked = false;
    if (QMessageBox::question(this, "Warning", "Are you sure?", QMessageBox::Yes|QMessageBox::No) == QMessageBox::Yes)
        QDialog::done(QDialog::Accepted);
}
