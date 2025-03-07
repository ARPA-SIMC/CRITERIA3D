#include "saveClimaLayout.h"

bool compareClimateElab(const QString &el1, const QString &el2)
{
    if (! el1.contains('_') || ! el2.contains('_'))
        return false;

    QString var1 = el1.split("_")[1];
    QString var2 = el2.split("_")[1];

    if (var1 != var2)
        return var1.compare(var2) < 0;

    if (el1.size() < 9 || el2.size() < 9)
        return false;

    int period1 = el1.mid(5, 4).toInt() - el1.left(4).toInt();
    int period2 = el2.mid(5, 4).toInt() - el2.left(4).toInt();
    return (period1 > period2);
}


SaveClimaLayout::SaveClimaLayout()
{
    listLayout.addWidget(&listView);

    saveList.setText("Save list");
    loadList.setText("Load list");
    saveButtonLayout.addWidget(&saveList);
    saveButtonLayout.addWidget(&loadList);

    connect(&saveList, &QPushButton::clicked, [=](){ this->saveElabList(); });
    connect(&loadList, &QPushButton::clicked, [=](){ this->loadElabList(); });

    mainLayout.addLayout(&listLayout);
    mainLayout.addLayout(&saveButtonLayout);

    setLayout(&mainLayout);

}


void SaveClimaLayout::addElab()
{
    QString elabAdded = firstYear + "-" + lastYear + "_" + variable.remove("_") + "_" + period;
    if (period == "Generic")
    {
        elabAdded = elabAdded + "_" + genericPeriodStartDay + "of" + genericPeriodStartMonth
                    + "-" + genericPeriodEndDay + "of" + genericPeriodEndMonth;
        if (genericNYear != "0")
        {
            elabAdded = elabAdded + "-+" + genericNYear + "y";
        }
    }
    if (!secondElab.isEmpty() && secondElab != "None" && secondElab != "No elaboration available")
    {
        elabAdded = elabAdded + "_" + secondElab;

        if (!elab2Param.isEmpty())
        {
            elabAdded = elabAdded + "_" + elab2Param;
        }
    }
    elabAdded = elabAdded + "_" + elab;
    if (!elab1Param.isEmpty())
    {
        elabAdded = elabAdded + "_" + elab1Param;
    }
    else if(!elab1ParamFromdB.isEmpty())
    {
        elabAdded = elabAdded + "_|" + elab1ParamFromdB + "||";
    }

    if (list.contains(elabAdded)!= 0)
    {
        return;
    }

    list.append(elabAdded);
    std::sort(list.begin(), list.end(), compareClimateElab);

    listView.clear();
    listView.addItems(list);
}


void SaveClimaLayout::deleteRaw()
{
    if (listView.selectedItems().size()==0)
    {
        return;
    }
    list.removeAt(listView.currentIndex().row());
    listView.takeItem(listView.currentIndex().row());
}

void SaveClimaLayout::deleteAll()
{
    list.clear();
    listView.clear();
}

void SaveClimaLayout::saveElabList()
{
    QString fileName = QFileDialog::getSaveFileName(this, tr("Save elaborations file"), "", tr("(*.txt)"));
    if (fileName == "") return;

    QFile elabList(fileName);
    if (elabList.open(QFile::WriteOnly | QFile::Text))
    {
        QTextStream s(&elabList);
        for (int i = 0; i < list.size(); ++i)
          s << list[i] << '\n';
    }
    else
    {
        qDebug() << "error opening output file\n";
        return;
    }
    elabList.close();
}


void SaveClimaLayout::loadElabList()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open elaborations file"), "", tr("(*.txt)"));
    if (fileName.isEmpty())
        return;

    QFile elabList(fileName);
    if (! elabList.open(QFile::ReadOnly | QFile::Text))
        return;

    QTextStream sIn(&elabList);
    while (! sIn.atEnd())
    {
        QString line = sIn.readLine();
        if (!line.isEmpty() && !list.contains(line))
        {
            list << line;
        }
    }

    std::sort(list.begin(), list.end(), compareClimateElab);

    listView.clear();
    listView.addItems(list);
}


QList<QString> SaveClimaLayout::getList() const
{
    return list;
}

void SaveClimaLayout::setList(const QList<QString> &valueList)
{
    list = valueList;
    listView.addItems(list);
}

QString SaveClimaLayout::getElab1ParamFromdB() const
{
    return elab1ParamFromdB;
}

void SaveClimaLayout::setElab1ParamFromdB(const QString &valueStr)
{
    elab1ParamFromdB = valueStr;
}

QString SaveClimaLayout::getFirstYear() const
{
    return firstYear;
}

void SaveClimaLayout::setFirstYear(const QString &valueStr)
{
    firstYear = valueStr;
}

QString SaveClimaLayout::getLastYear() const
{
    return lastYear;
}

void SaveClimaLayout::setLastYear(const QString &valueStr)
{
    lastYear = valueStr;
}

QString SaveClimaLayout::getVariable() const
{
    return variable;
}

void SaveClimaLayout::setVariable(const QString &valueStr)
{
    variable = valueStr;
}

QString SaveClimaLayout::getPeriod() const
{
    return period;
}

void SaveClimaLayout::setPeriod(const QString &valueStr)
{
    period = valueStr;
}

QString SaveClimaLayout::getGenericPeriodStartDay() const
{
    return genericPeriodStartDay;
}

void SaveClimaLayout::setGenericPeriodStartDay(const QString &value)
{
    genericPeriodStartDay = value;
}

QString SaveClimaLayout::getGenericPeriodStartMonth() const
{
    return genericPeriodStartMonth;
}

void SaveClimaLayout::setGenericPeriodStartMonth(const QString &valueStr)
{
    genericPeriodStartMonth = valueStr;
}

QString SaveClimaLayout::getGenericPeriodEndDay() const
{
    return genericPeriodEndDay;
}

void SaveClimaLayout::setGenericPeriodEndDay(const QString &valueStr)
{
    genericPeriodEndDay = valueStr;
}

QString SaveClimaLayout::getGenericPeriodEndMonth() const
{
    return genericPeriodEndMonth;
}

void SaveClimaLayout::setGenericPeriodEndMonth(const QString &valueStr)
{
    genericPeriodEndMonth = valueStr;
}

QString SaveClimaLayout::getGenericNYear() const
{
    return genericNYear;
}

void SaveClimaLayout::setGenericNYear(const QString &valueStr)
{
    genericNYear = valueStr;
}

QString SaveClimaLayout::getSecondElab() const
{
    return secondElab;
}

void SaveClimaLayout::setSecondElab(const QString &valueStr)
{
    secondElab = valueStr;
}

QString SaveClimaLayout::getElab2Param() const
{
    return elab2Param;
}

void SaveClimaLayout::setElab2Param(const QString &valueStr)
{
    elab2Param = valueStr;
}

QString SaveClimaLayout::getElab() const
{
    return elab;
}

void SaveClimaLayout::setElab(const QString &valueStr)
{
    elab = valueStr;
}

QString SaveClimaLayout::getElab1Param() const
{
    return elab1Param;
}

void SaveClimaLayout::setElab1Param(const QString &valueStr)
{
    elab1Param = valueStr;
}
