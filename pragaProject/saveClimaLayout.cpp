#include "saveClimaLayout.h"

bool compareClimateElab(const QString &el1, const QString &el2) {

    QString var1 = el1.split("_")[1];
    QString var2 = el2.split("_")[1];

    if (var1 != var2)
        return var1.compare(var2) < 0;

    return (el1.mid(5, 9).toInt() - el1.left(4).toInt()) > (el2.mid(5, 9).toInt() - el2.left(4).toInt());
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


    QString elabAdded = firstYear + "-" + lastYear + "_" + variable + "_" + period;
    if (period == "Generic")
    {
        elabAdded = elabAdded + "_" + genericPeriodStartDay + ":" + genericPeriodStartMonth + "-" + genericPeriodEndDay + ":" + genericPeriodEndMonth;
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
    if (fileName == "") return;

    QFile elabList(fileName);
    if (elabList.open(QFile::ReadOnly | QFile::Text))
    {
      QTextStream sIn(&elabList);
      while (!sIn.atEnd())
      {
          QString line = sIn.readLine();
          if (list.contains(line) == 0)
          {
            list << line;
          }
      }
      std::sort(list.begin(), list.end(), compareClimateElab);

      listView.clear();
      listView.addItems(list);

    }
    else
    {
      qDebug() << "error opening output file\n";
      return;
    }

}

QList<QString> SaveClimaLayout::getList() const
{
    return list;
}

void SaveClimaLayout::setList(const QList<QString> &value)
{
    list = value;
    listView.addItems(list);
}

QString SaveClimaLayout::getElab1ParamFromdB() const
{
    return elab1ParamFromdB;
}

void SaveClimaLayout::setElab1ParamFromdB(const QString &value)
{
    elab1ParamFromdB = value;
}

QString SaveClimaLayout::getFirstYear() const
{
    return firstYear;
}

void SaveClimaLayout::setFirstYear(const QString &value)
{
    firstYear = value;
}

QString SaveClimaLayout::getLastYear() const
{
    return lastYear;
}

void SaveClimaLayout::setLastYear(const QString &value)
{
    lastYear = value;
}

QString SaveClimaLayout::getVariable() const
{
    return variable;
}

void SaveClimaLayout::setVariable(const QString &value)
{
    variable = value;
}

QString SaveClimaLayout::getPeriod() const
{
    return period;
}

void SaveClimaLayout::setPeriod(const QString &value)
{
    period = value;
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

void SaveClimaLayout::setGenericPeriodStartMonth(const QString &value)
{
    genericPeriodStartMonth = value;
}

QString SaveClimaLayout::getGenericPeriodEndDay() const
{
    return genericPeriodEndDay;
}

void SaveClimaLayout::setGenericPeriodEndDay(const QString &value)
{
    genericPeriodEndDay = value;
}

QString SaveClimaLayout::getGenericPeriodEndMonth() const
{
    return genericPeriodEndMonth;
}

void SaveClimaLayout::setGenericPeriodEndMonth(const QString &value)
{
    genericPeriodEndMonth = value;
}

QString SaveClimaLayout::getGenericNYear() const
{
    return genericNYear;
}

void SaveClimaLayout::setGenericNYear(const QString &value)
{
    genericNYear = value;
}

QString SaveClimaLayout::getSecondElab() const
{
    return secondElab;
}

void SaveClimaLayout::setSecondElab(const QString &value)
{
    secondElab = value;
}

QString SaveClimaLayout::getElab2Param() const
{
    return elab2Param;
}

void SaveClimaLayout::setElab2Param(const QString &value)
{
    elab2Param = value;
}

QString SaveClimaLayout::getElab() const
{
    return elab;
}

void SaveClimaLayout::setElab(const QString &value)
{
    elab = value;
}

QString SaveClimaLayout::getElab1Param() const
{
    return elab1Param;
}

void SaveClimaLayout::setElab1Param(const QString &value)
{
    elab1Param = value;
}
