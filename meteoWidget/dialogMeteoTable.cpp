#include "dialogMeteoTable.h"
#include "utilities.h"

DialogMeteoTable::DialogMeteoTable(QVector<Crit3DMeteoPoint> meteoPoints, QDate firstDate, QDate lastDate, frequencyType currentFreq, QStringList currentVariables)
    :meteoPoints(meteoPoints), firstDate(firstDate), lastDate(lastDate), currentFreq(currentFreq), currentVariables(currentVariables)
{

    QString title = "Table meteo values ID: ";
    QStringList idList;
    for (int i=0; i<meteoPoints.size(); i++)
    {
        idList << QString::fromStdString(meteoPoints[i].id);
    }
    title = title+idList.join(",");
    this->setWindowTitle(title);
    QVBoxLayout* mainLayout = new QVBoxLayout;
    this->resize(800, 600);


    meteoTable = new QTableWidget();
    mainLayout->addWidget(meteoTable);

    int colNumber = currentVariables.size()+2; //ID, Data, variables
    int nValues = 0;

    if (currentFreq == daily)
    {
        nValues = firstDate.daysTo(lastDate)+1; // naVlues = nDays
    }
    else if (currentFreq == hourly)
    {
        nValues = (firstDate.daysTo(lastDate)+1)*24; // naVlues = nDays * 24 hours
    }
    int rowNumber = (meteoPoints.size())*nValues;

    meteoTable->setRowCount(rowNumber);
    meteoTable->setColumnCount(colNumber);

    std::string nameField;

    labels.clear();
    meteoTableHeader.clear();
    meteoTableHeader << "ID" << "Date";

    for (int i=0; i < currentVariables.size(); i++)
    {
        meteoTableHeader << currentVariables[i];
    }

    for (int j = 0; j < rowNumber; j++)
    {
        labels << QString::number(j);
    }

    QDate myDate;
    QDateTime firstDateTime(firstDate, QTime(0,0,0));
    QDateTime myDateTime;

    for (int row=0; row < rowNumber; row++)
    {
        for (int col = 0; col < colNumber; col++)
        {
            if (col == 0)
            {
                meteoTable->setItem(row, col, new QTableWidgetItem(idList[row / nValues]));
            }
            else if (col == 1)
            {
                if (currentFreq == daily)
                {
                    myDate = firstDate.addDays(row % nValues);
                    meteoTable->setItem(row, col, new QTableWidgetItem(myDate.toString("yyyy-MM-dd")));
                }
                else if (currentFreq == hourly)
                {
                    myDateTime = firstDateTime.addSecs(row % nValues * 3600);
                    meteoTable->setItem(row, col, new QTableWidgetItem(myDateTime.toString("yyyy-MM-dd hh:mm")));
                }
            }
            else
            {
                if (currentFreq == daily)
                {
                    meteoVariable meteoVar = MapDailyMeteoVar.at(currentVariables[col-2].toStdString());
                    double value = meteoPoints[row / nValues].getMeteoPointValueD(getCrit3DDate(myDate), meteoVar);
                    meteoTable->setItem(row, col, new QTableWidgetItem( QString::number(value)));
                }
                else if (currentFreq == hourly)
                {
                    meteoVariable meteoVar = MapHourlyMeteoVar.at(currentVariables[col-2].toStdString());
                    double value = meteoPoints[row / nValues].getMeteoPointValueH(getCrit3DDate(myDateTime.date()), myDateTime.time().hour(), 0, meteoVar);
                    meteoTable->setItem(row, col, new QTableWidgetItem( QString::number(value)));
                }

            }
        }
    }


    meteoTable->setVerticalHeaderLabels(labels);
    meteoTable->setHorizontalHeaderLabels(meteoTableHeader);
    meteoTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    meteoTable->setSelectionMode(QAbstractItemView::SingleSelection);
    meteoTable->setShowGrid(true);
    meteoTable->setStyleSheet("QTableView {selection-background-color: red;}");


    setLayout(mainLayout);
    exec();
}

DialogMeteoTable::~DialogMeteoTable()
{
    close();
}
