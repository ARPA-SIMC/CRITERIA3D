#include "dialogMeteoTable.h"

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
    int rowNumber = firstDate.daysTo(lastDate)+1;

    meteoTable->setRowCount(rowNumber);
    meteoTable->setColumnCount(colNumber);

    std::string nameField;

    labels.clear();
    meteoTableHeader.clear();
    meteoTableHeader << "ID" << "Date";

    for (int i=0; i < currentVariables.size(); i++)
    {
        meteoTableHeader << currentVariables[i];

        for (int j = 0; j < rowNumber; j++)
        {
            //meteoTable->setItem(j, i, new QTableWidgetItem( QString::fromStdString()));
        }
    }

    for (int j = 0; j < rowNumber; j++)
    {
        labels << QString::number(j);
    }

    meteoTable->setVerticalHeaderLabels(labels);
    meteoTable->setHorizontalHeaderLabels(meteoTableHeader);
    meteoTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    meteoTable->setSelectionMode(QAbstractItemView::SingleSelection);
    meteoTable->setShowGrid(true);
    meteoTable->setStyleSheet("QTableView {selection-background-color: red;}");

    //connect(meteoTable->horizontalHeader(), &QHeaderView::sectionClicked, [=](int index){ this->horizontalHeaderClick(index); });
    //connect(meteoTable->verticalHeader(), &QHeaderView::sectionClicked, [=](int index){ this->verticalHeaderClick(index); });

    setLayout(mainLayout);
    exec();
}

DialogMeteoTable::~DialogMeteoTable()
{
    close();
}
