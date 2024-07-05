#include <QWidget>
#include <QClipboard>
#include <QApplication>
#include <qevent.h>
#include <qtooltip.h>
#include <qdebug.h>
#include <QFile>

#include "soilTable.h"
#include "tableDelegate.h"


Crit3DSoilTable::Crit3DSoilTable(tableType type) : type(type)
{
    this->setMouseTracking(true);
    this->viewport()->setMouseTracking(true);

    QList<QString> tableHeader;
    if (type == dbTable)
    {
        this->setColumnCount(12);
        tableHeader << "Upper depth [cm]" << "Lower depth [cm]" << "Sand [%]" << "Silt [%]" << "Clay [%]" << "Coarse [%]" << "O.M. [%]"
                    << "B.D. [g/cm3]" << "K Sat [cm/d]" << "Theta S [-]" << "c' [kPa]" << "Φ' [°]";
    }
    else if (type == modelTable)
    {
        this->setColumnCount(13);
        tableHeader << "USDA Texture" << "Coarse [%]" << "O.M. [%]"
                        << "B.D. [g/cm3]" << "K Sat [cm/d]" << "ThetaS [-]" << "ThetaR [-]" << "Air entry [KPa]"
                        << "α [KPa^-1]" << "  n  [-] " << " m   [-] " << "c' [kPa]" << "Φ' [°]";
    }

    this->setHorizontalHeaderLabels(tableHeader);
    this->resizeColumnsToContents(); 
    this->setSelectionMode(QAbstractItemView::SingleSelection);
    this->setShowGrid(true);
    this->setStyleSheet("QTableView::item:selected { color:black;  border: 3px solid black}");

    if (type == dbTable)
    {
        this->setItemDelegate(new TableDelegate(this));
    }
    else if (type == modelTable)
    {
        this->setEditTriggers(QAbstractItemView::NoEditTriggers);
    }

    if (type == dbTable)
    {
        QTableWidgetItem *currentHeaderItem = horizontalHeaderItem(2);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Percentage of sand (from 2.0 to 0.05 mm)");

        currentHeaderItem = horizontalHeaderItem(3);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Percentage of silt (from 0.05 to 0.002 mm)");

        currentHeaderItem = this->horizontalHeaderItem(4);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Percentage of clay (minor than 0.002 mm)");

        currentHeaderItem = this->horizontalHeaderItem(5);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Percentage of coarse fragments (major than 2.0 mm)");

        currentHeaderItem = this->horizontalHeaderItem(6);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Percentage of organic matter");

        currentHeaderItem = this->horizontalHeaderItem(7);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Bulk density");

        currentHeaderItem = this->horizontalHeaderItem(8);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Saturated hydraulic conductivity");

        currentHeaderItem = this->horizontalHeaderItem(9);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Water content at saturation (SAT)");

        currentHeaderItem = this->horizontalHeaderItem(10);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Soil effective cohesion");

        currentHeaderItem = this->horizontalHeaderItem(11);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Soil friction angle");
    }
    else if (type == modelTable)
    {
        QTableWidgetItem *currentHeaderItem = horizontalHeaderItem(0);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("USDA textural soil classification");

        currentHeaderItem = horizontalHeaderItem(1);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Percentage of coarse fragments (major than 2.0 mm)");

        currentHeaderItem = this->horizontalHeaderItem(2);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Percentage of organic matter");

        currentHeaderItem = this->horizontalHeaderItem(3);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Bulk density");

        currentHeaderItem = this->horizontalHeaderItem(4);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Saturated hydraulic conductivity");

        currentHeaderItem = this->horizontalHeaderItem(5);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Water content at saturation (SAT)");

        currentHeaderItem = this->horizontalHeaderItem(6);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Water content at wilting point (WP)");

        currentHeaderItem = this->horizontalHeaderItem(7);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Air entry value");

        currentHeaderItem = this->horizontalHeaderItem(8);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Van Genuchten parameter α");

        currentHeaderItem = this->horizontalHeaderItem(9);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Van Genuchten parameter n");

        currentHeaderItem = this->horizontalHeaderItem(10);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Van Genuchten parameter m");

        currentHeaderItem = this->horizontalHeaderItem(11);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Soil effective cohesion");

        currentHeaderItem = this->horizontalHeaderItem(12);
        if (currentHeaderItem)
            currentHeaderItem->setToolTip("Soil friction angle");
    }
}


void Crit3DSoilTable::mouseMoveEvent(QMouseEvent *event)
{
    QPoint pos = event->pos();
    QTableWidgetItem *item = this->itemAt(pos);
    if(! item)
        return;

    if (item->background().color() == Qt::red)
    {
        if (type == dbTable)
        {
            QToolTip::showText(this->viewport()->mapToGlobal(pos), "wrong value", this, QRect(pos,QSize(100,100)), 800);
        }
        else if (type == modelTable)
        {
            QToolTip::showText(this->viewport()->mapToGlobal(pos), "wrong horizon or missing db", this, QRect(pos,QSize(100,100)), 800);
        }
    }
    else if(item->background().color() == Qt::yellow)
    {
        if (type == dbTable)
        {
            QToolTip::showText(this->viewport()->mapToGlobal(pos), "missing data", this, QRect(pos,QSize(100,100)), 800);
        }
        else if (type == modelTable)
        {
            QToolTip::showText(this->viewport()->mapToGlobal(pos), "estimated value", this, QRect(pos,QSize(100,100)), 800);
        }

    }
}


void Crit3DSoilTable::keyPressEvent(QKeyEvent *event)
{
    Q_UNUSED(event)
    return;
}

void Crit3DSoilTable::copyAll()
{
    int nRow = this->rowCount();
    int nCol = this->columnCount();
    QString text;
    QList<QString> headerContents;
    // copy header
    for(int j=0; j<nCol; j++)
    {
        headerContents << this->horizontalHeaderItem(j)->text();
    }
    text += headerContents.join("\t");
    text += "\n";

    // copy row
    for(int i=0; i<nRow; i++)
    {
        QList<QString> rowContents;
        for(int j=0; j<nCol; j++)
        {
            rowContents << model()->index(i,j).data().toString();
        }
        text += rowContents.join("\t");
        text += "\n";
    }
    QApplication::clipboard()->setText(text);
}

void Crit3DSoilTable::exportToCsv(QString csvFile)
{

    QFile file(csvFile);
    int nRow = this->rowCount();
    int nCol = this->columnCount();
    QString conTents;
    QList<QString> strList;
    // copy header
    QHeaderView * header = this->horizontalHeader() ;
    if (header)
    {
        for ( int i = 0; i < nCol; i++ )
        {
            QTableWidgetItem *item = this->horizontalHeaderItem(i);
            if (!item)
            {
                continue;
            }
            conTents += item->text() + ",";
        }
        conTents += "\n";
    }

    if (file.open(QFile::WriteOnly | QFile::Truncate))
    {
        QTextStream data( &file );
        data << strList.join(",") << "\n";

        for(int i=0; i<nRow; i++)
        {
            for(int j=0; j<nCol; j++)
            {
                QTableWidgetItem* item = this->item(i, j);
                if ( !item )
                    continue;
                QString str = item->text();
                str.replace(","," ");
                conTents += str + ",";
            }
            conTents += "\n";
        }
        data << conTents;
        file.close();
    }
}
