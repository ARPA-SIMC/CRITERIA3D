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
        this->setColumnCount(10);
        tableHeader << "Upper depth [cm]" << "Lower depth [cm]" << "Sand [%]" << "Silt  [%]" << "Clay [%]" << "Coarse [%]" << "O.M. [%]"
                        << "B.D. [g/cm3]" << "K Sat [cm/d]" << "Theta S [-]";
    }
    else if (type == modelTable)
    {
        this->setColumnCount(11);
        tableHeader << "USDA Texture" << "Coarse [%]" << "O.M. [%]"
                        << "B.D. [g/cm3]" << "K Sat [cm/d]" << "ThetaS [-]" << "ThetaR [-]" << "Air entry [KPa]"
                        << "alpha [KPa^-1]" << "  n  [-] " << " m   [-] ";
    }

    this->setHorizontalHeaderLabels(tableHeader);
    this->resizeColumnsToContents(); 
    this->setSelectionMode(QAbstractItemView::SingleSelection);
    this->setShowGrid(true);
    //this->setStyleSheet("QTableView {selection-background-color: green;}");
    this->setStyleSheet("QTableView::item:selected { color:black;  border: 3px solid black}");

    if (type == dbTable)
    {
        this->setItemDelegate(new TableDelegate(this));
    }
    else if (type == modelTable)
    {
        this->setEditTriggers(QAbstractItemView::NoEditTriggers);
    }
}


void Crit3DSoilTable::mouseMoveEvent(QMouseEvent *event)
{
    QPoint pos = event->pos();
    QTableWidgetItem *item = this->itemAt(pos);
    if(!item) return;

    if (item->background().color() == Qt::red)
    {
        if (type == dbTable)
        {
            QToolTip::showText(this->viewport()->mapToGlobal(pos), "wrong value", this, QRect(pos,QSize(100,100)), 800);
        }
        else if (type == modelTable)
        {
            QToolTip::showText(this->viewport()->mapToGlobal(pos), "wrong horizon", this, QRect(pos,QSize(100,100)), 800);
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
