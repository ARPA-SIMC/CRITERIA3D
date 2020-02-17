#ifndef TABLEWIDGETITEM_H
#define TABLEWIDGETITEM_H

#include <QTableWidgetItem>

class Crit3DTableWidgetItem : public QTableWidgetItem
{

public:

    Crit3DTableWidgetItem(const QString txt = QString("0"))
        :QTableWidgetItem(txt)
    {
    }

    bool operator <(const QTableWidgetItem& other) const
    {
        //qDebug() << "Sorting numbers";
        return text().toFloat() < other.text().toFloat();
       // text() is part of QTableWidgetItem, so you can write it as QTableWidgetItem::text().toFloat() as well
    }
};

#endif // TABLEWIDGETITEM_H
