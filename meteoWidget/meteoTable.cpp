#include "meteoTable.h"
#include <QClipboard>
#include <QApplication>

MeteoTable::MeteoTable()
{

}

void MeteoTable::keyPressEvent(QKeyEvent *event)
{
    // at least one cell selected
    if(!selectedIndexes().isEmpty())
    {
        if(event->matches(QKeySequence::Copy))
        {

                QString text;
                QItemSelectionRange range = selectionModel()->selection().first();
                QList<QString> headerContents;

                if (range.bottom() - range.top() == this->rowCount()-1)
                {
                    for (auto j = range.left(); j <= range.right(); ++j)
                    {
                        headerContents << this->horizontalHeaderItem(j)->text();
                    }
                }

                for (auto i = range.top(); i <= range.bottom(); ++i)
                {
                    QList<QString> rowContents;
                    for (auto j = range.left(); j <= range.right(); ++j)
                    {
                        rowContents << model()->index(i,j).data().toString();
                    }
                    if (!headerContents.isEmpty())
                    {
                        text += headerContents.join("\t");
                        text += "\n";
                        headerContents.clear();
                    }
                    text += rowContents.join("\t");
                    text += "\n";
                }
                QApplication::clipboard()->setText(text);
        }
         else
            QTableView::keyPressEvent(event);
    }
}
