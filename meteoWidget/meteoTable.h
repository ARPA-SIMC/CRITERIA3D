#ifndef METEOTABLE_H
#define METEOTABLE_H

#include <QTableWidget>
#include <QKeyEvent>
#include <QWidget>

// custom QTableWidget to implement keyPressEvent and achieve the copy functionality

class MeteoTable: public QTableWidget
{
    Q_OBJECT
public:
    MeteoTable();
    void keyPressEvent(QKeyEvent *event);
};

#endif // METEOTABLE_H
