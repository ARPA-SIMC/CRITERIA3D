#ifndef TABLEWATERRETENTION_H
#define TABLEWATERRETENTION_H

#include <QTableWidget>
#include <QKeyEvent>
#include <QWidget>

// custom QTableWidget to implement keyPressEvent and achieve the copy/paste functionality

class TableWaterRetention: public QTableWidget
{
Q_OBJECT
public:
    TableWaterRetention();
    void keyPressEvent(QKeyEvent *event);

};

#endif // TABLEWATERRETENTION_H
