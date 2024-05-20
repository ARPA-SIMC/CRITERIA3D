#ifndef DIALOGSUMMARY_H
#define DIALOGSUMMARY_H

#include <QtWidgets>
#include "waterTable.h"

class DialogSummary : public QDialog
{
public:
    DialogSummary(WaterTable myWaterTable);
    ~DialogSummary();
    void closeEvent(QCloseEvent *event);
};

#endif // DIALOGSUMMARY_H
