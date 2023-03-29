#ifndef DIALOGCELLSIZE_H
#define DIALOGCELLSIZE_H

#include <QtWidgets>

class DialogCellSize : public QDialog
{
    Q_OBJECT

private:
    QLineEdit cellSizeEdit;

public:
    DialogCellSize(int defaultCellSize);
    ~DialogCellSize();
    void done(bool res);
    double getCellSize() const;
};

#endif // DIALOGCELLSIZE_H
