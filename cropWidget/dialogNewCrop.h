#ifndef DIALOGNEWCROP_H
#define DIALOGNEWCROP_H

#include <QtWidgets>

class DialogNewCrop : public QDialog
{
    Q_OBJECT
public:
    DialogNewCrop();

private:
    QLineEdit *idCropValue;
    QLineEdit *nameCropValue;
    QLineEdit *typeCropValue;
};

#endif // DIALOGNEWCROP_H
