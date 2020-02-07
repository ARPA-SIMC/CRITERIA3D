#ifndef CROPWIDGET_H
#define CROPWIDGET_H

#include <QWidget>
#include <QComboBox>
#include <QGroupBox>
#include <QLineEdit>

class Crit3DCropWidget : public QWidget
{
    Q_OBJECT

    public:
        Crit3DCropWidget();
    private:
        QGroupBox *infoCropGroup;
        QGroupBox *infoMeteoGroup;
        QComboBox cropListComboBox;
        QComboBox meteoListComboBox;
        QLineEdit* cropIdValue;
        QLineEdit* cropTypeValue;
        QLineEdit* cropSowingValue;
        QLineEdit* cropCycleMaxValue;
        QTabWidget* tabWidget;
        QAction* saveChanges;
        QAction* restoreData;
};

#endif // CROPWIDGET_H
