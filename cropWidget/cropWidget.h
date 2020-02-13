#ifndef CROPWIDGET_H
#define CROPWIDGET_H

#include <QWidget>
#include <QComboBox>
#include <QGroupBox>
#include <QLineEdit>
#include <QLabel>
#include <QSqlDatabase>

#ifndef CROP_H
    #include "crop.h"
#endif

#include "tabLAI.h"
#include "meteoPoint.h"

class Crit3DCropWidget : public QWidget
{
    Q_OBJECT

    public:
        Crit3DCropWidget();
        void on_actionOpenCropDB();
        void on_actionChooseCrop(QString cropName);
        void on_actionOpenMeteoDB();
        void on_actionChooseMeteo(QString idMeteo);
        void on_actionChooseYear(QString year);
        void on_actionDeleteCrop();
        void on_actionRestoreData();
        void on_actionNewCrop();
    private:
        QSqlDatabase dbCrop;
        QSqlDatabase dbMeteo;
        Crit3DCrop* myCrop;
        QString tableMeteo;
        Crit3DMeteoPoint *meteoPoint;

        QGroupBox *infoCropGroup;
        QGroupBox *infoMeteoGroup;
        QComboBox cropListComboBox;
        QComboBox meteoListComboBox;
        QComboBox yearListComboBox;
        QLineEdit* cropIdValue;
        QLineEdit* cropTypeValue;
        QLabel cropSowing;
        QLabel cropCycleMax;
        QLineEdit* cropSowingValue;
        QLineEdit* cropCycleMaxValue;
        QLineEdit* latValue;
        QLineEdit* lonValue;
        QTabWidget* tabWidget;
        QAction* saveChanges;
        QAction* restoreData;

        TabLAI* tabLAI;
};

#endif // CROPWIDGET_H
