#ifndef METEOWIDGET_H
#define METEOWIDGET_H

    #include <QWidget>
    #include <QComboBox>
    #include <QGroupBox>
    #include <QLineEdit>
    #include <QLabel>
#include "meteoPoint.h"


    class Crit3DMeteoWidget : public QWidget
    {
        Q_OBJECT

        public:
            Crit3DMeteoWidget();
            void draw(Crit3DMeteoPoint* meteoPoint);

        private:
            QMap<QString, QStringList> MapCSVDefault;
            QMap<QString, QStringList> MapCSVStyles;

    };


#endif // METEOWIDGET_H
