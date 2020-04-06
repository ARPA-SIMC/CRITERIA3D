#ifndef METEOWIDGET_H
#define METEOWIDGET_H

    #include <QtWidgets>
    #include <QtCharts>
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
            void draw(QVector<Crit3DMeteoPoint> meteoPoint);

        private:
            QChartView *chartView;
            QChart *chart;
            QBarCategoryAxis *axisX;
            QDateTimeAxis *axisXvirtual;
            QValueAxis *axisY;
            QValueAxis *axisYdx;
            QMap<QString, QStringList> MapCSVDefault;
            QMap<QString, QStringList> MapCSVStyles;
            QVector<QLineSeries*> lineSeries;
            QBarSeries* barSeries;
            QVector<QBarSet*> setVector;

    };


#endif // METEOWIDGET_H
