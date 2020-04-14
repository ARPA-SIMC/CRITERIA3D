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
            void draw(Crit3DMeteoPoint mpVector);
            void resetValues();
            void drawDailyVar();
            void drawHourlyVar();
            void showDailyGraph();
            void showHourlyGraph();
            void updateSeries();
            void showVar();

        private:
            QPushButton *addVarButton;
            QPushButton *dailyButton;
            QPushButton *hourlyButton;
            QChartView *chartView;
            QChart *chart;
            QBarCategoryAxis *axisX;
            QDateTimeAxis *axisXvirtual;
            QValueAxis *axisY;
            QValueAxis *axisYdx;
            QMap<QString, QStringList> MapCSVDefault;
            QMap<QString, QStringList> MapCSVStyles;
            QStringList currentVariables;
            QStringList nameLines;
            QStringList nameBar;
            QVector<QVector<QLineSeries*>> lineSeries;
            QVector<QBarSeries*> barSeries;
            QVector<QVector<QBarSet*>> setVector;
            QStringList categories;
            QVector<Crit3DMeteoPoint> meteoPoints;
            frequencyType currentFreq;
            bool isLine;
            bool isBar;

    };


#endif // METEOWIDGET_H
