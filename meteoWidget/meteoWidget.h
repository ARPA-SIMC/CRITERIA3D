#ifndef METEOWIDGET_H
#define METEOWIDGET_H

    #include <QtWidgets>
    #include <QtCharts>
    #include <QComboBox>
    #include <QGroupBox>
    #include <QLineEdit>
    #include <QLabel>
    #include "meteoPoint.h"
    #include "callout.h"


    class Crit3DMeteoWidget : public QWidget
    {
        Q_OBJECT

        public:
            Crit3DMeteoWidget();
            void draw(Crit3DMeteoPoint mp);
            void resetValues();
            void drawDailyVar();
            void drawHourlyVar();
            void showDailyGraph();
            void showHourlyGraph();
            void updateSeries();
            void updateDate();
            void showTable();
            void showVar();
            void tooltipLineSeries(QPointF point, bool state);
            bool computeTooltipLineSeries(QLineSeries *series, QPointF point, bool state);
            void tooltipBar(bool state, int index, QBarSet *barset);
            void handleMarkerClicked();
            void closeEvent(QCloseEvent *event);

        private:
            QPushButton *addVarButton;
            QPushButton *dailyButton;
            QPushButton *hourlyButton;
            QPushButton *tableButton;
            QPushButton *redrawButton;
            QDateTimeEdit *firstDate;
            QDateTimeEdit *lastDate;
            QChartView *chartView;
            QChart *chart;
            QBarCategoryAxis *axisX;
            QBarCategoryAxis *axisXvirtual;
            QValueAxis *axisY;
            QValueAxis *axisYdx;
            QMap<QString, QStringList> MapCSVDefault;
            QMap<QString, QStringList> MapCSVStyles;
            QStringList currentVariables;
            QStringList nameLines;
            QStringList nameBar;
            QVector<QColor> colorBar;
            QVector<QVector<QLineSeries*>> lineSeries;
            QVector<QBarSeries*> barSeries;
            QVector<QVector<QBarSet*>> setVector;
            QStringList categories;
            QStringList categoriesVirtual;
            QVector<Crit3DMeteoPoint> meteoPoints;
            frequencyType currentFreq;
            QDate firstDailyDate;
            QDate lastDailyDate;
            QDate firstHourlyDate;
            QDate lastHourlyDate;
            bool isLine;
            bool isBar;
            Callout *m_tooltip;
    signals:
        void closeWidget();

    };


#endif // METEOWIDGET_H
