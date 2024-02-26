#ifndef METEOWIDGET_H
#define METEOWIDGET_H

    #include <QtWidgets>
    #include <QtCharts>
    #include "meteo.h"
    #include "meteoPoint.h"
    #include "callout.h"

    qreal findMedian(QList<double> sortedList, int begin, int end);

    class Crit3DMeteoWidget : public QWidget
    {
        Q_OBJECT

        public:
            Crit3DMeteoWidget(bool isGrid, QString projectPath, Crit3DMeteoSettings* meteoSettings_);
            ~Crit3DMeteoWidget() override;

            int getMeteoWidgetID() const;
            void setMeteoWidgetID(int value);

            void setCurrentDate(QDate myDate);
            void setDateIntervalDaily(QDate firstDate, QDate lastDate);
            void setDateIntervalHourly(QDate firstDate, QDate lastDate);

            void addMeteoPointsEnsemble(Crit3DMeteoPoint mp);

            void draw(Crit3DMeteoPoint mp, bool isAppend);
            void drawEnsemble();

            void resetValues();
            void resetEnsembleValues();
            void drawDailyVar();
            void drawEnsembleDailyVar();
            void drawHourlyVar();
            void showDailyGraph();
            void showHourlyGraph();
            void updateSeries();
            void redraw();
            void shiftPrevious();
            void shiftFollowing();
            void showTable();
            void showVar();
            void tooltipLineSeries(QPointF point, bool state);
            bool computeTooltipLineSeries(QLineSeries *series, QPointF point, bool state);
            void tooltipBar(bool state, int index, QBarSet *barset);
            void handleMarkerClicked();
            void closeEvent(QCloseEvent *event) override;
            void setIsEnsemble(bool value);
            bool getIsEnsemble();
            void setNrMembers(int value);
            void on_actionChangeLeftAxis();
            void on_actionChangeRightAxis();
            void on_actionExportGraph();
            void on_actionRemoveStation();
            void on_actionInfoPoint();
            void on_actionDataAvailability();

    private:
            int meteoWidgetID;
            bool isGrid;
            bool isEnsemble;
            bool isInitialized;
            int nrMembers;

            QVector<Crit3DMeteoPoint> meteoPoints;
            QVector<Crit3DMeteoPoint> meteoPointsEnsemble;
            Crit3DMeteoSettings* meteoSettings;

            frequencyType currentFreq;
            QDate firstDailyDate;
            QDate lastDailyDate;
            QDate firstHourlyDate;
            QDate lastHourlyDate;
            QDate currentDate;

            QPushButton *addVarButton;
            QPushButton *dailyButton;
            QPushButton *hourlyButton;
            QPushButton *tableButton;
            QPushButton *redrawButton;
            QPushButton *shiftPreviousButton;
            QPushButton *shiftFollowingButton;
            QDateTimeEdit *firstDate;
            QDateTimeEdit *lastDate;
            QChartView *chartView;
            QChart *chart;
            QBarCategoryAxis *axisX;
            QBarCategoryAxis *axisXvirtual;
            QValueAxis *axisY;
            QValueAxis *axisYdx;
            QMap<QString, QList<QString>> MapCSVDefault;
            QMap<QString, QList<QString>> MapCSVStyles;
            QList<QString> currentVariables;
            QList<QString> nameLines;
            QList<QString> nameBar;
            double maxEnsembleBar;
            double maxEnsembleLine;
            double minEnsembleLine;
            QVector<QColor> colorLines;
            QVector<QColor> colorBar;
            QVector<QVector<QLineSeries*>> lineSeries;
            QVector<QBarSeries*> barSeries;
            QVector<QBoxPlotSeries*> ensembleSeries;
            QVector<QList<QBoxSet*>> ensembleSet;
            QVector<QVector<QBarSet*>> setVector;
            QList<QString> categories;
            QList<QString> categoriesVirtual;

            bool isLine;
            bool isBar;
            Callout *m_tooltip;
    signals:
        void closeWidgetPoint(int);
        void closeWidgetGrid(int);

    };


#endif // METEOWIDGET_H
