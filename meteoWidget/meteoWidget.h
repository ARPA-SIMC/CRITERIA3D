#ifndef METEOWIDGET_H
#define METEOWIDGET_H

    #include <QtWidgets>
    #include <QtCharts>
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
            void setDateInterval(QDate date0, QDate date1);
            void draw(Crit3DMeteoPoint mp, bool isAppend);
            void addMeteoPointsEnsemble(Crit3DMeteoPoint mp);
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

    private:
            int meteoWidgetID;
            bool isGrid;
            bool isEnsemble;
            bool isInitialized;
            int nrMembers;
            Crit3DMeteoSettings* meteoSettings;
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
            QVector<Crit3DMeteoPoint> meteoPoints;
            QVector<Crit3DMeteoPoint> meteoPointsEnsemble;
            frequencyType currentFreq;
            QDate firstDailyDate;
            QDate lastDailyDate;
            QDate firstHourlyDate;
            QDate lastHourlyDate;
            QDate currentDate;
            bool isLine;
            bool isBar;
            Callout *m_tooltip;
    signals:
        void closeWidgetPoint(int);
        void closeWidgetGrid(int);

    };


#endif // METEOWIDGET_H
