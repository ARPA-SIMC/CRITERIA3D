#ifndef TABROOTDENSITY_H
#define TABROOTDENSITY_H

    #include <QtWidgets>
    #include <QtCharts>

    #include "callout.h"
    #include "criteria1DProject.h"

    class Crit3DMeteoPoint;

    class TabRootDensity : public QWidget
    {
        Q_OBJECT

    public:
        TabRootDensity();
        void computeRootDensity(const Crit1DProject &myProject, int firstYear, int lastYear);

        void on_actionChooseYear(QString myYear);
        void updateDate();
        void updateRootDensity();
        void tooltip(bool state, int index, QBarSet *barset);

    private:
        Crit3DCrop myCrop;
        Crit3DMeteoPoint mp;
        std::vector<soil::Crit1DLayer> layers;

        unsigned int nrLayers;
        int year;
        QDate lastMeteoDate;

        QComboBox yearComboBox;
        QList<double> depthLayers;
        QSlider* slider;
        QDateEdit *currentDate;
        QChartView *chartView;
        QChart *chart;
        QHorizontalBarSeries *seriesRootDensity;
        QBarSet *set;
        QValueAxis *axisX;
        QBarCategoryAxis *axisY;
        QList<QString> categories;
        Callout *m_tooltip;
    };

#endif // TABROOTDENSITY_H
