#ifndef LOCALPROXYWIDGET_H
#define LOCALPROXYWIDGET_H

#include <QtWidgets>
#include <QtCharts>
#include "chartView.h"
#include "meteo.h"
#include "meteoPoint.h"
#include "interpolationSettings.h"
#include "interpolationPoint.h"

class Crit3DLocalProxyWidget : public QWidget
{
    Q_OBJECT

public:
    Crit3DLocalProxyWidget(double x, double y, double zDEM, double zGrid, gis::Crit3DGisSettings gisSettings,
                           Crit3DInterpolationSettings* interpolationSettings, Crit3DMeteoPoint* meteoPoints,
                           int nrMeteoPoints, meteoVariable currentVariable, frequencyType currentFrequency,
                           QDate currentDate, int currentHour, Crit3DQuality* quality,
                           Crit3DInterpolationSettings* SQinterpolationSettings, Crit3DMeteoSettings *meteoSettings,
                           Crit3DClimateParameters *climateParameters, bool checkSpatialQuality);

    ~Crit3DLocalProxyWidget();
    void closeEvent(QCloseEvent *event);
    void updateDateTime(QDate newDate, int newHour);
    void updateFrequency(frequencyType newFrequency);
    void changeProxyPos(const QString proxyName);
    void changeVar(const QString varName);
    void plot();
    void climatologicalLRClicked(int toggled);
    void modelLRClicked(int toggled);
    void showParametersDetails();

private:
    double _x;
    double _y;
    double _zDEM;
    double _zGrid;

    gis::Crit3DGisSettings _gisSettings;
    Crit3DInterpolationSettings* _interpolationSettings;

    Crit3DMeteoPoint* _meteoPoints;
    int _nrMeteoPoints;
    meteoVariable _currentVariable;
    frequencyType _currentFrequency;

    QDate _currentDate;
    int _currentHour;

    Crit3DQuality* _quality;
    Crit3DInterpolationSettings* _SQinterpolationSettings;
    Crit3DMeteoSettings *_meteoSettings;
    Crit3DClimateParameters *_climateParameters;

    bool _checkSpatialQuality;

    std::vector <Crit3DInterpolationDataPoint> outInterpolationPoints;
    std::vector <Crit3DInterpolationDataPoint> subsetInterpolationPoints;

    QComboBox comboVariable;
    QComboBox comboAxisX;
    QCheckBox detrended;
    QCheckBox climatologicalLR;
    QCheckBox modelLR;
    QCheckBox stationWeights;
    QTextEdit r2;
    QTextEdit lapseRate;
    QTextEdit par0;
    QTextEdit par1;
    QTextEdit par2;
    QTextEdit par3;
    QTextEdit par4;
    QTextEdit par5;
    ChartView *chartView;
    meteoVariable myVar;
    int proxyPos;
    std::vector <QGraphicsTextItem*> weightLabels;

    Crit3DTime getCurrentTime();

signals:
    void closeLocalProxyWidget();
};


#endif // PROXYWIDGET_H
