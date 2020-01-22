#ifndef CURVEPICKER_H
#define CURVEPICKER_H

    // custom QwtPlotPicker to select a curve

    #include <qwt_plot_curve.h>
    #include <qwt_plot.h>
    #include <qwt_plot_picker.h>

    class Crit3DCurvePicker : public QwtPlotPicker
    {
        Q_OBJECT
    public:
        Crit3DCurvePicker(QwtPlot *plot, QList<QwtPlotCurve*> allCurveList, QMap< int, QwtPlotCurve* > allCurveMarkerMap);
        Crit3DCurvePicker(QwtPlot *plot, QList<QwtPlotCurve*> allCurveList);

        Crit3DCurvePicker( int xAxis = QwtPlot::xBottom,
                  int yAxis = QwtPlot::yLeft,
                  RubberBand rubberBand = CrossRubberBand,
                  DisplayMode trackerMode = QwtPicker::AlwaysOn,
                  QwtPlot *plot = nullptr );

        void highlightCurve( bool isHightlight );
        int closestPoint(QwtPlotCurve& curve, const QPointF &pos, double *dist );

        int getSelectedCurveIndex() const;
        void setSelectedCurveIndex(int value);

    public slots:
        void slotSelected( const QPointF &pos);

    private:
        QwtPlot *qwtPlot;
        int selectedCurveIndex;
        QList<QwtPlotCurve*> allCurveList;
        QMap< int, QwtPlotCurve* > allCurveMarkerMap;

    signals:
           void clicked(int curveIndex);
    };


#endif // CURVEPICKER_H
