#ifndef COLORLEGEND_H
#define COLORLEGEND_H

    #include <QWidget>

    #ifndef CRIT3DCOLOR_H
        #include "color.h"
    #endif

    class ColorLegend : public QWidget
    {
        Q_OBJECT

    public:
        explicit ColorLegend(QWidget *parent = nullptr);
        ~ColorLegend();

        Crit3DColorScale *colorScale;

    private:
        void paintEvent(QPaintEvent *event);
    };

#endif // COLORLEGEND_H
