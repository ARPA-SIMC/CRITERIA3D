#ifndef BARHORIZON_H
#define BARHORIZON_H

    #include <QWidget>
    #include <QFrame>
    #include <QPainter>
    #include <QLabel>
    #include <QGroupBox>
    #include <QLayout>
    #include "soil.h"

    class BarHorizon : public QFrame
    {
        Q_OBJECT
    public:
        BarHorizon(QWidget *parent = nullptr);
        void setClass(int classUSDA);
        int getIndex() const;
        void setIndex(int value);
        bool getSelected() const;
        void restoreFrame();
        void setSelected(bool value);
        void setSelectedFrame();

    protected:
        void mousePressEvent(QMouseEvent* event);

    private:
        QLabel *labelNumber;
        int index;
        int classUSDA;
        bool selected;

    signals:
           void clicked(int index);

    };


    class BarHorizonList
    {
    private:
        QVBoxLayout* barLayout;
        QVBoxLayout* depthLayout;
        QHBoxLayout* mainLayout;

    public:
        QGroupBox* groupBox;
        QList<BarHorizon*> barList;
        QList<QLabel*> labelList;

        BarHorizonList();

        void clear();
        void selectItem(int index);
        void deselectAll(int index);
        void draw(soil::Crit3DSoil *soil);
        QColor getColor(int index);

    };


#endif // BARHORIZON_H
