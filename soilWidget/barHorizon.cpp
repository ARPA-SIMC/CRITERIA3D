#include "barHorizon.h"
#include "soil.h"
#include "commonConstants.h"
#include <qdebug.h>

BarHorizon::BarHorizon(QWidget *parent)
{
    Q_UNUSED(parent);

    selected = false;
    this->setFrameStyle(QFrame::NoFrame);
    labelNumber = new QLabel;

    // font size
    QFont font = labelNumber->font();
    font.setPointSize(8);
    labelNumber->setFont(font);

    QHBoxLayout *layoutNumber = new QHBoxLayout;
    layoutNumber->setAlignment(Qt::AlignCenter);
    layoutNumber->addWidget(labelNumber);
    setLayout(layoutNumber);
}

void BarHorizon::setClass(int classUSDA)
{

    this->classUSDA = classUSDA;
    QPalette linePalette = palette();

    switch (classUSDA) {
    // sand
    case 1:
    {
        linePalette.setColor(QPalette::Window, QColor(255,206,156));
        break;
    }
    // loamy sand
    case 2:
    {
        linePalette.setColor(QPalette::Window, QColor(240,190,190));
        break;
    }
    // sandy loam
    case 3:
    {
        linePalette.setColor(QPalette::Window, QColor(240,190,240));
        break;
    }
    // silt loam
    case 4:
    {
        linePalette.setColor(QPalette::Window, QColor(156,206,000));
        break;
    }
    // loam
    case 5:
    {
        linePalette.setColor(QPalette::Window, QColor(206,156,000));
        break;
    }
    // silt
    case 6:
    {
        linePalette.setColor(QPalette::Window, QColor(000,255,49));
        break;
    }
    // sandy clayloam
    case 7:
    {
        linePalette.setColor(QPalette::Window, QColor(255,156,156));
        break;
    }
    // silty clayloam
    case 8:
    {
        linePalette.setColor(QPalette::Window, QColor(99,206,156));
        break;
    }
    // clayloam
    case 9:
    {
        linePalette.setColor(QPalette::Window, QColor(206,255,99));
        break;
    }
    // sandy clay
    case 10:
    {
        linePalette.setColor(QPalette::Window, QColor(255,000,000));
        break;
    }
    // silty clay
    case 11:
    {
        linePalette.setColor(QPalette::Window, QColor(128,255,206));
        break;
    }
    // clay
    case 12:
    {
        linePalette.setColor(QPalette::Window, QColor(220,220,128));
        break;
    }


    }
    this->setAutoFillBackground(true);
    this->setPalette(linePalette);

}

void BarHorizon::mousePressEvent(QMouseEvent* event)
{
    Q_UNUSED(event);

    // select the element
    if (selected == false)
    {
        selected = true;       
        setSelectedFrame();
    }
    // de-select the element
    else
    {
        selected = false;
        restoreFrame();
    }
    emit clicked(index);

}

void BarHorizon::setSelected(bool value)
{
    selected = value;
}

void BarHorizon::setSelectedFrame()
{
    this->setFrameStyle(QFrame::Box);
    this->setLineWidth(2);
}

void BarHorizon::restoreFrame()
{
    this->setFrameStyle(QFrame::NoFrame);
}

bool BarHorizon::getSelected() const
{
    return selected;
}

int BarHorizon::getIndex() const
{
    return index;
}

void BarHorizon::setIndex(int value)
{
    index = value;
    labelNumber->setText(QString::number( (value+1) ));
}


BarHorizonList::BarHorizonList()
{

    groupBox = new QGroupBox();
    groupBox->setMinimumWidth(90);
    groupBox->setTitle("Depth [cm]");

    depthLayout = new QVBoxLayout;
    depthLayout->setAlignment(Qt::AlignTop);

    barLayout = new QVBoxLayout;
    barLayout->setAlignment(Qt::AlignTop | Qt::AlignHCenter);

    mainLayout = new QHBoxLayout;
    mainLayout->addLayout(depthLayout);
    mainLayout->addLayout(barLayout);
    groupBox->setLayout(mainLayout);

}


void BarHorizonList::draw(soil::Crit3DSoil *soil)
{
    int totHeight = int(groupBox->height() * 0.9);
    double soilDepth = soil->horizon[soil->nrHorizons - 1].dbData.lowerDepth;

    for (unsigned int i = 0; i < soil->nrHorizons; i++)
    {
        int length = 0;
        if (soilDepth > 0)
        {
            length = int(totHeight * (soil->horizon[i].dbData.lowerDepth - soil->horizon[i].dbData.upperDepth) / soilDepth);
        }

        BarHorizon* newBar = new BarHorizon();
        newBar->setIndex(signed(i));
        newBar->setFixedWidth(28);
        newBar->setFixedHeight(length);
        newBar->setClass(soil->horizon[i].texture.classUSDA);
        barLayout->addWidget(newBar);
        barList.push_back(newBar); 

        QLabel *depthLabel = new QLabel();
        // font size
        QFont font = depthLabel->font();
        font.setPointSize(8);
        depthLabel->setFont(font);
        if (soil->horizon[i].dbData.upperDepth != NODATA)
        {
            depthLabel->setText(QString::number( (soil->horizon[i].dbData.upperDepth) ));
        }

        depthLabel->setFixedWidth(20);
        depthLabel->setFixedHeight(10);

        depthLayout->addWidget(depthLabel);
        depthLayout->addSpacing(length-13);
        labelList.push_back(depthLabel);

        if (i == soil->nrHorizons -1)
        {
            QLabel *lastLabel = new QLabel();
            // font size
            QFont font = lastLabel->font();
            font.setPointSize(8);
            lastLabel->setFont(font);
            if (soil->horizon[i].dbData.lowerDepth != NODATA)
            {
                lastLabel->setText(QString::number( (soil->horizon[i].dbData.lowerDepth) ));
            }
            lastLabel->setFixedWidth(20);
            lastLabel->setFixedHeight(10);

            depthLayout->addWidget(lastLabel);
            labelList.push_back(lastLabel);
        }
    }
}


void BarHorizonList::clear()
{
    if (!barList.isEmpty())
    {
        qDeleteAll(barList);
        barList.clear();
        qDeleteAll(labelList);
        labelList.clear();
    }
    if ( depthLayout != nullptr )
    {
        QLayoutItem* item;
        while ( ( item = depthLayout->takeAt( 0 ) ) != nullptr )
        {
            delete item->widget();
            delete item;
        }
    }
    if ( barLayout != nullptr )
    {
        QLayoutItem* item;
        while ( ( item = barLayout->takeAt( 0 ) ) != nullptr )
        {
            delete item->widget();
            delete item;
        }
    }
}


void BarHorizonList::selectItem(int index)
{
    for (int i = 0; i < barList.size(); i++)
    {
        if (i != index)
        {
            barList[i]->restoreFrame();
            barList[i]->setSelected(false);
        }
        else
        {
            barList[i]->setSelected(true);
            barList[i]->setSelectedFrame();
        }
    }
}


void BarHorizonList::deselectAll(int index)
{
    for(int i = 0; i < barList.size(); i++)
    {
        if (i != index)
        {
            barList[i]->restoreFrame();
            barList[i]->setSelected(false);
        }
    }
}


QColor BarHorizonList::getColor(int index)
{
    if (index < barList.size())
    {
        return barList[index]->palette().color(QPalette::Window);
    }
    else return QColor(0, 0, 0);
}

