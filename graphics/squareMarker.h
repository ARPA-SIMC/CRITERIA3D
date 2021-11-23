#ifndef SQUAREMARKER_H
#define SQUAREMARKER_H

    #include "MapGraphics_global.h"
    #include "SquareObject.h"

    class SquareMarker : public SquareObject
    {
        Q_OBJECT

        public:
            explicit SquareMarker(qreal side, bool sizeIsZoomInvariant, QColor fillColor, MapGraphicsObject *parent = nullptr);

            std::string id() const;
            bool active() const;

            void setId(std::string id);
            void setActive(bool active);
            void setCurrentValue(float currentValue);
            void setToolTip();

    private:
            std::string _id;
            bool _active;
            float _currentValue;

        protected:
            void mousePressEvent(QGraphicsSceneMouseEvent *event);
            void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
        signals:

    };

#endif // SQUAREMARKER_H


