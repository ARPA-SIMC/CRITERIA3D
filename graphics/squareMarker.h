#ifndef SQUAREMARKER_H
#define SQUAREMARKER_H

    #include "MapGraphics_global.h"
    #include "SquareObject.h"

    class SquareMarker : public SquareObject
    {
        Q_OBJECT

        private:
            bool _active;

        public:
            explicit SquareMarker(qreal side, bool sizeIsZoomInvariant, QColor fillColor, MapGraphicsObject *parent = nullptr);

            bool active() const { return _active; }
            void setActive(bool active) { _active = active; }

            void setToolTip();

        protected:
            void mousePressEvent(QGraphicsSceneMouseEvent *event);
            void mouseReleaseEvent(QGraphicsSceneMouseEvent *event);
        signals:

    };

#endif // SQUAREMARKER_H


