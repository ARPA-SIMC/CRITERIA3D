#ifndef SOILTABLE_H
#define SOILTABLE_H

    #include <QTableWidget>

    // custom QTableWidget to implement mouseMoveEvent and manage QToolTip timeout
    // tables can be of 2 different types: dbTable or modelTable (each with specific header and background color)

    enum tableType{dbTable, modelTable};

    class Crit3DSoilTable: public QTableWidget
    {
    Q_OBJECT
    public:
        Crit3DSoilTable(tableType type);

        void mouseMoveEvent(QMouseEvent *event);
        void keyPressEvent(QKeyEvent *event);
        void copyAll();
        void exportToCsv(QString csvFile);
    private:
        tableType type;

    };


#endif // SOILTABLE_H
