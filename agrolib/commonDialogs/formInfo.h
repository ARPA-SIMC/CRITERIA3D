#ifndef FORMINFO_H
#define FORMINFO_H

    #include <QWidget>
    #include <QLabel>
    #include <QProgressBar>

    class FormInfo : public QWidget
    {
        Q_OBJECT

    public:
        explicit FormInfo();

        int start(QString info, int nrValues);
        void setValue(int myValue);
        void setText(QString myText);
        void showInfo(QString info);

    private:
        QLabel* label;
        QProgressBar* progressBar;

    };


#endif // FORMINFO_H
