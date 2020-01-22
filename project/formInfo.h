#ifndef FORMINFO_H
#define FORMINFO_H

    #include <QDialog>

    namespace Ui {
    class formInfo;
    }

    class FormInfo : public QDialog
    {
        Q_OBJECT

    public:
        explicit FormInfo(QWidget *parent = nullptr);
        ~FormInfo();

        int start(QString info, int nrValues);
        void setValue(int myValue);
        void setText(QString myText);
        void showInfo(QString info);

    private:
        Ui::formInfo *ui;
    };

#endif // FORMINFO_H
