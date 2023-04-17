#ifndef FORMTEXT_H
#define FORMTEXT_H

    #include <QtWidgets>

    class FormText : public QDialog
    {
        Q_OBJECT

    private:
        QLineEdit textEdit;

    public:
        FormText(QString title);

        QString getText() const;
    };

#endif // FORMTEXT_H
