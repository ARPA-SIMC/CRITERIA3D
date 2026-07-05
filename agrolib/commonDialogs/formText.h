#ifndef FORMTEXT_H
#define FORMTEXT_H

    #include <QDialog>
    #include <QLineEdit>

    class FormText : public QDialog
    {
        Q_OBJECT

    private:
        QLineEdit textEdit;

    public:
        FormText(const QString &title, const QString &text);

        QString getText() const
        { return textEdit.text(); }
    };

#endif // FORMTEXT_H
