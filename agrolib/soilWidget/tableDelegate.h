#ifndef TABLEDELEGATE_H
#define TABLEDELEGATE_H

    #include <QStyledItemDelegate>
    // class to have finer control over the editing of items

    #define EMPTYVALUE -9999  //string for empty input

    // custom QDoubleValidator to allow empty values
    class myValidator : public QDoubleValidator {

    public:
        myValidator ( double bottom, double top, int decimals, QObject* parent );
        QValidator::State validate ( QString& input, int& ) const;
    };

    class TableDelegate : public QStyledItemDelegate
    {
    Q_OBJECT

    public:
        TableDelegate(QObject* parent = nullptr);
        QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const;
        void setEditorData(QWidget *editor, const QModelIndex &index) const;
        void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const;
        void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const;
    };


#endif // TABLEDELEGATE_H
