#ifndef TABLEDELEGATEWATERRETENTION_H
#define TABLEDELEGATEWATERRETENTION_H
#include <QStyledItemDelegate>

class TableDelegateWaterRetention : public QStyledItemDelegate
{
    Q_OBJECT
public:
    TableDelegateWaterRetention(QObject* parent = nullptr);
    QWidget *createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const;
    void setEditorData(QWidget *editor, const QModelIndex &index) const;
    void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const;
    void updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const;
};

#endif // TABLEDELEGATEWATERRETENTION_H
