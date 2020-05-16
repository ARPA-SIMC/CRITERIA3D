#include "tableDelegateWaterRetention.h"
#include <QLineEdit>
#include <QDoubleValidator>

TableDelegateWaterRetention::TableDelegateWaterRetention(QObject *parent) : QStyledItemDelegate(parent)
{

}

QWidget* TableDelegateWaterRetention::createEditor(QWidget* parent,const QStyleOptionViewItem &option,const QModelIndex &index) const
{
    Q_UNUSED(option);
    Q_UNUSED(index);

    QLineEdit* editor = new QLineEdit(parent);
    QDoubleValidator* val = new QDoubleValidator(editor);
    val->setBottom(0);
    val->setNotation(QDoubleValidator::StandardNotation);
    editor->setValidator(val);
    return editor;
}

void TableDelegateWaterRetention::setEditorData(QWidget *editor, const QModelIndex &index) const
{ 
    double value = index.model()->data(index,Qt::EditRole).toDouble();
    QLineEdit* line = static_cast<QLineEdit*>(editor);
    line->setText(QString().setNum(value));
}

void TableDelegateWaterRetention::setModelData(QWidget* editor,QAbstractItemModel* model,const QModelIndex &index) const
{
    QLineEdit* line = static_cast<QLineEdit*>(editor);
    QString value = line->text();
    model->setData(index,value);
}

void TableDelegateWaterRetention::updateEditorGeometry(QWidget *editor, const QStyleOptionViewItem &option, const QModelIndex &index) const
{
    Q_UNUSED(index);
    editor->setGeometry(option.rect);
}
