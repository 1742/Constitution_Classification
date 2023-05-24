import json


if __name__ == '__main__':
    effect_path = r'C:\Users\13632\Documents\Python_Scripts\wuzhou.Tongue\Mine\Constitution_Classification\runs\effect.json'

    with open(effect_path, 'r', encoding='utf-8') as f:
        effect = json.load(f)

    train_loss = effect['loss'][0]
    trian_acc = effect['acc'][0]
    train_precision = effect['precision'][0]
    train_recall = effect['recall'][0]
    train_f1 = effect['f1'][0]

    val_loss = effect['loss'][1]
    val_acc = effect['acc'][1]
    val_precision = effect['precision'][1]
    val_recall = effect['recall'][1]
    val_f1 = effect['f1'][1]

    # test_loss = effect['loss']
    # test_acc = effect['acc']
    # test_IOU = effect['mIOU']

    print('训练损失:', sum(train_loss) / len(train_loss))
    print('训练准确率:', sum(trian_acc) / len(trian_acc))
    print('训练精确率:', sum(train_precision) / len(train_precision))
    print('训练召回率:', sum(train_recall) / len(train_recall))
    print('训练f1率:', sum(train_f1) / len(train_f1))

    # print('测试损失:', sum(test_loss) / len(test_loss))
    # print('测试准确率:', sum(test_acc) / len(test_loss))
    # print('测试交并比:', sum(test_IOU) / len(test_IOU))
