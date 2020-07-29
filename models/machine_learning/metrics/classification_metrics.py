from sklearn import metrics

# ref: https://blog.csdn.net/Yqq19950707/article/details/90169913
def get_classification_metrics(y_pred, y_test):
    # 精度
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"acc score: {acc}")

    # auc, ROC曲线下的面积;较大的AUC代表了较好的performance。
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=2)
    auc = metrics.auc(fpr, tpr)
    print(f"auc: {auc}")

    # average precision score 根据预测得分计算平均精度(AP)
    # y_score = 
    # ap = metrics.average_precision_score(y_test, y_score)
    # print(f"average precision score: {ap}")

    # confusion matrix 通过计算混淆矩阵来评估分类的准确性 返回混淆矩阵
    mat = metrics.confusion_matrix(y_test, y_pred)
    print(f"confusion matrix: {mat}")

    # f1 score F1 = 2 * (precision * recall) / (precision + recall) precision(查准率)=TP/(TP+FP) recall(查全率)=TP/(TP+FN)
    f1 = metrics.f1_score(y_test, y_pred)
    print(f"f1 score: {f1}")

    # log_loss 对数损耗，又称逻辑损耗或交叉熵损耗
    log_loss = metrics.log_loss(y_test, y_pred)
    print(f"log_loss: {log_loss}")

    # precision score 查准率或者精度； precision(查准率)=TP/(TP+FP)
    precision = metrics.precision_score(y_test, y_pred)
    print(f"precision score: {precision}")

    # recall score 查全率 ；recall(查全率)=TP/(TP+FN)
    recall = metrics.recall_score(y_test, y_pred)
    print(f"recall score: {recall}")

    # roc_auc_score 计算ROC曲线下的面积就是AUC的值，the larger the better
    # y_score = 
    # roc_auc = metrics.roc_auc_score(y_test, y_score)
    # print(f"roc auc score: {roc_auc}")

    # roc_curve 计算ROC曲线的横纵坐标值，TPR，FPR. TPR = TP/(TP+FN) = recall(真正例率，敏感度) FPR = FP/(FP+TN)(假正例率，1-特异性)
    # y_score = 
    # roc_curve = metrics.roc_curve(y_test, y_score)
    # print(f"roc curve: {roc_curve}")

    # Cohen's kappa: a statistic that measures inter-annotator agreement.
    kappa = metrics.cohen_kappa_score(y_test, y_pred)
    print(f"kappa score: {kappa}")