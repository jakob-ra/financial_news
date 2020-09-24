from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn

df_test_pred = pd.read_pickle('/Users/Jakob/Documents/financial_news_data/df_test_pred.pkl')

LABEL_COLS = ['Deal', 'Pending', 'JointVenture', 'Marketing', 'Manufacturing', 'ResearchandDevelopment', 'Licensing', 'Supply', 'Exploration', 'TechnologyTransfer']
NUM_LABELS = len(LABEL_COLS)
fpr = dict()
tpr = dict()
thresholds = dict()
roc_auc = dict()
for label in range(NUM_LABELS):
    fpr[label], tpr[label], thresholds[label] = roc_curve(df_test_pred.labels.map(lambda x: x[label]),
        df_test_pred.raw_probs.map(lambda  x: x[label]))
    roc_auc[label] = auc(fpr[label], tpr[label])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
for label in range(NUM_LABELS):
    label_to_plot = label
    plt.plot(fpr[label_to_plot], tpr[label_to_plot],
             lw=lw, label=LABEL_COLS[label_to_plot] + ' (area = %0.2f)' % roc_auc[label_to_plot])
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.show()

tnr = dict()
optimal_thresholds = dict()
for label in range(NUM_LABELS):
    tnr[label] = 1-fpr[label]
    sens_plus_spec = tpr[label] + tnr[label]
    optimal_thresholds[label] = thresholds[label][np.argmax(sens_plus_spec)]

thresholds = optimal_thresholds
optimal_thresholds[3] = 0.95

df_test_pred['pred_adj'] = df_test_pred.raw_probs.map(lambda scores: [score > optimal_thresholds[i] for
    i,score in enumerate(scores)])

df_performance = pd.DataFrame(index=['Precision', 'Recall', 'F1-score', 'Accuracy', 'ROC_AUC'], columns=LABEL_COLS).T

for label in range(NUM_LABELS):
  if label == 0: # use all examples to evaluate the 'relevant deal or not' classification
    y_true, y_pred = df_test_pred.labels.map(lambda x: x[label]), df_test_pred.pred.map(lambda x: x[label])
    y_score = df_test_pred.raw_probs.map(lambda x: x[label])
  else: # use only the KB examples as evaluation for relationship classification
    y_true, y_pred = df_test_pred[df_test_pred.Deal == 1].labels.map(lambda x: x[label]), df_test_pred[df_test_pred.Deal == 1].pred.map(lambda x: x[label])
    y_score = df_test_pred[df_test_pred.Deal == 1].raw_probs.map(lambda x: x[label])
  precision = sklearn.metrics.precision_score(y_true, y_pred)
  recall = sklearn.metrics.recall_score(y_true, y_pred)
  f1_score = sklearn.metrics.f1_score(y_true, y_pred)
  accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
  roc_auc = sklearn.metrics.roc_auc_score(y_true, y_score)
  df_performance.loc[LABEL_COLS[label]] = [precision, recall, f1_score, accuracy, roc_auc]

df_performance