def cal_metrics (all_targets, all_outputs):
  from sklearn import metrics
  all_targets = all_targets.detach().cpu().numpy()
  all_outputs = all_outputs.detach().cpu().numpy()

  acc = metrics.accuracy_score(all_targets, all_outputs)
  macro_precision = metrics.precision_score(all_targets, all_outputs, average = 'macro', zero_division=1)
  macro_recall = metrics.recall_score(all_targets, all_outputs, average = 'macro')
  macro_f1 = metrics.f1_score(all_targets, all_outputs, average = 'macro')

  return acc, macro_precision, macro_recall, macro_f1


def num_params(model):
  nums = sum(p.numel() for p in model.parameters())/1e6
  return nums

def num_trainable_params(model):
  nums = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6
  return nums

def calculate_metrics(predictions, targets):
    # Convert softmax predictions to class labels
    predicted_labels = torch.argmax(predictions, dim=1)

    # Calculate true positives, false positives, and false negatives
    true_positives = torch.sum((predicted_labels == 1) & (targets == 1)).item()
    false_positives = torch.sum((predicted_labels == 1) & (targets == 0)).item()
    false_negatives = torch.sum((predicted_labels == 0) & (targets == 1)).item()

    # Calculate precision
    precision = true_positives / (true_positives + false_positives + 1e-7)

    # Calculate recall
    recall = true_positives / (true_positives + false_negatives + 1e-7)

    # Calculate F1 score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

    return f1_score, precision, recall

from sklearn.metrics import confusion_matrix
def save_confusion_matrix(targets, predicted_labels, classes, save_path):
    predicted_labels = torch.argmax(predicted_labels, dim=1)
    cm = confusion_matrix(targets.cpu().numpy(), predicted_labels.cpu().numpy())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize confusion matrix

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Format and display the confusion matrix values
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt), ha='center', va='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, format='png')
    plt.close()
    # Calculate sensitivity and specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity

from sklearn import metrics
def plot_ROC(targets, predicted_labels, save_path):
  # predicted_labels = torch.argmax(predicted_labels, dim=1)
  fpr, tpr, _ = metrics.roc_curve(targets.cpu().numpy(),  predicted_labels[:,1].cpu().numpy())

  noskill_probabilities = [0 for number in range(len(targets.cpu().numpy()))]
  fprno, tprno, _ = metrics.roc_curve(targets.cpu().numpy(),  noskill_probabilities)
  #create ROC curve
  plt.plot(fprno,tprno,'b--')
  plt.plot(fpr,tpr,'r')
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.savefig(save_path, format='png')
  plt.close()
  return 0

def loss_fn_kd(outputs, labels, teacher_outputs, T, alpha):
  loss = F.kl_div(F.log_softmax(outputs/T, dim=1),
                  F.softmax(teacher_outputs/T, dim=1),
                  reduction='batchmean') * (alpha * T**2) + \
         F.cross_entropy(outputs, labels) * (1 - alpha)
  return loss