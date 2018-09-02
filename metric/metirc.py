from sklearn import metrics

def classification_acc(y_true,y_pre):
    '''
    Accuracy classification score.
    '''
    return metrics.accuracy_score(y_true, y_pred)



def classification_loss(y_true,y_pre):
    '''
    Log loss, aka logistic loss or cross-entropy loss.
    '''
    return metrics.log_loss(y_true, y_pred)



def regression_l1(y_true,y_pre):
    '''
    Mean absolute error regression loss
    '''
    return metrics.mean_absolute_error(y_true, y_pred)


def regression_l2(y_true,y_pre):
    '''
    Mean squared error regression loss
    '''
    return metrics.mean_squared_error(y_true, y_pred)
