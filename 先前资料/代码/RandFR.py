import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import xlrd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split  # 这里是引用了交叉验证

import FE


def get_feature(path):
    """获得原始的特征矩阵"""
    data = scipy.io.loadmat(path)
    featureMatrix = FE.maxminnorm(np.array(data['feature']))

    return featureMatrix


def get_score(path):
    """获得人格特征评分"""
    data = xlrd.open_workbook(path)
    table = data.sheet_by_name('Sheet1')
    scoremat = np.array(table.col_values(1, 1))
    for i in range(2, 6):
        temp = np.array(table.col_values(i, 1))
        scoremat = np.vstack((scoremat, temp))

    return scoremat.T


def getPearson(featuremap_y):
    '''获得特征与目标值的皮尔逊系数'''
    corr = np.empty(featuremap_y.shape[1] - 1)
    for i in range(featuremap_y.shape[1] - 1):
        corr[i] = np.corrcoef(featuremap_y[:, [i, -1]], rowvar=False)[1, 0]

    return corr


def grid_cv_model(X, y, corrcoef):
    """使用嵌套交叉验证进行调整参数和模型选择"""
    # featureMatrix_score = np.hstack((new_X, score[:, 4].reshape(new_X.shape[0], 1)))
    # corrcoef = getPearson(featureMatrix_score)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=53)  # 选择20%为测试集
    print('训练集测试及参数:')
    print('X_train.shape={}\n y_train.shape ={}\n X_test.shape={}\n y_test.shape={}'.format(X_train.shape,
                                                                                            y_train.shape,
                                                                                            X_test.shape,
                                                                                            y_test.shape))

    # corr_sort_index = np.argsort(np.abs(corrcoef))[::-1]
    max_score = 0
    optimal_model=RandomForestRegressor()
    model = RandomForestRegressor(random_state=34)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(score)
    if score < max_score:
        max_score = score
        optimal_model = model

    # 做ROC曲线
    y_pred = optimal_model.predict(X_test)
    r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
    print(r2)
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, label="predict")
    plt.plot(range(len(y_pred)), y_test,'.', label="test")
    plt.legend(loc="upper right")  # 显示图中的标签
    plt.xlabel("the number of subjects")
    plt.ylabel('value of personality traits')
    plt.title('')
    plt.show()

    return optimal_model, max_score


if __name__ == "__main__":
    featureMatrix = get_feature('F:\作业\本基项目\\feature_mat.mat')
    # 标准化
    featureMatrix = (featureMatrix - featureMatrix.mean(axis=0)) / featureMatrix.std(axis=0)
    print(featureMatrix.shape)
    pca = PCA(copy=True, n_components=0.7, whiten=False, svd_solver='auto')
    pca.fit(featureMatrix)

    # print(pca.explained_variance_ratio_)
    # print(pca.explained_variance_)
    # print(pca.components_)
    new_X = pca.transform(featureMatrix)
    print(new_X.shape)

    score = get_score('F:\作业\本基项目\人格特质结果.xlsx')
    print(score.shape)

    featureMatrix_score = np.hstack((featureMatrix, score))
    optimal_model, R_2 = grid_cv_model(new_X, score, getPearson(featureMatrix_score))
