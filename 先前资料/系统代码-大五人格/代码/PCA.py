import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import xlrd
from sklearn import metrics  # 用于模型评估
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, ElasticNet  # 线性回归
from sklearn.model_selection import train_test_split, GridSearchCV  # 这里是引用了交叉验证
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import time
from sklearn import metrics
import csv
from sklearn.svm import SVR
import matplotlib.pyplot as plt

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


def build_lr(X, y):
    """岭回归"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=53)  # 选择20%为测试集
    print('训练集测试及参数:')
    print('X_train.shape={}\n y_train.shape ={}\n X_test.shape={}\n y_test.shape={}'.format(X_train.shape,
                                                                                            y_train.shape,
                                                                                            X_test.shape,
                                                                                            y_test.shape))
    linreg = Ridge()
    # 训练
    model = linreg.fit(X_train, y_train)
    print('模型参数:')
    print(model)
    # 训练后模型截距
    print('模型截距:')
    print(linreg.intercept_)
    # 训练后模型权重（特征个数无变化）
    print('参数权重:')
    print(linreg.coef_)
    print("训练集")
    y_train_pred = linreg.predict(X_train)
    sum_mean, ssr, sse = 0, 0, 0
    for i in range(len(y_train_pred)):
        sum_mean += (y_train_pred[i] - y_train[i]) ** 2
        ssr += (y_train_pred[i] - np.mean(y_train)) ** 2
        sse += (y_train_pred[i] - y_train[i]) ** 2
    sum_erro = np.sqrt(sum_mean / len(y_train_pred))  # 测试级的数量
    # calculate RMSE, SSR, SSE, R^2
    R_2 = ssr / (ssr + sse)
    adj_R_2 = 1 - ((1 - R_2) * ((len(y_train_pred) - 1) / (len(y_train_pred) - X_train.shape[1] - 1)))
    print("RMSE by hand:", sum_erro)
    print("SSR :", ssr, "\nSSE :", sse, "\nR^2 :", R_2, "\nadjust R^2 :", adj_R_2)
    # 做ROC曲线
    plt.figure()
    plt.plot(range(len(y_train_pred)), y_train_pred, 'b', label="predict")
    plt.plot(range(len(y_train_pred)), y_train, 'r.', label="test")
    plt.legend(loc="upper right")  # 显示图中的标签
    plt.xlabel("the number of sales")
    plt.ylabel('value of sales')
    plt.show()

    print("测试集")
    y_pred = linreg.predict(X_test)
    sum_mean, ssr, sse = 0, 0, 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - y_test[i]) ** 2
        ssr += (y_pred[i] - np.mean(y_test)) ** 2
        sse += (y_pred[i] - y_test[i]) ** 2
    sum_erro = np.sqrt(sum_mean / len(y_pred))  # 测试级的数量
    # calculate RMSE, SSR, SSE, R^2
    R_2 = ssr / (ssr + sse)
    adj_R_2 = 1 - ((1 - R_2) * ((len(y_pred) - 1) / (len(y_pred) - X_test.shape[1] - 1)))
    print("RMSE by hand:", sum_erro)
    print("SSR :", ssr, "\nSSE :", sse, "\nR^2 :", R_2, "\nadjust R^2 :", adj_R_2)

    # 做ROC曲线
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    plt.plot(range(len(y_pred)), y_test, 'r.', label="test")
    plt.legend(loc="upper right")  # 显示图中的标签
    plt.xlabel("the number of sales")
    plt.ylabel('value of sales')
    plt.show()


def nested_cv_model(X, y, corrcoef):
    """使用嵌套交叉验证进行调整参数和模型选择"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=53)  # 选择20%为测试集
    print('训练集测试及参数:')
    print('X_train.shape={}\n y_train.shape ={}\n X_test.shape={}\n y_test.shape={}'.format(X_train.shape,
                                                                                            y_train.shape,
                                                                                            X_test.shape,
                                                                                            y_test.shape))

    # nested-LOOCV 嵌套交叉验证
    alpha_ls = [x / 10.0 for x in range(0, 11)]
    l1_ratio_ls = [x / 10.0 for x in range(0, 11)]
    corr_sort_index = np.argsort(np.abs(corrcoef))[::-1]
    max_R2 = 0
    max_scores = 0
    for alpha in alpha_ls:
        for l1_ratio in l1_ratio_ls:
            for corr_sum in range(1, corrcoef.shape[0]):
                print('-------alpha:%f-----l1_ratio:%f-----corr_sum:%d--------' % (alpha, l1_ratio, corr_sum))
                linreg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
                # 训练
                model = linreg.fit(X_train[:, corr_sort_index[0:corr_sum]], y_train)
                print('模型参数:')
                print(model)
                # 训练后模型截距
                print('模型截距:')
                print(linreg.intercept_)
                # 训练后模型权重（特征个数无变化）
                print('参数权重:')
                print(linreg.coef_)
                print("训练集")
                y_train_pred = linreg.predict(X_train[:, corr_sort_index[0:corr_sum]])
                sum_mean, ssr, sse = 0, 0, 0
                for i in range(len(y_train_pred)):
                    sum_mean += (y_train_pred[i] - y_train[i]) ** 2
                    ssr += (y_train_pred[i] - np.mean(y_train)) ** 2
                    sse += (y_train_pred[i] - y_train[i]) ** 2
                sum_erro = np.sqrt(sum_mean / len(y_train_pred))  # 测试级的数量
                # calculate RMSE, SSR, SSE, R^2
                R_2 = ssr / (ssr + sse)
                adj_R_2 = 1 - ((1 - R_2) * ((len(y_train_pred) - 1) / (len(y_train_pred) - X_train.shape[1] - 1)))
                print("RMSE by hand:", sum_erro)
                print("SSR :", ssr, "\nSSE :", sse, "\nR^2 :", R_2, "\nadjust R^2 :", adj_R_2)
                R_2_train = R_2

                print("测试集")
                y_pred = linreg.predict(X_test[:, corr_sort_index[0:corr_sum]])
                sum_mean, ssr, sse = 0, 0, 0
                for i in range(len(y_pred)):
                    sum_mean += (y_pred[i] - y_test[i]) ** 2
                    ssr += (y_pred[i] - np.mean(y_test)) ** 2
                    sse += (y_pred[i] - y_test[i]) ** 2
                sum_erro = np.sqrt(sum_mean / len(y_pred))  # 测试级的数量
                # calculate RMSE, SSR, SSE, R^2
                R_2 = ssr / (ssr + sse)
                adj_R_2 = 1 - ((1 - R_2) * ((len(y_pred) - 1) / (len(y_pred) - X_test.shape[1] - 1)))
                print("RMSE by hand:", sum_erro)
                print("SSR :", ssr, "\nSSE :", sse, "\nR^2 :", R_2, "\nadjust R^2 :", adj_R_2)
                R_2_test = R_2

                # scores = np.sqrt(-cross_val_score(linreg,X[:,corr_sort_index[0:corr_sum]],y,cv=10)).mean()
                if R_2 > max_R2:
                    optimal_model = linreg  # 最优模型
                    optimal_feature = corr_sort_index[0:corr_sum]  # 最优特征
                    max_R2 = R_2  # 最大测试集R^2
                    final_R2_train, final_R2_test = R_2_train, R_2_test  # 最优模型的训练集和测试集R^2
                    optimal_alpha_l1 = [alpha, l1_ratio]  # 最优参数

    return optimal_model, optimal_feature, [final_R2_train, final_R2_test], optimal_alpha_l1


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

    alpha_ls = [x / 10.0 for x in range(0, 11)]
    l1_ratio_ls = [x / 10.0 for x in range(0, 11)]
    corr_sort_index = np.argsort(np.abs(corrcoef))[::-1]
    cv_scores = 0
    R2_best = 0
    max_scores = 0
    optimal_feature = []
    optimal_alpha_l1 = []
    linreg_best_final = ElasticNet()
    for corr_sum in range(1, corrcoef.shape[0]):
        param_grid = dict(alpha=alpha_ls, l1_ratio=l1_ratio_ls)
        linreg = ElasticNet(max_iter=1e04,tol=1e-4)
        grid = GridSearchCV(linreg, param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=8)
        grid.fit(X_train[:, corr_sort_index[0:corr_sum]], y_train)
        # 返回最佳参数组合
        print('Best：%f using %s' % (grid.best_score_, grid.best_params_))

        print('测试集测试及参数:')

        linreg_best = grid.best_estimator_
        print(linreg_best, linreg_best.__dict__)
        y_pred = linreg_best.predict(X_test[:, corr_sort_index[0:corr_sum]])
        R2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
        print('模型预测的R2：{0}'.format(R2))
        if R2 > R2_best:
            linreg_best_final = linreg_best
            optimal_feature = corr_sort_index[0:corr_sum]  # 最优特征
            print(optimal_feature)
            cv_scores = grid.best_score_
            R2_best = R2
            optimal_alpha_l1 = [grid.best_params_['alpha'], grid.best_params_['l1_ratio']]

    # # 做ROC曲线
    # y_pred = linreg_best_final.predict(X_test[:, corr_sort_index[0:corr_sum]])
    # plt.figure()
    # plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
    # plt.plot(range(len(y_pred)), y_test, 'r.', label="test")
    # plt.legend(loc="upper right")  # 显示图中的标签
    # plt.xlabel("the number of subjects")
    # plt.ylabel('value of personality traits')
    # plt.title('')
    # plt.show()

    return linreg_best_final, optimal_feature, [cv_scores,R2_best], optimal_alpha_l1


def model_train_nested(new_X, score):
    '''分别进行交叉验证测试和无交叉验证测试'''
    # dirs = 'testModel'
    dirs = 'cvModel'
    model_name_ls = ['neuroticism', 'extraversion', 'openness', 'agreeableness', 'conscientiousness']
    model_list = []
    for kind in range(5):
        print('--------' + model_name_ls[kind] + ' model---------')
        featureMatrix_score = np.hstack((new_X, score[:, kind].reshape(new_X.shape[0], 1)))
        # optimal_model, optimal_feature, R_2, alpha_l1 = nested_cv_model(new_X, score[:, kind],
        #                                                                        getPearson(featureMatrix_score))
        optimal_model, optimal_feature, R_2, alpha_l1 = grid_cv_model(new_X, score[:, kind],
                                                                      getPearson(featureMatrix_score))
        model_list.append(
            {'model_name': model_name_ls[kind], 'optimal_model': optimal_model, 'optimal_feature': optimal_feature,
             'R2_train': R_2[0], 'R2_test': R_2[1], 'alpha': alpha_l1[0], 'l1_ratio': alpha_l1[1]})

        # 保存模型
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        joblib.dump(optimal_model, dirs + '/' + model_name_ls[kind] + '.pkl')
    # 存储训练结果数组
    # np.save('model_results_list.npy', np.array(model_list))
    np.save('cv_model_list.npy', np.array(model_list))

    print('\n')
    print('-----------------------最佳模型----------------------------')
    for kind in range(5):
        model_name = model_list[kind]['model_name']
        optimal_model = model_list[kind]['optimal_model']
        optimal_feature = model_list[kind]['optimal_feature']
        R_2 = [model_list[kind]['R2_train'], model_list[kind]['R2_test']]
        alpha_l1 = [model_list[kind]['alpha'], model_list[kind]['l1_ratio']]
        print('model_name:', model_name)
        print(optimal_model)
        print('特征数：', len(optimal_feature))
        print("训练集R^2(使用留一法交叉验证时为均方根值):", R_2[0], "测试集R^2:", R_2[1], "\nalpha:", alpha_l1[0], "  l1_ratio:", alpha_l1[1])
        print()

        y_test = score[:, kind]
        y_pred = optimal_model.predict(new_X[:, optimal_feature])
        # 做ROC曲线
        plt.figure(kind)
        plt.plot(range(len(y_pred)), y_pred, 'b', label="predict")
        plt.plot(range(len(y_pred)), y_test, 'r.', label="test")
        plt.legend(loc="upper right")  # 显示图中的标签
        plt.xlabel("the number of subjects")
        plt.ylabel('value of personality traits')
        plt.title(model_name)
    plt.show()


def svr_model(new_X, score):
    # feature_test = new_X[int(len(new_X) * 0.9):int(len(new_X))]
    # target_test = score[int(len(score) * 0.9):int(len(score))]
    feature_train, feature_test, target_train, target_test = train_test_split(new_X, score,
                                                                              test_size=0.3, random_state=10)

    start1 = time.time()
    model_svr = SVR(C=0.26944241, epsilon=0.5919577, gamma=24.95089257)
    model_svr.fit(feature_train, target_train)
    predict_results1 = model_svr.predict(feature_test)
    end1 = time.time()

    plt.plot(target_test)  # 测试数组
    plt.plot(predict_results1)  # 测试数组
    plt.legend(['True', 'SVR'])
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.title("SVR")  # 标题
    plt.show()
    print("EVS:", explained_variance_score(target_test, predict_results1))
    print("R2:", metrics.r2_score(target_test, predict_results1))
    print("Time:", end1 - start1)


if __name__ == "__main__":
    featureMatrix = get_feature('F:\作业\本基项目\\feature_mat.mat')
    # featureMatrix = (featureMatrix)/(featureMatrix.max(axis=0)-featureMatrix.min(axis=0))
    # featureMatrix = (featureMatrix - featureMatrix.min(axis=0)) / (
    #             featureMatrix.max(axis=0) - featureMatrix.min(axis=0))
    pca = PCA(copy=True, n_components=0.99, whiten=True, svd_solver='full')
    pca.fit(featureMatrix)
    new_X = pca.transform(featureMatrix)

    print(new_X.shape)

    score = get_score('F:\作业\本基项目\人格特质结果.xlsx')
    print(score.shape)

    # 训练模型
    model_train_nested(new_X,score)
    # featureMatrix_score = np.hstack((new_X, score[:, 0].reshape(new_X.shape[0], 1)))
    # pear = getPearson(featureMatrix_score)
    # arg_pear = np.argsort(pear)
    # print(arg_pear)
    #
    # optimal_model, optimal_feature, R_2, alpha_l1 = grid_cv_model(new_X, score[:, 0],pear)
    # print(optimal_feature)

