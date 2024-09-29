import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import scipy.stats as sci
import joblib
import scipy.io
import xlrd
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


def myPCA(featureMatrix, n_components):
    pca = PCA(copy=True, n_components=n_components, whiten=True, svd_solver='full')
    pca.fit(featureMatrix)
    new_X = pca.transform(featureMatrix)
    Ureduce = pca.components_
    np.save('Ureduce.npy', Ureduce)

    return new_X


if __name__ == "__main__":
    featureMatrix = get_feature('F:\作业\本基项目\\feature_mat.mat')
    # featureMatrix = (featureMatrix - featureMatrix.mean(axis=0)) / featureMatrix.std(axis=0)
    # featureMatrix = (featureMatrix-featureMatrix.min(axis=0))/(featureMatrix.max(axis=0)-featureMatrix.min(axis=0))
    new_X = myPCA(featureMatrix, 0.99)
    print(new_X.shape)
    score = get_score('F:\作业\本基项目\人格特质结果.xlsx')

    # dirs = 'testModel'
    dirs = 'cvModel'
    model_name_ls = ['neuroticism', 'extraversion', 'openness', 'agreeableness', 'conscientiousness']
    # model_list = np.load('model_results_list.npy', allow_pickle=True)
    model_list = np.load('cv_model_list.npy', allow_pickle=True)
    #     {'model_name': model_name_ls[kind], 'optimal_model': optimal_model, 'optimal_feature':optimal_feature,
    #      'R2_train': R_2[0], 'R2_test': R_2[1], 'alpha': alpha_l1[0], 'l1_ratio': alpha_l1[1]}
    print('-----------------------最佳模型----------------------------')
    for kind in range(5):
        model = joblib.load(dirs + '/' + model_name_ls[kind] + '.pkl')  # 读取训练好的模型
        model_name = model_name_ls[kind]
        features = model_list[kind]['optimal_feature']
        R_2 = [model_list[kind]['R2_train'], model_list[kind]['R2_test']]
        alpha_l1 = [model_list[kind]['alpha'], model_list[kind]['l1_ratio']]
        print('model_name:', model_name_ls[kind])
        print(model)
        print('特征数：', len(features))
        print(features)
        print("训练集R^2(使用留一法交叉验证时为均方根值):", R_2[0], "测试集R^2:", R_2[1], "\nalpha:", alpha_l1[0], "  l1_ratio:", alpha_l1[1])

        y_test = score[:, kind]
        y_pred = model.predict(new_X[:, features])
        # plt.subplot(3,2,kind+1)
        # plt.plot(y_test,y_pred,'g.')
        final_R_2 = r2_score(y_test, y_pred)
        print('R_2:', final_R_2)
        r, p = sci.pearsonr(y_test, y_pred)
        print('r: %.2f\np: %e' % (r, p))

        # # 设置风格
        # sns.set_style('white')
        df = pd.DataFrame({'self-reported score': y_test, 'EEG-predicted score': y_pred})
        # g = sns.pairplot(df, vars=['y_test', 'y_pred'],
        #                  kind='reg', diag_kind='kde', palette='husl')

        # 画图
        sns.jointplot(x=df['self-reported score'], y=df['EEG-predicted score'],  # 设置xy轴，显示columns名称
                      data=df,  # 设置数据
                      color='g',  # 设置颜色
                      # s=50, edgecolor='w', linewidth=1,  # 设置散点大小、边缘颜色及宽度(只针对scatter)
                      kind='reg',  # 设置类型：'scatter','reg','resid','kde','hex'
                      # stat_func=<function pearsonr>,
                      space=0.1,  # 设置散点图和布局图的间距
                      height=8,  # 图表大小(自动调整为正方形))
                      ratio=5,  # 散点图与布局图高度比，整型
                      marginal_kws=dict(bins=15, rug=True),  # 设置柱状图箱数，是否设置rug
                      )

    plt.show()
