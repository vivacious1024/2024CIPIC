import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import joblib
import scipy.io
import xlrd
import FE

from grouping import group, screen
from featuremat import feature_save, feature_mat

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

    return new_X


if __name__=="__main__":
    # root = ''  # 数据根目录
    gp = 'F:\作业\本基项目\\val_group'  # 数据切割路径
    feature = 'F:\作业\本基项目\\val_featureMatrix'  # 特征路径
    # group(root)
    # screen(gp)
    # feature_save(gp)
    featureMatrix=feature_mat(feature)
    U = np.load('Ureduce.npy')
    test_data = featureMatrix * np.asmatrix(U).I
    score = get_score('人格特质结果val.xlsx')

    dirs = 'cvModel'
    model_name_ls = ['neuroticism', 'extraversion', 'openness', 'agreeableness', 'conscientiousness']
    # model_list = np.load('model_results_list.npy', allow_pickle=True)
    model_list = np.load('cv_model_list.npy', allow_pickle=True)
    #     {'model_name': model_name_ls[kind], 'optimal_model': optimal_model, 'optimal_feature':optimal_feature,
    #      'R2_train': R_2[0], 'R2_test': R_2[1], 'alpha': alpha_l1[0], 'l1_ratio': alpha_l1[1]}
    for kind in range(5):
        model = joblib.load(dirs + '/' + model_name_ls[kind] + '.pkl')  # 读取训练好的模型
        model_name = model_name_ls[kind]
        features = model_list[kind]['optimal_feature']

        y_val = score[:, kind]
        y_pred = model.predict(test_data[:, features])
        print(model_name, '评分: ', y_pred, '  r2: ', r2_score(y_val, y_pred))

