# -*- coding: UTF-8 -*-

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder

from numpy import vstack, array, nan
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import PolynomialFeatures

from numpy import log1p
from sklearn.preprocessing import FunctionTransformer

from sklearn.feature_selection import VarianceThreshold

from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr

from sklearn.feature_selection import chi2

from minepy import MINE

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# 导入IRIS数据集
iris = load_iris()

# 特征矩阵
# 包含4个特征（Sepal.Length（花萼长度）、Sepal.Width（花萼宽度）、Petal.Length（花瓣长度）、Petal.Width（花瓣宽度）），
# 特征值都为正浮点数，单位为厘米
#
print(iris.data)

# 目标向量
# 目标值为鸢尾花的分类: Iris Setosa（山鸢尾）、Iris Versicolour（杂色鸢尾），Iris Virginica（维吉尼亚鸢尾）
print(iris.target)

# 标准化，返回值为标准化后的数据
stand = StandardScaler().fit_transform(iris.data)
print(stand)

# 区间缩放，返回值为缩放到[0,1]区间的数据
mm = MinMaxScaler().fit_transform(iris.data)
print(mm)

# 标准化是依照特征矩阵的列处理数据，其通过求z-score的方法，将样本的特征值转换到同一量纲下。
# 归一化是依照特征矩阵的行处理数据，其目的在于样本向量在点乘运算或其他核函数计算相似性时，拥有统一的标准，也就是说都转化为“单位向量”
# 归一化，返回值为归一化后的数据
normal = Normalizer().fit_transform(iris.data)
print(normal)

# 二值化，阈值设置为3，返回值为二值化后的数据
bi = Binarizer(threshold=3).fit_transform(iris.data)
print(bi)

# 哑编码，对IRIS数据集的目标值，返回值为哑编码后的数据
# 由于IRIS数据集的特征皆为定量特征，故使用其目标值进行哑编码（实际上是不需要的）。
one = OneHotEncoder().fit_transform(iris.target.reshape(-1, 1))
print(one)

# 缺失值计算，返回值为计算缺失值后的数据
# 参数missing_value为缺失值的表示形式，默认为NaN
# 参数strategy为缺失值填充方式，默认为mean(均值)
miss = SimpleImputer().fit_transform(vstack((array([nan, nan, nan, nan]), iris.data)))
print(miss)

# 多项式转换,常见的数据变换有基于多项式的、基于指数函数的、基于对数函数的
# 参数degree为度，默认值为2
poly = PolynomialFeatures().fit_transform(iris.data)
print(poly)

# 基于单变元函数的数据变换可以使用一个统一的方式完成
func = FunctionTransformer(log1p).fit_transform(iris.data)
print(func)

# 特征选择 #
# 特征是否发散
# 特征与目标的相关性

# Filter #

# 方差选择法，返回值为特征选择后的数据,大于阈值的特质
# 参数threshold为方差的阈值
var = VarianceThreshold(threshold=3).fit_transform(iris.data)
print(var)

# 相关系数法，先要计算各个特征对目标值的相关系数以及相关系数的P值
# 选择K个最好的特征，返回选择特征后的数据
# 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，数组第i个特征的评分和P值。
# 在此定义为计算相关系数。
# 参数k为选择的特征个数
best = SelectKBest(lambda X, Y: array(list(map(lambda x: pearsonr(x, Y), X.T))).T[0], k=2).fit_transform(iris.data,
                                                                                                         iris.target)
print(best)

# 卡方检验
# 经典的卡方检验是检验定性自变量对定性因变量的相关性
# 不难发现，这个统计量的含义简而言之就是自变量对因变量的相关性
# 选择K个最好的特征，返回选择特征后的数据
select = SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)
print(select)


# 互信息法
# 经典的互信息也是评价定性自变量对定性因变量的相关性的
# 由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)


# 选择K个最好的特征，返回特征选择后的数据
kbest = SelectKBest(lambda X, Y: array(list(map(lambda x: mic(x, Y), X.T))).T[0], k=2).fit_transform(iris.data,
                                                                                                     iris.target)
print(kbest)

# Wrapper #

# 递归特征消除法
# 使用一个基模型来进行多轮训练，每轮训练后，消除若干权值系数的特征，再基于新的特征集进行下一轮训练。
# 递归特征消除法，返回特征选择后的数据
# 参数estimator为基模型
# 参数n_features_to_select为选择的特征个数
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)
print(rfe)

# Embedded #

# 基于惩罚项的特征选择法
# 使用带惩罚项的基模型，除了筛选出特征外，同时也进行了降维。
# 带L1惩罚的逻辑回归作为基模型的特征选择
l1 = SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target)
print(l1)


# 实际上，L1惩罚项降维的原理在于保留多个对目标值具有同等相关性的特征中的一个，所以没选到的特征不代表不重要
# 故，可结合L2惩罚项来优化
# 具体操作为：若一个特征在L1中的权值为1，选择在L2中权值差别不大且在L1中权值为0的特征构成同类集合，
# 将这一集合中的特征平分L1中的权值，故需要构建一个新的逻辑回归模型：

class LR(LogisticRegression):
    def __init__(self, threshold=0.1, dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):
        # 权值相近的阈值
        self.threshold = threshold

        LogisticRegression.__init__(self, penalty='l1', dual=dual, tol=tol, C=C,
                                    fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                    class_weight=class_weight, random_state=random_state, solver=solver,
                                    max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start,
                                    n_jobs=n_jobs)
        # 使用相同的参数创建L2逻辑回归
        self.l2 = LogisticRegression(penalty='l2', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
                                     intercept_scaling=intercept_scaling, class_weight=class_weight,
                                     random_state=random_state, solver=solver, max_iter=max_iter,
                                     multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
        # 训练L1逻辑回归
        super(LR, self).fit(X, y, sample_weight=sample_weight)
        self.coef_old_ = self.coef_.copy()
        # 训练L2逻辑回归
        self.l2.fit(X, y, sample_weight=sample_weight)

        cntOfRow, cntOfCol = self.coef_.shape
        # 权值系数矩阵的行数对应目标值的种类数据
        for i in range(cntOfRow):
            for j in range(cntOfCol):
                coef = self.coef_[i][j]
                # L1逻辑回归的权值系数不为0
                if coef != 0:
                    idx = [j]
                    # 对应在L2逻辑回归中的权值系数
                    coef1 = self.l2.coef_[i][j]
                    for k in range(cntOfCol):
                        coef2 = self.l2.coef_[i][k]
                        # 在L2逻辑回归中，权值系统之差小于设定的阈值，
                        # 且在L1中对应的权值为0
                        if abs(coef1 - coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                            idx.append(k)
                    # 计算这一类特征权值系数均值
                    mean = coef / len(idx)
                    self.coef_[i][idx] = mean
        return self


# 带L1和L2惩罚项的逻辑回归作为基模型的特征选择
# 参数threshold为权值系数之差的阈值
l2 = SelectFromModel(LR(threshold=0.5, C=0.1)).fit_transform(iris.data, iris.target)
print(l2)

# 基于树模型的特征选择法
# 树模型中GBDT也可用来作为基模型进行特征选择
# GBDT作为基模型的特征选择
gbdt = SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)
print(gbdt)

# 降维 #
# 当特征选择完成后，可以直接训练模型了，但是可能由于特征矩阵过大，导致计算量大，训练时间长的问题，因此降低特征矩阵维度是必不可少。
# 常见的降维方法：基于L1惩罚项的模型外
# 主成分分析法（PCA）：为了让映射后的样本具有最大的发散性
# 线性判别分析（LDA）：为了让映射后的样本有最好的分类性能
# 共同点：其本质是要讲原始的样本映射到维度更低的样本空间中

# 主成分分析法，返回降维后的数据
# 参数n_components为主成分数目
pca = PCA(n_components=2).fit_transform(iris.data)
print(pca)

# 线性判别分析法，返回降维后的数据
# 参数n_components为降维后的维数
lda = LDA(n_components=2).fit_transform(iris.data, iris.target)
print(lda)
