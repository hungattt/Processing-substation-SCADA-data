import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import hdbscan
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from pandasgui import show

class PreprocessingDataMissing:
    def __init__(self, path):
        self.path = path
        self.data = pd.read_feather(path)
        self.df_name = self.data.columns.tolist()

        plt.figure(figsize=(15, 15))
        plt.title("Data before removing missing", fontsize=16, fontweight='bold')
        sns.heatmap(self.data.isna(), cbar=False, cmap='viridis', yticklabels=False)
        plt.savefig('output/original_data.jpg')
        plt.show()
        print(self.data)

    def mean_missing_data(self):  # Cách 1 : thay thế bằng giá trị trung bình
        X = self.data.iloc[:, :].values
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        imputer.fit(X[:, 2:])
        X[:, 2:] = imputer.transform(X[:, 2:])
        a = pd.DataFrame(X[:, :])
        a.columns = self.df_name
        return a

    def delete_missing_data(self):  # Cách xóa dòng giá trị bị trống
        return self.data.dropna()  # inplace=True thay đổi trực tiếp trên dataframe gốc

    def linear_missing_data(self):  # nội suy các giá trị còn thiếu bằng phương pháp Tuyến tính
        return self.data.interpolate(method='linear', limit_direction='forward')  # backward

    def KNN_missing_data(self):  # KNN Imputer để ước tính dữ liệu bị thiếu dựa trên K điểm dữ liệu gần nhất.
        X = self.data.iloc[:, :].values
        preprocessor = KNNImputer(n_neighbors=5, weights="distance")
        preprocessor.fit(X[:, 2:])
        X[:, 2:] = preprocessor.transform(X[:, 2:])
        b = pd.DataFrame(X[:, :])
        b.columns = self.df_name
        return b


class PreprocessingDataOutliers:
    def __init__(self, data):
        self.data = data

    def iqr_outliers_data(self):

        for col in self.data.columns.tolist()[2:]:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR

            outlier_list_col = self.data[(self.data[col] < Q1 - outlier_step) | (self.data[col] > Q3 + outlier_step)]
            if len(outlier_list_col) > 0:
                self.data[col] = np.where(self.data[col] > Q3 + outlier_step, Q3 + outlier_step,
                                          np.where(
                                              self.data[col] < Q1 - outlier_step, Q1 - outlier_step, self.data[col]
                                          )
                                          )

        return self.data

    def zscore_outliers_data(self, alpha=3):
        for col in self.data.columns[2:]:
            upper_limit = self.data[col].mean() + alpha * self.data[col].std()
            lower_limit = self.data[col].mean() - alpha * self.data[col].std()
            outlier_list_col = self.data[(self.data[col] > upper_limit) | (self.data[col] < lower_limit)]

            if len(outlier_list_col) > 0:
                self.data[col] = np.where(self.data[col] > upper_limit, upper_limit,
                                          np.where(
                                              self.data[col] < lower_limit, lower_limit, self.data[col]
                                          )
                                          )
        return self.data

    def percentile_outliers_data(self, threshold_up=0.99, threshold_low=0.01):
        for col in self.data.columns[2:]:
            upper_limit = self.data[col].quantile(threshold_up)
            lower_limit = self.data[col].quantile(threshold_low)
            outlier_list_col = self.data[(self.data[col] >= upper_limit) | (self.data[col] <= lower_limit)]

            if len(outlier_list_col) > 0:
                self.data[col] = np.where(self.data[col] >= upper_limit, upper_limit,
                                          np.where(self.data[col] <= lower_limit, lower_limit, self.data[col]))

        return self.data

    def hdbscan_outliers_data(self, cluster_num=5):
        for col in self.data.columns.tolist()[2:]:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.data[col].values.reshape(-1, 1))

            clusterer = hdbscan.HDBSCAN(min_cluster_size=cluster_num, gen_min_span_tree=True)
            clusterer.fit(scaled_data)

            # Tạo một bản sao của dữ liệu và đánh dấu giá trị ngoại lệ là NaN
            copy_df = self.data.copy()

            copy_df.loc[clusterer.labels_ == -1, col] = np.nan

            # Thực hiện nội suy tuyến tính
            self.data[col] = copy_df[col].interpolate(method='linear', limit_direction='forward')
            self.data[col] = self.data[col].interpolate(method='linear', limit_direction='backward')
        return self.data

    """
    n_neighbors: Số lượng hàng xóm được xem xét trong việc tính toán độ đo LOF. Giá trị mặc định là 20. Một giá trị cao hơn
    có thể làm giảm nhiễu, nhưng cũng có thể làm mờ điểm ngoại lệ thực sự.

    contamination: Tỷ lệ các điểm ngoại lệ trong dữ liệu. Nó được sử dụng để xác định ngưỡng cho quyết định outlier.
    Mặc định là 'auto', nghĩa là LOF sẽ tự động xác định ngưỡng dựa trên dữ liệu. Nhưng bạn cũng có thể đặt nó thành một 
    giá trị thực trong khoảng từ 0 đến 0.5, với 0.1 có nghĩa là 10% dữ liệu được cho là ngoại lệ.
    """
    def lof_outliers_data(self, n_num=5):  # Local Outlier Factor (LOF)
        # Sử dụng LOF để tìm ngoại lệ từng cột và thay thế giá trị ngoại lệ bằng cách nội suy
        for col in self.data.columns.tolist()[2:]:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.data[col].values.reshape(-1, 1))

            lof = LocalOutlierFactor(n_neighbors=n_num)  # ,contamination=0.1
            outliers = lof.fit_predict(scaled_data)

            # Tạo một bản sao của dữ liệu và đánh dấu giá trị ngoại lệ là NaN
            copy_df = self.data.copy()
            copy_df.loc[outliers == -1, col] = np.nan

            # Thực hiện nội suy tuyến tính
            self.data[col] = copy_df[col].interpolate(method='linear', limit_direction='forward')
            self.data[col] = self.data[col].interpolate(method='linear', limit_direction='backward')
        return self.data

    """
    nu=0.1: Đây là một tham số giữa 0 và 1, biểu thị tỷ lệ ước lượng của ngoại lệ so với toàn bộ tập dữ liệu.
     Ở đây, nó được đặt thành 0.1, có nghĩa là 10% số lượng dữ liệu được giả định là ngoại lệ.

    kernel="rbf": Kernel được sử dụng trong mô hình. rbf là viết tắt của Radial basis function, là một hàm nhân phổ biến 
    được sử dụng trong thuật toán SVM. Các lựa chọn khác có thể là 'linear', 'poly', 'sigmoid', v.v.

    gamma=0.1: Tham số gamma cho hàm nhân RBF, Poly và Sigmoid. Nó cung cấp mức độ ảnh hưởng của một mẫu đơn lẻ. 
    Một gamma thấp có nghĩa là mức ảnh hưởng lớn, và ngược lại.
    """

    def onesvm_outliers_data(self):  # One-Class SVM
        # Sử dụng One-Class SVM để tìm outlier
        ocsvm = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        for col in self.data.columns.tolist()[2:]:
            # Chuẩn hóa dữ liệu
            data = self.data[col].values.reshape(-1, 1)
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            # Dự đoán outliers
            preds = ocsvm.fit_predict(data)
            # Tạo một bản sao của dữ liệu và đánh dấu giá trị ngoại lệ là NaN
            copy_df = self.data.copy()
            copy_df.loc[preds == -1, col] = np.nan

            # Thực hiện nội suy tuyến tính
            self.data[col] = copy_df[col].interpolate(method='linear', limit_direction='forward')
            self.data[col] = self.data[col].interpolate(method='linear', limit_direction='backward')
        return self.data

    """
    Từng tham số trong hàm IsolationForest() có ý nghĩa như sau:

    contamination: Đây là một tham số quan trọng chỉ định tỷ lệ các điểm ngoại lệ trong tập dữ liệu. 
    Trong trường hợp này, nó được đặt là 12% (tức là 0.12). Nếu bạn có kiến thức trước về tỷ lệ điểm ngoại lệ trong tập 
    dữ liệu của mình, đặt tham số này có thể giúp cải thiện hiệu suất của mô hình.

    max_samples: Đây là số lượng mẫu tối đa từ tập dữ liệu mà mỗi cây trong rừng cô lập sẽ sử dụng để phát triển. 
    'auto' có nghĩa là mỗi cây sẽ sử dụng tất cả các mẫu có sẵn.

    random_state: Tham số này xác định cách các số ngẫu nhiên được tạo. Đây cũng là cách để đảm bảo rằng kết quả có thể tái 
    tạo được - nghĩa là, mỗi lần bạn chạy mô hình với cùng một trạng thái ngẫu nhiên, kết quả sẽ giống hệt nhau. 
    Trạng thái ngẫu nhiên có thể là bất kỳ số nguyên nào. Trong trường hợp này, random_state được thiết lập là 42.

    Isolation Forest là một phương pháp phát hiện ngoại lệ dựa trên cây quyết định. Ý tưởng chính của nó là các điểm ngoại 
    lệ có thể được "cô lập" bằng ít điều kiện hơn so với điểm bình thường. "+- Nó tạo ra nhiều cây quyết định ngẫu nhiên và 
    sau đó dựa vào kết quả để xác định một thực thể có phải là ngoại lệ hay không
    """

    def isolationForest_outliers_data(self):  # One-Class SVM
        # Sử dụng Isolation Forest để tìm outlier
        # iforest = IsolationForest()  # contamination=0.1
        iforest = IsolationForest(contamination=0.09, max_samples='auto', random_state=42)
        for col in self.data.columns.tolist()[2:]:
            # Chuẩn hóa dữ liệ
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.data[col].values.reshape(-1, 1))

            # Dự đoán outliers
            preds = iforest.fit_predict(scaled_data)

            # Tạo một bản sao của dữ liệu và đánh dấu giá trị ngoại lệ là NaN
            copy_df = self.data.copy()
            copy_df.loc[preds == -1, col] = np.nan

            # Thực hiện nội suy tuyến tính
            self.data[col] = copy_df[col].interpolate(method='linear', limit_direction='forward')
            self.data[col] = self.data[col].interpolate(method='linear', limit_direction='backward')
        return self.data


"""
Chuẩn hóa dữ liệu

- Tại sao phải chuẩn hóa dữ liệu: Các giá trị x tại các tỉ lệ khác nhau (scale) đóng góp khác nhau cho mô hình.

- Ví dụ tiền lương tính bằng triệu nhưng tuổi tác tính bằng chục tuy nhiên trong model hiện tại ta ưu tiên cân bằng 
sự đóng góp của 2 giá trị này.
"""


class NormalizationData:
    def __init__(self, data):
        self.data = data
    """
    MinMaxScaler
    MinMaxScaler thường được sử dụng để đưa các dữ liệu về giá trị nằm trong đoạn [0, 1].

    - Điểm yếu: phụ thuộc vào các outliners (Giá trị lớn đột biến)
    """

    def minmaxsaler_data(self):
        min_max_scaler = MinMaxScaler()
        x = self.data[self.data.columns.tolist()[2:]].values

        x_scaled = min_max_scaler.fit_transform(x)

        normalized_features = pd.DataFrame(x_scaled, columns=self.data.columns.tolist()[2:], index=self.data.index)

        self.data[self.data.columns.tolist()[2:]] = normalized_features

        return self.data#[self.data.columns.tolist()[2:]]

    """
    StandardScaler
    Chuẩn hóa dữ liệu về phân phối chuẩn đơn vị có trung bình bằng 0 và độ lệch chuẩn bằng 1.

    - Chuẩn hóa dữ liệu dạng này thường được sử dụng với dữ liệu theo hoặc gần theo phân phối chuẩn.
    """

    def standardscaler_data(self):
        standard_scaler = StandardScaler()

        x = self.data[self.data.columns.tolist()[2:]].values

        x_scaled = standard_scaler.fit_transform(x)

        standardized_features = pd.DataFrame(x_scaled, columns=self.data.columns.tolist()[2:], index=self.data.index)

        self.data[self.data.columns.tolist()[2:]] = standardized_features
        return self.data#[self.data.columns.tolist()[2:]]

    """
    Chuẩn hóa Normalize là một phương pháp chuẩn hóa dữ liệu, trong đó các giá trị được chuẩn hóa sao cho chúng có cùng 
    độ lớn (length hoặc norm).

    Cụ thể, Normalize sẽ thực hiện các bước sau:

    Tính toán norm (độ lớn) của mỗi điểm dữ liệu, thường sử dụng norm Euclidean.

    Chia mỗi giá trị trong điểm dữ liệu đó cho norm tương ứng của nó.

    Kết quả là mỗi điểm dữ liệu sau Normalize sẽ có độ lớn bằng 1.
    """

    def normalize_data(self):
        normalizer_scaler = Normalizer()
        x = self.data[self.data.columns.tolist()[2:]].values
        x_scaled = normalizer_scaler.fit_transform(x)
        normalizer_features = pd.DataFrame(x_scaled, columns=self.data.columns.tolist()[2:], index=self.data.index)
        self.data[self.data.columns.tolist()[2:]] = normalizer_features
        return self.data#[self.data.columns.tolist()[2:]]


def set_labels_data(path, columns_to_check):
    data = pd.read_feather(path)

    dataset = data.interpolate(method='linear', limit_direction='forward')

    for col in dataset.columns.tolist()[2:]:
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR

        outlier_list_col = dataset[(dataset[col] < Q1 - outlier_step) | (dataset[col] > Q3 + outlier_step)]
        if len(outlier_list_col) > 0:

            dataset[col] = np.where(dataset[col] > Q3 + outlier_step, np.nan,
                                      np.where(
                                          dataset[col] < Q1 - outlier_step, np.nan, dataset[col]
                                      )
                                      )

    dataset['labels'] = np.where(dataset[columns_to_check].apply(lambda x: x.isnull().any(), axis=1), 1, 0)

    return dataset

def draw_outliers_data(data, title):
    plt.figure(figsize=(20, 50))  # (20, 50)
    plt.suptitle(f"{title}", fontsize=16, fontweight='bold')
    for idx, name in enumerate(data.columns.tolist()[2:], start=1):
        # distplot
        plt.subplot(14, 2, 2 * idx - 1)
        sns.histplot(data[f'{name}'])

        # boxplot
        plt.subplot(14, 2, 2 * idx)
        sns.boxplot(data[f'{name}'])

    plt.savefig(f'output/{title}.jpg')
    plt.show()


def draw_data_missing(data):
    plt.figure(figsize=(15, 15))
    plt.title("Data after removing missing", fontsize=16, fontweight='bold')
    sns.heatmap(data.isna(), cbar=False, cmap='viridis', yticklabels=False)
    plt.savefig('output/removing_missing.jpg')
    plt.show()


def draw_data_normalization(data):
    plt.figure(figsize=(10, 50))
    plt.suptitle("Data normalization", fontsize=16, fontweight='bold')
    for idx, name in enumerate(data.columns.tolist()[2:-1], start=1):
        # distplot #histplot
        plt.subplot(14, 1, idx)
        sns.histplot(data[f'{name}'])
        # data[f'{name}'].hist()
    plt.savefig('output/data_normalization.jpg')
    plt.show()


if __name__ == "__main__":
    path = 'D:/Data_Cadar_dienluc/data_test/Instant_MaiDong_All.feather'
    Importer = PreprocessingDataMissing(path)
    data = Importer.linear_missing_data()
    draw_data_missing(data)

    draw_outliers_data(data, 'Data before removing outliers')
    Outliers = PreprocessingDataOutliers(data)
    data_not_outliers = Outliers.iqr_outliers_data()
    draw_outliers_data(data_not_outliers, 'Data after removing outliers')

    columns_to_check = ['F(Hz)', 'Ua(V)', 'Ub(V)', 'Uc(V)']
    labels_data = set_labels_data(path, columns_to_check)

    Standardized_data = NormalizationData(data_not_outliers)
    data_out = Standardized_data.minmaxsaler_data()

    data_out['labels'] = labels_data.iloc[:, -1]
    draw_data_normalization(data_out)

    show(data_out)


