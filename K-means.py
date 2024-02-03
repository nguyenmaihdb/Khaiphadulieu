import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import silhouette_score


data = pd.read_csv('data.csv', index_col = None)
#data.head()

#Thống kê mô tả các trường của bộ dữ liệu
data.describe()

#Chọn các thuộc tính là dữ liệu số và lưu vào datafram
numeric_columns = data.select_dtypes(include=[np.number])
#Lấy tập dữ liệu X từ dữ liệu để đưa vào dự đoán nhưng bỏ cột target vì đây là cột nhãn
X = numeric_columns.drop('class',axis=1).copy()
#X = X.drop(['class', ''], axis=1)
columns = X.columns
data_value = X.values

print('data:\n',X)
print('mean:\n',X.mean())

#Tìm các giá trị bị thiếu trong các cột thuộc tính
##for col in columns:
####    print(col)
##    missing_data = X[col].isna().sum()
##    missing_percent = missing_data/len(X)*100
##    print(f"Cột {col} có {missing_percent}% missing data")

###Thay những giá trị trống, thiếu = mean của cột
##for i in range(X.shape[1]):
##    X[columns[i]].fillna(X.mean().iloc[i], inplace=True)

#Khởi tạo k tâm cụm ngẫu nhiên
k_min = 1
k_max=100

#Khởi tạo ngẫu nhiên k tâm cụm bằng cách lấy mẫu ngẫu nhiên từ DataFrame X.
Centroids = (X.sample(n=k))
print("- Tâm cụm khởi tạo: \n",Centroids)
k = random.randint(min_k, max_k)
Zzzzz
# Print the chosen value for k
print(f"Chosen value for k: {k}")

#Tính khoảng cách giữa 2 điểm
def distance(row_c, row_x):
    d = sqrt((row_c["Glucose"]-row_x["Glucose"])**2 + (row_c["BloodPressure"]-row_x["BloodPressure"])**2
           + (row_c["SkinThickness"]-row_x["SkinThickness"])**2 + (row_c["Insulin"]-row_x["Insulin"])**2
           + (row_c["BMI"]-row_x["BMI"])**2 + (row_c["DiabetesPedigreeFunction"]-row_x["DiabetesPedigreeFunction"])**2
           + (row_c["Age"]-row_x["Age"])**2)
    return d

#K-means
diff=1  #số lần thay đổi tâm cụm
j=0
loop = 1
while(diff!=0): #nếu số lần thay đổi khác 0 thì sẽ thực hiện lặp để tìm ra tâm cụm mới
    print("\n\n~~~~~~~~~~~~Lần lặp: ",loop)
    i=1
    for index1, row_c in Centroids.iterrows(): #1 row_c là 1 series: chứa thông tin của 1 dòng dữ liệu
        ED=[]
        for index2, row_x in X.iterrows():
            d=distance(row_c,row_x)
            ED.append(d)
        X["d(C"+str(i)+")"]=ED
        i=i+1

    C=[]
    for index, row in X.iterrows():
        min_dist=row["d(C1)"]
        pos=1
        for i in range(k):
            if row["d(C"+str(i+1)+")"]<min_dist:
                min_dist = row["d(C"+str(i+1)+")"]
                pos=i+1
        C.append(str(pos))

    X["Cum"]=C
    print("\n+ Dữ liệu X: \n",X)

    #Nhóm các tâm cụm có chỉ số giống nhau để tính trung bình cộng
    Centroids_new = X.groupby(["Cum"]).mean()[["Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]]

    print("\n+ Tâm cụm mới: \n",Centroids_new)
    print("\n+ Tâm cụm cũ: \n",Centroids)

    #Lấy tâm cụm mới - tâm cụm cũ rồi tính tổng, nếu tổng =0 thì không có sự khác nhau => dừng thuật toán và ngược lại
    diff = (Centroids_new["Glucose"]-Centroids["Glucose"]).sum() + (Centroids_new["BloodPressure"]-Centroids["BloodPressure"]).sum()
    + (Centroids_new["SkinThickness"]-Centroids["SkinThickness"]).sum() + (Centroids_new["Insulin"]-Centroids["Insulin"]).sum()
    + (Centroids_new["BMI"]-Centroids["BMI"]).sum() + (Centroids_new["DiabetesPedigreeFunction"]-Centroids["DiabetesPedigreeFunction"]).sum()
    + (Centroids_new["Age"]-Centroids["Age"]).sum()
    print("\n+ So sánh sự khác nhau: diff =",diff)
    if j==0:
        diff=1
        j=j+1

    loop+=1
    Centroids = Centroids_new
print("\n- Tâm cụm cuối cùng:\n",Centroids)

#Thống kê tổng số mẫu trong mỗi cụm qua nhãn
x, y = np.unique(data['class'], return_counts = True)
print('\n- Tổng số mẫu trong mỗi cụm:\n',x,"\n",y)
# Thống kê tổng số mẫu trong mỗi cụm qua nhãn K-means
#cluster_counts = X["Cum"].value_counts()
#print('\n- Tổng số mẫu trong mỗi cụm K-means:\n', cluster_counts)

#Độ phù hợp
print("\n- Mức độ phù hợp silhouette_score = ", silhouette_score(X, X["Cum"]))

#Scatter
#Lấy x, y
x = data['Glucose'].values
y = data['BloodPressure'].values
colors = data['class'].values

# Lấy tâm cụm
centroids1 = Centroids.iloc[0][['Glucose','BloodPressure']]
centroids2 = Centroids.iloc[1][['Glucose','BloodPressure']]

#Vẽ biểu đồ
plt.scatter(x, y, c=colors, marker='*')
plt.scatter(centroids1['Glucose'], centroids1['BloodPressure'], c='red', marker='o', s=100)
plt.scatter(centroids2['Glucose'], centroids2['BloodPressure'], c='blue', marker='o', s=100)

plt.xlabel('Glucose', fontsize=16)
plt.ylabel('BloodPressure', fontsize=16)
plt.title("Diabetes clustering chart", fontsize=18)

# Add a legend to distinguish the centroids
#plt.legend()

plt.show()









#plt.scatter(X['Glucose'], X['BloodPressure'], c=X['Cum'].astype('category').cat.codes, cmap='viridis', s=50, alpha=0.5)
#plt.scatter(Centroids['Glucose'], Centroids['BloodPressure'], c='red', marker='X', s=200, label='Centroids')
#colors = data['Cum'].astype('category').cat.codes
#colors  = np.where(X['Cum'].astype('category').cat.codes == 1, 'green', 'yellow')
#colors = np.where(data['class'].values == 0, 'green', 'yellow')

# Lấy tâm cụm
#centroids1 = Centroids.iloc[0][['Glucose', 'BloodPressure']]
###centroids2 = Centroids.iloc[1][['Glucose', 'BloodPressure']]

# Vẽ biểu đồ
#plt.scatter(x, y, c=colors_data[X["Cum"].astype('category').cat.codes],marker='*', cmap='rainbow', s=50, alpha=0.5)
#plt.scatter(x, y, c=colors,marker='*', s=50, alpha=0.5)
#plt.scatter(centroids1['Glucose'], centroids1['BloodPressure'], c=colors_centroids[0], marker='o', s=100, label='Centroid 1 (Glucose)')
#plt.scatter(centroids2['Glucose'], centroids2['BloodPressure'], c=colors_centroids[1], marker='o', s=100, label='Centroid 2 (BloodPressure)')


