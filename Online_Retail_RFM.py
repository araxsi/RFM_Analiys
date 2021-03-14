# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:40:44 2021

@author: ISKENDER.YILMAZ
"""
"""
https://www.kaggle.com/mahmoudelfahl/cohort-analysis-customer-segmentation-with-rfm/data
E-ticaret web sitesinde daha çok kullandığımız Kohort Analizi 
(Kullanıcı ve Ürün Ömrü Üzerinden Tutma) ve Mobil Uygulama
"""

# import library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt

#For Data  Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#For Machine Learning Algorithm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


df = pd.read_excel(r'C:\Users\ISKENDER.YILMAZ\Desktop\Online Retail.xlsx')

#Fonksiyonlar klasörüne taşınacak.
def data_understanding(dataframe):
    print(dataframe.head(2),"\n\n")
    print(round(dataframe.describe().T),"\n\n")
    print(dataframe.info(),"\n\n")
    print(df.isnull().sum(),"\n\n")
    # plt.figure(figsize=(12, 10))
    # cor = dataframe.corr()
    # sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    # plt.show()
    # sns.pairplot(dataframe.drop(["CustomerID"], axis=1))
    # plt.show()

data_understanding(df)


#CustomerID bazında boş olan kayıtların silinmesini sağlıyoruz.
df= df.dropna(subset=['CustomerID'])
#Çift kayırların temizlenmesi için inceleme yapıyoruz.Daha sonrasında çift kayıtları siliyoruz
df.duplicated().sum()
df = df.drop_duplicates()

#Veri özetini görmek için aşağıdaki gibi bir işlem yapıyoruz.
df.describe()

#Data özetinde bakıldığında Quantity bazında negatif değerler olduğunu görebiliyoruz. 
#bu sebeble aşağıdaki şekilde negatif değerlerden kurtuluyoruz.
df=df[(df['Quantity']>0) & (df['UnitPrice']>0)]
df.describe() 
df.shape

#Tarihin düzenlenmesi konusunda aşağıdaki gibi bir düzenleme yapar COHORT analizi konusunda 
#kullanacağımız yapıyı oluşturuyoruz.
def get_month(x) : return dt.datetime(x.year,x.month,1)
df['InvoiceMonth'] = df['InvoiceDate'].apply(get_month)
grouping = df.groupby('CustomerID')['InvoiceMonth']
df['CohortMonth'] = grouping.transform('min')
df.tail()
#İlgili fonksiyon ile tarihleri parçalayıp geri dönderiyoruz. Yeni alanların üretilmesini sağlıyoruz.
def get_month_int (dframe,column):
    year = dframe[column].dt.year
    month = dframe[column].dt.month
    day = dframe[column].dt.day
    return year, month , day
invoice_year,invoice_month,_ = get_month_int(df,'InvoiceMonth')
cohort_year,cohort_month,_ = get_month_int(df,'CohortMonth')
year_diff = invoice_year - cohort_year 
month_diff = invoice_month - cohort_month 
df['CohortIndex'] = year_diff * 12 + month_diff + 1 


#Aylık Aktif müşterileri bir gruplama yöntemini kullanıyoruz.
grouping = df.groupby(['CohortMonth', 'CohortIndex'])
cohort_data = grouping['CustomerID'].apply(pd.Series.nunique)
# Tekil olarak gruplardaki sayıyı geri alıyoruz.
cohort_data = cohort_data.reset_index()
cohort_counts = cohort_data.pivot(index='CohortMonth',columns='CohortIndex',values='CustomerID')
cohort_counts



# Ayrılma (Terk etme) Tablosu 
cohort_size = cohort_counts.iloc[:,0]
retention = cohort_counts.divide(cohort_size,axis=0) #
retention.round(3) * 100 #Yüzdesel bazda gösterimi konusunda çalışmamızı sağlıyoruz. 

#Headmap tablosu ile görselleştirmesi ile aşağıdaki şekilde yapılmaktadır.
plt.figure(figsize=(15, 8))
plt.title('Ayrılma oranları')
sns.heatmap(data=retention,annot = True,fmt = '.0%',vmin = 0.0,vmax = 0.5,cmap="BuPu_r")
plt.show()

#Her grup için ortalama miktar
grouping = df.groupby(['CohortMonth', 'CohortIndex'])
cohort_data = grouping['Quantity'].mean()
cohort_data = cohort_data.reset_index()
average_quantity = cohort_data.pivot(index='CohortMonth',columns='CohortIndex',values='Quantity')
average_quantity.round(1)
average_quantity.index = average_quantity.index.date

#Heatmap
plt.figure(figsize=(15, 8))
plt.title('Her grup için ortalama miktar')
sns.heatmap(data=average_quantity,annot = True,vmin = 0.0,vmax =20,cmap="BuGn_r")
plt.show()



#Yeni Toplam Kolonu  
df['TotalSum'] = df['UnitPrice']* df['Quantity']

#Veri hazırlama adımları
print('Min Invoice Date:',df.InvoiceDate.dt.date.min(),'max Invoice Date:',
       df.InvoiceDate.dt.date.max())

df.head(3)


#Gerçek dünyada, bugünün veya dünün verilerinin en son anlık görüntüsü ile çalışıyor olurduk.

snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
snapshot_date



# RFM ölçümlerini hesaplıyoruz.
rfm = df.groupby(['CustomerID']).agg({'InvoiceDate': lambda x : (snapshot_date - x.max()).days,
                                      'InvoiceNo':'count','TotalSum': 'sum'})


#Colon isimlerini değiştiriyoruz.
rfm.rename(columns={'InvoiceDate':'Recency','InvoiceNo':'Frequency','TotalSum':'MonetaryValue'}
           ,inplace= True)

#Final RFM values
rfm.head()

"""# Son zamanlarda daha aktif olan "Yenilik" müşterisini daha az yeni olan müşteriden daha iyi değerlendireceğiz, çünkü her şirket müşterilerinin yeni olmasını istiyor

# Müşterinin daha fazla para harcamasını ve daha sık ziyaret etmesini istediğimiz için (bu, yenilikten farklı bir sıradır) "Sıklık" ve "Parasal Değer" i daha yüksek etiketle derecelendireceğiz."""



#Segmentleri oluşturuyoruz.
r_labels =range(4,0,-1)
f_labels=range(1,5)
m_labels=range(1,5)
r_quartiles = pd.qcut(rfm['Recency'], q=4, labels = r_labels)
f_quartiles = pd.qcut(rfm['Frequency'],q=4, labels = f_labels)
m_quartiles = pd.qcut(rfm['MonetaryValue'],q=4,labels = m_labels)
rfm = rfm.assign(R=r_quartiles,F=f_quartiles,M=m_quartiles)

# Segment ve skorlarını oluşturuyoruz
def add_rfm(x) : return str(x['R']) + str(x['F']) + str(x['M'])
rfm['RFM_Segment'] = rfm.apply(add_rfm,axis=1 )
rfm['RFM_Score'] = rfm[['R','F','M']].sum(axis=1)

rfm.head()


#RFM Analizi yapılırken en yüksek RFM gruplaması fazla olandan başlanırö

rfm.groupby(['RFM_Segment']).size().sort_values(ascending=False)[:5]
#RFM Segment sayısı "111" olan grubu analiz etmeye başlıyoruz.
rfm[rfm['RFM_Segment']=='1.01.01.0'].head()


rfm.groupby('RFM_Score').agg({'Recency': 'mean','Frequency': 'mean',
                             'MonetaryValue': ['mean', 'count'] }).round(1)

"""Müşterileri gruplarındaki değerlendirmelerine göre ayırmaç için 
segmentlerine ayırmak için RFM puanını kullanın"""

def segments(df):
    if df['RFM_Score'] > 9 :
        return 'Gold'
    elif (df['RFM_Score'] > 5) and (df['RFM_Score'] <= 9 ):
        return 'Sliver'
    else:  
        return 'Bronze'

rfm['General_Segment'] = rfm.apply(segments,axis=1)

rfm.groupby('General_Segment').agg({'Recency':'mean','Frequency':'mean',
                                    'MonetaryValue':['mean','count']}).round(1)

"""Kmeans Kümeleme için Veri Ön İşleme¶

Kmeans Kümeleme Modumuzu uygulamadan önce bu Anahtar k-anlamı varsayımlarını kontrol etmeliyiz

    Değişkenlerin simetrik dağılımı (çarpık değil)
    Aynı ortalama değerlere sahip değişkenler
    Aynı varyansa sahip değişkenler

"""

rfm_rfm = rfm[['Recency','Frequency','MonetaryValue']]
print(rfm_rfm.describe())

"""Bu tablodan şu Problemi buluyoruz: Ortalama ve Varyans Eşit Değil

Soluation: Scikit-learn kütüphanesinden bir ölçekleyici kullanarak değişkenleri ölçekleme"""



# RFM dağılımını Plot Alanında görülmesi 
f,ax = plt.subplots(figsize=(10, 12))
plt.subplot(3, 1, 1); sns.distplot(rfm.Recency, label = 'Recency')
plt.subplot(3, 1, 2); sns.distplot(rfm.Frequency, label = 'Frequency')
plt.subplot(3, 1, 3); sns.distplot(rfm.MonetaryValue, label = 'Monetary Value')
plt.style.use('fivethirtyeight')
plt.tight_layout()
plt.show()

"""Ayrıca, başka bir Sorun daha var: Değişkenlerin simetrik olmayan dağılımı (çarpık veriler)

Çözüm: Logaritmik dönüşüm (yalnızca pozitif değerler) çarpıklığı yönetecek

Ön işleme adımlarını yapılandırmanın Sırasını kullanıyoruz: 1. Veriyi eğriltme - günlük dönüşümü

2. Aynı ortalama değerlere standardize edin

3. Aynı standart sapmaya göre ölçeklendirin

4. Kümeleme için kullanılacak ayrı bir dizi olarak saklayın

Sıra neden önemlidir?

    Günlük dönüşümü yalnızca pozitif verilerle çalışır
    Normalleştirme, verileri negatif değerlere sahip olmaya zorlar ve günlük çalışmaz"""
    
    
    
#Günlük dönüşümü ile verileri eğriltme
rfm_log = rfm[['Recency', 'Frequency', 'MonetaryValue']].apply(np.log, axis = 1).round(3)
#or rfm_log = np.log(rfm_rfm)


# RFM değerlerinin dağılımını çizilmesi
f,ax = plt.subplots(figsize=(10, 12))
plt.subplot(3, 1, 1); sns.distplot(rfm_log.Recency, label = 'Recency')
plt.subplot(3, 1, 2); sns.distplot(rfm_log.Frequency, label = 'Frequency')
plt.subplot(3, 1, 3); sns.distplot(rfm_log.MonetaryValue, label = 'Monetary Value')
plt.style.use('fivethirtyeight')
plt.tight_layout()
plt.show()

#*******************************K-Means Kümelemesinin Uygulanması************************************

"""
Anahtar adımlar

    1-Veri ön işleme
    2-Birkaç küme seçme
    3-K-anlamına gelir kümeleme önceden işlenmiş veriler üzerinde çalıştırma
    4-Her kümenin ortalama RFM değerlerini analiz etme

"""
#1 Veri Önişlemesi
#Değişkenleri StandardScaler ile normalleştirin
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(rfm_log)
#Store it separately for clustering
rfm_normalized= scaler.transform(rfm_log)

#2 Küme Seçme işlemi
"""
Küme sayısını tanımlama yöntemleri

    Görsel yöntemler - dirsek kriteri
    Matematiksel yöntemler - siluet katsayısı
    Deney ve yorumlama

Dirsek kriteri yöntemi

    Küme içi kareler toplamı hatalarına (SSE) göre küme sayısını çizin -
    her veri noktasından küme merkezine kadar olan mesafelerin karesi toplamı
    Grafikte bir "dirsek" tanımlayın
    Dirsek - "optimal" sayıda kümeyi temsil eden bir nokta
    
"""

from sklearn.cluster import KMeans

#First : En İyi KMean'leri Alın
ks = range(1,8)
inertias=[]
for k in ks :
    # Create a KMeans clusters
    kc = KMeans(n_clusters=k,random_state=1)
    kc.fit(rfm_normalized)
    inertias.append(kc.inertia_)

# Plot ks vs inertias
f, ax = plt.subplots(figsize=(15, 8))
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.style.use('ggplot')
plt.title('En iyi Kmeans Hangisidir ?')
plt.show()



# clustering
kc = KMeans(n_clusters= 3, random_state=1)
kc.fit(rfm_normalized)

#Create a cluster label column in the original DataFrame
cluster_labels = kc.labels_

#Calculate average RFM values and size for each cluster:
rfm_rfm_k3 = rfm_rfm.assign(K_Cluster = cluster_labels)

#Calculate average RFM values and sizes for each cluster:
rfm_rfm_k3.groupby('K_Cluster').agg({'Recency': 'mean','Frequency': 'mean',
                                         'MonetaryValue': ['mean', 'count'],}).round(0)

"""

Segmentleri anlamak ve karşılaştırmak için yılan grafikleri

    Farklı segmentleri karşılaştırmak için pazar araştırması tekniği
    Her segmentin özelliklerinin görsel temsili
    Önce verileri normalleştirmeniz gerekir (merkez ve ölçek)
    Her bir özelliğin her bir kümenin ortalama normalleştirilmiş değerlerinin grafiğini çizin
    
"""

rfm_normalized = pd.DataFrame(rfm_normalized,index=rfm_rfm.index,columns=rfm_rfm.columns)
rfm_normalized['K_Cluster'] = kc.labels_
rfm_normalized['General_Segment'] = rfm['General_Segment']
rfm_normalized.reset_index(inplace = True)

#Melt the data into a long format so RFM values and metric names are stored in 1 column each
rfm_melt = pd.melt(rfm_normalized,id_vars=['CustomerID','General_Segment','K_Cluster'],value_vars=['Recency', 'Frequency', 'MonetaryValue'],
var_name='Metric',value_name='Value')
rfm_melt.head()




f, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 8))
sns.lineplot(x = 'Metric', y = 'Value', hue = 'General_Segment', data = rfm_melt,ax=ax1)

# a snake plot with K-Means
sns.lineplot(x = 'Metric', y = 'Value', hue = 'K_Cluster', data = rfm_melt,ax=ax2)

plt.suptitle("Snake Plot of RFM",fontsize=24) #make title fontsize subtitle 
plt.show()


"""
Segment özelliklerinin göreceli önemi

    Her segmentin özelliğinin göreceli önemini belirlemek için faydalı teknik
    Her kümenin ortalama değerlerini hesaplayın
    Nüfusun ortalama değerlerini hesaplayın
    Önem puanını bölerek ve 1 çıkararak hesaplayın (küme ortalaması, nüfus ortalamasına eşit olduğunda 0'ın döndürülmesini sağlar)

Bir ısı haritasıyla tekrar deneyelim.
 Isı haritaları, daha büyük değerlerin daha koyu ölçeklerde ve daha küçük değerlerin daha açık 
 ölçeklerde renklendirildiği verilerin grafik gösterimidir. Gruplar arasındaki varyansı renklere 
 göre oldukça sezgisel olarak karşılaştırabiliriz.
 
 """
 
# Oran 0'dan ne kadar uzaksa, özellik toplam popülasyona göre bir segment için o kadar önemlidir
cluster_avg = rfm_rfm_k3.groupby(['K_Cluster']).mean()
population_avg = rfm_rfm.mean()
relative_imp = cluster_avg / population_avg - 1
relative_imp.round(2)


# toplamdaki ortalama değer
total_avg = rfm.iloc[:, 0:3].mean()
# orantılı boşluğu toplam ortalama ile hesaplayalım
cluster_avg = rfm.groupby('General_Segment').mean().iloc[:, 0:3]
prop_rfm = cluster_avg/total_avg - 1
prop_rfm.round(2)


# RFM Isı Haritasına uygulandığında
f, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 5))
sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='Blues',ax=ax1)
ax1.set(title = "Heatmap of K-Means")

# K-Means ile bir yılan komplosu
sns.heatmap(prop_rfm, cmap= 'Oranges', fmt= '.2f', annot = True,ax=ax2)
ax2.set(title = "Heatmap of RFM quantile")

plt.suptitle("Heat Map of RFM",fontsize=20) #make title fontsize subtitle 

plt.show()


"""

Tenure değişkenini ekleyerek RFM verilerini güncelleyebilirsiniz:
    -Tenure: ilk işlemden itibaren geçen süre ،
    Müşterinin şirkette ne kadar süredir çalıştığını tanımlar

Sonuç: Müşteri satın alma verilerinden RFM değerlerinin nasıl alınacağından bahsettik ve 
RFM miktarları ve K-Means kümeleme yöntemleriyle iki tür segmentasyon yaptık.

"""







    



