print("HGELLOO")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
# Lue kaikki data Pandas dataframeksi
#    Sarakkeet datassa:
#    MEASUREMENT_INDEX               : kyseisen MEMS-kiekon mittauksen yksilöllinen tunniste, yksilöllinen jokaiselle kiekko/mittaus -parille
#    LOT                             : Anonymisoitu valmistuserän koodi. Jokaisessa valmistuserässä on 1-25 MEMS-kiekkoa, joiden numero erän sisällä on merkattu WAFER_ID:llä.
#    WAFER_ID                        : Yksittäisen kiekon numerojärjestys erän sisällä (eli WAFER_ID 5 ja LOT_7 = Viides kiekko erästä LOT_7).
#    WAFER_X_COORDINATE              : Anturin X-koordinaatti MEMS-kiekolla. Origo on MEMS-kiekon keskikohta.
#    WAFER_Y_COORDINATE              : Anturin Y-koordinaatti MEMS-kiekolla. Origo on MEMS-kiekon keskikohta.
#    Count(defect_class_nn)          : Montako kpl defektityyppiä NN kyseisen MEMS-kiekon kyseisellä anturilla on.
#    Sum(DEFECT_AREAdefect_class_nn) : Mikä on defektityypin NN yhteenlaskettu pinta-ala kyseisen MEMS-kiekon kyseisellä anturilla.
#   
#    Muuta:
#    - Yksi rivi = yksi anturi. Yhdellä MEMS-kiekolla on 3300 anturia.
#    - Defektiluokkia (defect_class_nn) on 33 kpl (20 - 32 sekä "no_defect")
#    - Sum(DEFECT_AREA,defect_class_no_defect) on *aina* 0 eli turha sarake. Koska ei-defektiä -luokalle ei voi laskea defektin pinta-alaa.
data_pandas = pd.read_csv("./RJ structure wfr defect data - one sensor per row.tsv", sep = "\t")

# Negatiiviset koordinaatit rikkovat Pythonin indeksoinnin --> shiftataan positiiviselle alueelle
data_pandas.WAFER_X_COORDINATE += abs( data_pandas.WAFER_X_COORDINATE.min() )
data_pandas.WAFER_Y_COORDINATE += abs( data_pandas.WAFER_Y_COORDINATE.min() )



# Määritellään mitkä sarakkeet datasta toimivat kuvan "värikanavina"
# Värikanava = yksittäinen Count(defect_class_nn) tai Sum(defect_class_nn) -sarake
sarakkeet         = data_pandas.columns
kanava_sarakkeet  = []
kanava_sarakkeet += list( sarakkeet[ sarakkeet.str.startswith("Count") ] )
#kanava_sarakkeet += list( sarakkeet[ sarakkeet.str.startswith("Sum")   ] ) #EI OTETA SUMMIA MUKAAN DATAAN


# Initialisoidaan (N mittausta/kiekkoa, leveys_y, leveys_x, "värikanavat") kokoinen matriisi "images".
# Lopputuloksena pitäisi olla (507, 47, 47, 28) -kokoinen matriisi (tai 4. asteen tensori, miten sitä haluaakaan kutsua),
# Jokainen matriisin alkio initialisoidaan arvolle -1 (arvo, jota ei löydy datasta)
n_measurements = len(data_pandas.MEASUREMENT_INDEX.unique())
n_channels     = len(kanava_sarakkeet)
max_x          = int(data_pandas.WAFER_X_COORDINATE.max())
max_y          = int(data_pandas.WAFER_Y_COORDINATE.max())
images = -1 * np.ones( (n_measurements, max_y, max_x, n_channels) )

# Populoidaan matriisi
# 1.   MEASUREMENT_INDEX on pitkä numerosarja, lisätään dataan yksinkertaisempi mittauksen järjestysnumero.
#      Tämän avulla voidaan hakea images-matriisista esim. kuudennen kiekon mittaus. Tämä helpottaa datan populointia.
#      https://saturncloud.io/blog/how-to-convert-categorical-data-to-numerical-data-with-pandas/
# 1.5. Kuvan järjestysnumeron voi jälkikäteen mäpätä MEASUREMENT_INDEX:iin esim. tällä komennolla:
#      data_pandas.MEASUREMENT_INDEX[ data_pandas.MEASUREMENT == kuvan_järjestysnumero ].unique()
#
# 2. Populointi tapahtuu subsettaamalla images-matriisista kaikki mittaus-x-y-kanava -yhdistelmät ja asettamalla niiden arvoksi halutun sarakkeen datat.
#    Menee kategoriaan "kind of a hack" mutten keksinyt muutakaan (poislukien neljä sisäkkäistä for-looppia jotka täyttävät aina yhden alkion matriisista kerrallaan)
data_pandas["MEASUREMENT"] = data_pandas.MEASUREMENT_INDEX.astype("category").cat.codes

data_numpy = data_pandas[ ["MEASUREMENT", "WAFER_Y_COORDINATE", "WAFER_X_COORDINATE"] ].to_numpy()
for idx in range( n_channels ):
    images[ 
        data_numpy[:, 0],
        data_numpy[:, 1] - 1,
        data_numpy[:, 2] - 1,
        idx

    ] = data_pandas[kanava_sarakkeet[idx]].to_numpy()

# Sanity check:
print(images.shape)
#   np.savetxt("test.tsv", images[0, :, :, 0], delimiter = "\t")

for value in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    train_images, test_images = train_test_split(images, test_size=value, random_state=42, shuffle = True)
    #Number of wafers and channels OF TRAIN DATA
    n_wafers = train_images.shape[0]
    _y = train_images.shape[1]
    n_x = train_images.shape[2]
    n_channels = train_images.shape[3]
    #flatten TRAIN data
    flattened_data = train_images.reshape(n_wafers, -1)  # Shape: (n_wafers, 47*47*n_channels)
    #Number of wafers and channels OF ALL DATA
    n_wafers2 = images.shape[0]
    n_y2 = images.shape[1]
    n_x2 = images.shape[2]
    n_channels2 = images.shape[3]
    #flatten ALL data
    flattened_data_all = images.reshape(n_wafers2, -1)  # Shape: (n_wafers, 47*47*n_channels)

    # Apply PCA
    N=2
    pca = PCA(n_components=N)  # Choose N principal components for reduction
    pca.fit_transform(flattened_data) #train PCA
    pca_result = pca.transform(flattened_data_all) #fit all the data to PCA

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()
    for i,  k in enumerate(np.arange(2,6,1)):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pca_result)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        axs[i].scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='rainbow')
        axs[i].scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', s=100)
        axs[i].set_title(f'K-Means k={k}')
        axs[i].set_xlabel('PCA Feature 1')
        axs[i].set_ylabel('PCA Feature 2')
    fig.suptitle(f'train_size = {1 - value}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the main title
    plt.show()