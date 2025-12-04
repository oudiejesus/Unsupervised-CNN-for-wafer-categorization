import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
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
#data_pandas = pd.read_csv("./RJ structure wfr defect data - one sensor per row.tsv", sep = "\t")
data_pandas = pd.read_csv("./defect data.tsv", sep="\t")

# Negatiiviset koordinaatit rikkovat Pythonin indeksoinnin --> shiftataan positiiviselle alueelle
data_pandas.WAFER_X_COORDINATE += abs( data_pandas.WAFER_X_COORDINATE.min() )
data_pandas.WAFER_Y_COORDINATE += abs( data_pandas.WAFER_Y_COORDINATE.min() )



# Määritellään mitkä sarakkeet datasta toimivat kuvan "värikanavina"
# Värikanava = yksittäinen Count(defect_class_nn) tai Sum(defect_class_nn) -sarake
sarakkeet         = data_pandas.columns
kanava_sarakkeet  = []
kanava_sarakkeet += list( sarakkeet[ sarakkeet.str.startswith("Count") ] )
kanava_sarakkeet += list( sarakkeet[ sarakkeet.str.startswith("Sum")   ] ) #SUMMAT


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

print(images.shape)
#permute to [i , channeltype , channelnro , x , y]
'''

#LOPULLINEN MALLI
images = np.load('data_all.npy') 
#data_all.npy on numpy-array, joka sisältää 3949 kiekon datasetin, joka on käsitelty yllä olevan koodin avulla ja sitten tallennettu np.save komennolla. 
#data on permutoitu ennen tallennusta ja tallennettu muodossa [datapiste, kanava (1-28) , x , y]

print(images.shape)




#muodostetaan datasetti 3dCNN:lle. [datapiste, kanava (1-14) , vikatyyppi (0 = määrä, 1 = pinta-ala) ,  x , y ]
images = images.reshape(3949,14,2,109,74)
indices = np.arange(len(images))  # Create an array of original indices

#images_torch = torch.tensor(images).float()

#split
train_images, test_images, train_indices, test_indices = train_test_split(
    images, indices, test_size=0.2, random_state=42, shuffle=True
) #huom shuffle on päällä. random state 42 tuottaa aina saman jaon
images_torch = torch.tensor(train_images).float() #tällä cnn koulutettiin
test_torch = torch.tensor(test_images).float()


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=14, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(2,3,3), stride=2, padding=1), 
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(2,3,3), stride=2, padding=(0,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(256* 1* 28 * 19, 256),
            nn.ReLU(),
            
            nn.Linear(256,3) )
        self.decoder = nn.Sequential(
            
            nn.Linear(3,256),
            nn.ReLU(),
            
            nn.Linear(256, 256 *1 * 28 * 19),
            nn.Unflatten(1,(256,1,28,19)),   
            
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=(2,3,3), stride=2, padding=(0,1,1)),
            
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(2,3,3), stride=2, padding=1),
            
            nn.ReLU(),
            nn.Upsample((2,109,74), mode = "nearest"),
            
            nn.ReLU(),
            nn.ConvTranspose3d(in_channels=32, out_channels=14, kernel_size=3, stride=1, padding=1) )

    def forward(self, x):
        encoded = self.encoder(x)
        #print(encoded.shape)
        decoded = self.decoder(encoded)
        return decoded

 
#treenifunktio
'''
def train_autoencoder(model, dataloader, criterion, optimizer, num_epochs, device='cuda'):
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs in dataloader:
            # Move inputs to the correct device
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            #print(outputs.shape)
            #print(inputs.shape)
            loss = criterion(outputs, inputs)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


# train autoencoder-decoder
if __name__ == "__main__":
    # Hyperparameters
    num_epochs = 50
    batch_size = 10
    learning_rate = 0.00005
    dataloader = DataLoader(images_torch, batch_size=batch_size, shuffle=True)
    
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_autoencoder(model, dataloader, criterion, optimizer, num_epochs=num_epochs, device='cuda')
'''
'''
#tallennus
# Save only the encoder's state_dict
torch.save(model.state_dict(), 'cnn_standard_norm_herkku_state_dict.pth')
#print("cnn saved successfully!")
'''


#painojen lataus
model = Autoencoder()  # Replace `Autoencoder` with your autoencoder class
model.load_state_dict(torch.load('cnn_norm_herkku_state_dict.pth'))
#Move the model to GPU if necessary
model = model.to('cuda')
print("Autoencoder state_dict loaded successfully!")


#eristä encoder
encoder = model.encoder 


#opetusaineisto enkoodaus
batch_size = 64
dataset = torch.utils.data.TensorDataset(images_torch)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
encoded_data = []
with torch.no_grad():
    for batch in train_loader:
        images = batch[0].to('cuda').float()  # Move batch to GPU
        encoded_batch = encoder(images)      # Encode batch
        encoded_data.append(encoded_batch.cpu())  # Move to CPU and store
# Concatenate all batches into a single tensor
encoded_data = torch.cat(encoded_data, dim=0)
print("Encoded train data shape:", encoded_data.shape)



##TEST DATA ENCODE
dataset = torch.utils.data.TensorDataset(test_torch)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
encoded_test_data = []
with torch.no_grad():
    for batch in train_loader:
        images = batch[0].to('cuda').float()  # Move batch to GPU
        encoded_batch = encoder(images)      # Encode batch
        encoded_test_data.append(encoded_batch.cpu())  # Move to CPU and store
# Concatenate all batches into a single tensor
encoded_test_data = torch.cat(encoded_test_data, dim=0)
print("Encoded test data shape:", encoded_test_data.shape)



# Scatter plot
encoded_data = encoded_data.cpu().numpy() #opetusaineisto numpyksi
encoded_test_data = encoded_test_data.cpu().numpy()  # Test data
'''
#save
np.save('encoded_train_data.npy', encoded_data)
np.save('encoded_test_data.npy', encoded_test_data)
combined_data = np.concatenate((encoded_data, encoded_test_data), axis=0)
np.save('encoded_all_data.npy', combined_data)
'''

x = encoded_data[:, 0]
y = encoded_data[:, 1]
z = encoded_data[:, 2]
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# Scatter-plot 3D
ax.scatter(x, y, z, alpha=0.5, color = "blue",label = "Opetusaineisto")
ax.scatter(encoded_test_data[:,0], encoded_test_data[:,1], encoded_test_data[:,2], alpha = 0.5, color = "red", label = "Testiaineisto")
'''
# Annotate train data with original indices
for i, (x_, y_, z_) in enumerate(zip(x, y, z)):
    ax.text(x_, y_, z_ + 0.01, str(train_indices[i]), color='blue', fontsize=8)

# Annotate test data with original indices
for i, (x_, y_, z_) in enumerate(zip(encoded_test_data[:, 0], 
                                      encoded_test_data[:, 1], 
                                      encoded_test_data[:, 2])):
    ax.text(x_, y_, z_ + 0.01, str(test_indices[i]), color='red', fontsize=8)
'''

# Akselien nimet ja otsikko
ax.set_title('Enkoodattu numeroitu data 3 ulottuvuudessa')
ax.set_xlabel('Ulottuvuus 1')
ax.set_ylabel('Ulottuvuus 2')
ax.set_zlabel('Ulottuvuus 3')
ax.legend()
plt.show()