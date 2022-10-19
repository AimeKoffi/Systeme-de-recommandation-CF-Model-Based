import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

years = []

# read CSVs
movies = pd.read_csv('datas/movies.csv')
ratings = pd.read_csv('datas/ratings.csv')

# merge on movieId column
data = pd.merge(left=movies, right=ratings, on='movieId')
for title in data['title']:
    year_subset = title[-5:-1]
    try: years.append(int(year_subset))
    except: years.append(9999)
        
data['moviePubYear'] = years
print(len(data[data['moviePubYear'] == 9999]))

genre_df = pd.DataFrame(data['genres'].str.split('|').tolist(), index=data['movieId']).stack()
genre_df = genre_df.reset_index([0, 'movieId'])
genre_df.columns = ['movieId', 'Genre']
tab = data

useri,frequsers=np.unique(tab.userId,return_counts=True)#useri les id des users, frequsers les freq de chaque user
itemi,freqitems=np.unique(tab.movieId,return_counts=True)#itemi les id des item, freqitem les freq de chaque item
n_users=len(useri)
n_items=len(itemi)
print("le nombre des utilisateurs est :"+ str(n_users) + " Et le nombre des items est: "+ str(n_items))

"""
Un des problèmes que j'ai rencontré était le fait que les ids des users et des items n'était pas ordonnée.
C'est à dire on peut trouver l'utilisateur 1,2,3,5 et 8 sans trouver les utilisateurs 4 ,6 et 7. 
Ceci a posé un problème dans la création de la matrice user-item parce que on risque d'avoir plusieurs lignes et colonnes vides.
Pour ça, j'ai crée un tableau indice_user et un tableau indice_item qui contiennent les anciens id et les nouvelles id 
par expl (1,2,5,6)=>(1,2,3,4) puis j'ai ajouté deux colonnes sur le tableau principale qui contient ces nouveaux IDs.
(ps) ce traitement est très couteux!
"""

indice_user = pd.DataFrame()
indice_user["indice"]=range(1,len(useri)+1)
indice_user["useri"]=useri

indice_item = pd.DataFrame()
indice_item["indice"]=range(1,len(itemi)+1)
indice_item["itemi"]=itemi

#create user_ID_new and Item_ID_new
x=[]
y=[]
for i in range(0,len(tab)):
    x.append((indice_user.indice[indice_user.useri==tab.userId[i]].axes[0]+1)[0])
    y.append((indice_item.indice[indice_item.itemi==tab.movieId[i]].axes[0]+1)[0])

tab["User_ID_new"]=x
tab["Item_ID_new"]=y

#create a train_data and test_data
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(tab[["User_ID_new","Item_ID_new","rating"]], test_size=0.25,random_state=123)


"""
Dans cette partie du projet, j'applique le deuxième sous-type du fitrage collaboratif : "Model-based". 
Il consiste à appliquer la matrice de factorisation (MF) : c'est une méthode d'apprentissage non supervisé de décomposition
et de réduction de dimensionnalité pour les variables cachées. 

Le but de la matrice de factorisation est d'apprendre les préférences cachées des utilisateurs et les attributs cachés des items
depuis les ratings connus dans notre jeu de données, pour enfin prédire les ratings inconnus en multipliant les matrices de varibales 
cachées des utilisateurs et des items. 

Pour le projet, j'ai utilisé l'algorithme:
- ALS : Alternating Least Squares

"""

""" 
Pour le faire, j'ai commencé par créer les matrice user-item train et test. Ce sont les deux matrices qui vont croisé les notes de utilsiateurs et des items.
Puis, j'ai créé une fonction pour faire les prédictions

"""
train_data_matrix = np.zeros((n_users, n_items))#matrice nulle de longuer tous les users et tous les items
for line in train_data.itertuples():#parcourire la ligne col par col
    train_data_matrix[line[1]-1, line[2]-1] = line[3] 

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]

# La fonction prediction permet de prédire les ratings inconnus en multipliant les matrices P et la transposée de Q
def prediction(P,Q):
    return np.dot(P.T,Q)

""" Il existe plusieurs métriques d'évaluation, mais la plus populaire des métriques utilisée pour évaluer l'exactitude des ratings prédits
est l'erreur quadratique moyenne (RMSE) que j'ai utilisé dans le projet :
RMSE = RacineCarrée{(1/N) * sum (r_i -estimé{r_i})^2}
"""

def rmse(I,R,Q,P):
    return np.sqrt(np.sum((I * (R - prediction(P,Q)))**2)/len(R[R > 0]))    


# Script for training model with Alternating Least Squares algorithm
train_errors = []
test_errors = []

# Index matrix for training data
I = train_data_matrix.copy()
I[I > 0] = 1
I[I == 0] = 0

# Index matrix for test data
I2 = test_data_matrix.copy()
I2[I2 > 0] = 1
I2[I2 == 0] = 0

lmbda = 0.1 
k = 20 
n_epochs = 2 # number of epochs
m, n = train_data_matrix.shape # Number of users and items
P = 3 * np.random.rand(k,m) # Latent user feature matrix
Q = 3 * np.random.rand(k,n) # Latent item feature matrix
Q[0,:] = train_data_matrix[train_data_matrix != 0].mean(axis=0) # Avg. rating for each movie
E = np.eye(k) # (k x k)-dimensional idendity matrix

# Repeat until convergence
for epoch in range(n_epochs):
    # Fix Q and estimate P
    for i, Ii in enumerate(I):
        nui = np.count_nonzero(Ii) # Number of items user i has rated
        if (nui == 0): nui = 1 # Be aware of zero counts!
    
        # Least squares solution
        Ai = np.dot(Q, np.dot(np.diag(Ii), Q.T)) + lmbda * nui * E
        Vi = np.dot(Q, np.dot(np.diag(Ii), train_data_matrix[i].T))
        P[:,i] = np.linalg.solve(Ai,Vi)
        
    # Fix P and estimate Q
    for j, Ij in enumerate(I.T):
        nmj = np.count_nonzero(Ij) # Number of users that rated item j
        if (nmj == 0): nmj = 1 # Be aware of zero counts!
        
        # Least squares solution
        Aj = np.dot(P, np.dot(np.diag(Ij), P.T)) + lmbda * nmj * E
        Vj = np.dot(P, np.dot(np.diag(Ij), train_data_matrix[:,j]))
        Q[:,j] = np.linalg.solve(Aj,Vj)
    
    train_rmse = rmse(I,train_data_matrix,Q,P)
    test_rmse = rmse(I2,test_data_matrix,Q,P)
    train_errors.append(train_rmse)
    test_errors.append(test_rmse)
    
    print("[Epoch %d/%d] train error: %f, test error: %f"
             % (epoch+1, n_epochs, train_rmse, test_rmse))

print("Algorithm converged")



# Maitenant, après avoir obtenus toutes les valeurs de l'erreur à chaque étape,on peut tracer la courbe d'apprentissage.
# ==> On Vérifie la performance en traçant les erreurs du train et du test
plt.plot(range(n_epochs), train_errors, marker='o', label='Training Data')
plt.plot(range(n_epochs), test_errors, marker='v', label='Test Data')
plt.title('Courbe d apprentissage SGD')
plt.xlabel('Nombre d etapes')
plt.ylabel('RMSE')
plt.legend()
plt.grid()
plt.show()