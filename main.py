# On veut détecter une région statistiquement peu probble
# On a des points 0 (bleu) et des points 1 (rouge)
# On classifie les points en des cases selon une certaine résolution
# Puis on met un flou gaussien
# Puis on fait une montée de gradient d'un rectangle
# Puis on enlève le flou gaussien
# Puis on mesure la proportion trouvée par le rectangle

import math
import random
import numpy as np
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=2)
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
import timeit
import scipy.stats as st
from scipy.special import comb
from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable


####################################################
####################################################

        # Définition de diverses fonctions

def factoriel(n): # n!
    return np.math.factorial(n)
def choose_n_k(n,k): # C(n,k) = n!/((n-k)!k!)
    return comb(N=n, k=k, exact=False)
def loi_binomiale(n,x,p): # P(X=x)
    return st.binom.pmf(k=x,n=n,p=p)
def mu_loi_binomiale(n,p): # espérance de la loi binomiale
    mu = n*p
    return mu
def var_loi_binomiale(n,p): # variance de la loi binomiale
    q = 1-p
    var = n*p*q
    return var
def sigma_loi_binomiale(n,p):
    return math.sqrt(var_loi_binomiale(n=n,p=p))
def cote_Z(x,mu,sigma):
    return (x-mu)/sigma
# Une fonction qui donne un array population [0,1,2,...,n] et l'autre les probabilités [P(X=0),P(X=1),...,P(X=n)] qu'il y ait exactement x bons pixels dans une image
def population_and_weights_binomial_law(n,p):
    population = list(range(n+1))
    weights = []
    for x in population:
        prob = loi_binomiale(n=n,x=x,p=p)
        weights.append(prob)
    return population,weights
def loi_normale_densite(x,mu,sigma):
    return exp(-(x-mu)**2 / (2*sigma**2)) / (sigma * math.sqrt(2*math.pi))
def loi_normale_prob_Z_moins_que_z(z): # P(Z<z)
    return ( 1 + math.erf(z/math.sqrt(2)) )/2
def loi_normale_prob_Z_entre_a_et_b(a,b): # P(a<Z<b)
    return loi_normale_prob_Z_moins_que_z(b) - loi_normale_prob_Z_moins_que_z(a)
def loi_normale_prob_Z_plus_que_z(z) : # P(Z>z)
    return 1-loi_normale_prob_Z_moins_que_z(z)
def cote_Z(x,mu,sigma):
    return (x-mu)/sigma
def loi_normale_prob_X_moins_que_x(x,mu,sigma):
    return loi_normale_prob_Z_moins_que_z(cote_Z(x,mu,sigma))
def loi_normale_prob_X_entre_a_et_b(a,b,mu,sigma):
    return loi_normale_prob_X_moins_que_x(b,mu,sigma) - loi_normale_prob_X_moins_que_x(a,mu,sigma)
def loi_normale_prob_X_plus_que_x(x,mu,sigma):
    return 1-loi_normale_prob_X_moins_que_x(x,mu,sigma)
def loi_de_Poisson(x,l): # pour x=0,1,2,3,...
    return st.poisson.pmf(k=x, mu=l)
    #return math.exp(-l)*math.pow(l,x)/factoriel(x)
def chi2(a):
    a = np.array(a) # taille (m,n)
    m,n = a.shape # m est la longueur de la variable X, n est la longueur de la variable Y
    somme = np.sum(a) # = sum(Y_sum)
    X_sum = np.sum(a,axis=0) # vecteur de longueur n
    Y_sum = np.sum(a,axis=1) # vecteur de longueur m
    X_sum = np.reshape(X_sum,newshape=(1,n)) # taille (1,n)
    Y_sum = np.reshape(Y_sum,newshape=(m,1)) # taille (m,1)
    a_theorique = X_sum*Y_sum/somme # taille (m,n), la table théorique
    nu = (m-1)*(n-1) # nombre de degrés de libertés
    chi2_somme = np.sum((a_theorique-a)**2/a_theorique)
    return chi2_somme, nu

####################################################
####################################################

        # Initialisation des données



resolutions = np.full(fill_value=1,shape=2,dtype=int) # pour l'instant la résolution est [1,1]
grid_f = np.zeros(shape=resolutions)


"""
Deux possibilités :

1. importer un fichier .csv ayant m lignes et n=2 colonnes
2. générer des données aléatoires de m points en dimension n=2 
"""




import_data_or_generate_it = 0
if import_data_or_generate_it==0:
    # On importe des données
    grid_f = pd.read_csv('mt_bruno_elevation.csv').values # source : https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv
    #grid_f = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv').values # source : https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv
    grid_f = grid_f - 100 # On shift la valeur de grid_f pour avoir des régions négatives, ce qui revient à changer une montagne en île
    resolutions = grid_f.shape
if import_data_or_generate_it==1:
    # On choisit le nombre de points
    n = 2    # n = nombre de dimensions continues
    m = 50000 # m = nombre de points
    # On choisit un écart-type pour les gaussiennes
    pixels_largeur_image = 25
    resolutions = np.full(fill_value=pixels_largeur_image,shape=n,dtype=int)
    # On se donne des données
    X = np.zeros((m,n)) # position des m points selon les n variables continues
    y = np.zeros(m,int) # valeurs binaires {0,1} des m points
    # On initialise les points avec des valeurs aléatoires (optionnel) :
    p=0.5 # probabilité qu'un point soit un 1
    q=1-p # probabilité qu'un point soit un 0
    # On se donne une distribution uniforme de points dans un carré [0,1] :
    X = np.random.random((m,n)) # taille (m,n) de nombres aléatoires en [0.0, 1.0[
    minimums_points = np.min(X,axis=0) # taille n, c'est le min dans chaque variables continues
    maximums_points = np.max(X,axis=0) # taille n, c'est le max dans chaque variables continues
    # On normalise les points
    X = (X-minimums_points)/(maximums_points-minimums_points) # On normalise les positions pour que le min et le max dans chaque dimensions soit 0 et 1, par une transformation linéaire
    minimums_points = np.min(X,axis=0) # taille n, c'est le min dans chaque variables continues (désormais des 0)
    maximums_points = np.max(X,axis=0) # taille n, c'est le max dans chaque variables continues (désormais des 1)
    x_min_points = minimums_points[0] # normalement ça vaut 0
    y_min_points = minimums_points[1] # normalement ça vaut 0
    x_max_points = maximums_points[0] # normalement ça vaut 1
    y_max_points = maximums_points[1] # normalement ça vaut 1
    # On catégorise les points dans des bins
    X_cat = np.zeros((m,n),int) # cat pour catégorie
    for j in range(n):
        min_j = minimums_points[j]
        max_j = maximums_points[j]
        res_j = resolutions[j]
        bin_j = np.linspace(start=min_j,stop=max_j,num=res_j+1)
        ind_j  = np.digitize(X[:,j],bin_j) # min <= x < max
        res_j = resolutions[j]
        ind_j[ind_j==res_j+1] = res_j # pour x==max on doit corriger la catégorie
        X_cat[:,j] = ind_j-1 # on commence les indices à 0 et non à 1
    # On choisit un type de nuage de points
    type_de_nuage = 1 # 0:aléatoire, 1:sourire, 2:deux zones
    # On met les couleurs dans les nuages de points
    if type_de_nuage==0: # random
        y = np.random.choice(a=[0,1], p=[q,p], size=m) # taille m de nombres aléatoires en {0,1}
    if type_de_nuage==1: # smiley
        #y = np.random.choice(a=[0,1], p=[q,p], size=m) # taille m de nombres aléatoires en {0,1}
        y = np.ones(shape=m) # taille m de nombres aléatoires en {0,1} # pour avoir visage rouge et yeux et bouche bleu
        #y = np.zeros(shape=m) # taille m de nombres aléatoires en {0,1} # pour avoir visage bleu et yeux et bouche rouge
        shift_eye_hor = 0.15
        shift_eye_ver = 0.15
        for i in range(m):
            u,v = tuple(X[i]) # coord. (u,v) en [0,1]x[0,1] en R^2
            if 0.45**2<=(u-0.5)**2+(v-0.5)**2 and (u-0.5)**2+(v-0.5)**2<=0.5**2: # tour de tête
                y[i] = 0 # 0:bleu, 1:rouge
            if 0.25**2<=(u-0.5)**2+(v-0.5)**2 and (u-0.5)**2+(v-0.5)**2<=0.35**2 and v<0.4: # sourire
                y[i] = 0
            if (u-0.5+shift_eye_hor)**2+(v-0.5-shift_eye_ver)**2<=0.10**2: # oeil gauche
                y[i] = 0
            if (u-0.5-shift_eye_hor)**2+(v-0.5-shift_eye_ver)**2<=0.10**2: # oeil droit
                y[i] = 0
    if type_de_nuage==2: # bas : bleu, haut : rouge
        for i in range(m):
            u,v = tuple(X[i]) # coord. (u,v) en [0,1]x[0,1] en R^2
            if v<=1/2: # en bas on met du bleu
                y[i] = 0
            if v>1/2: # en haut on met du rouge
                y[i] = 1
    # On crée un array sparse de 0 pour chaque catégorie
    grid_tot = np.zeros(resolutions,int) # taille (resolutions) = (resolutions[0],resolutions[1],...,resolutions[n-1]), c'est le nombre de points total dans chaque case
    grid_0   = np.zeros(resolutions,int) # taille (resolutions), on y met le nombre de valeurs 1 dans la case
    grid_1   = np.zeros(resolutions,int) # taille (resolutions), on y met le nombre de valeurs 1 dans la case
    for i in range(m):
        coord_cat = tuple(X_cat[i,:]) # coordonnées catégoriques (i_1,...,i_n) du point
        grid_tot[coord_cat] += 1 # on augmente le nombre de points
        if y[i]==0:grid_0[coord_cat] += 1 # on augmente le nombre de points à valeurs 0
        if y[i]==1:grid_1[coord_cat] += 1 # on augmente le nombre de points à valeurs 1
    # Calcul de la proportion du nombre de 1
    nombre_de_1 = np.count_nonzero(y)
    nombre_de_0 = m - nombre_de_1
    proportion  = nombre_de_1 / m # proportion de points qui sont des 1
    # On crée grid_f où on met les proportions de 0 et de 1
    grid_tot_temp = grid_tot
    grid_tot_temp[grid_tot==0] = -1
    grid_f = grid_1.astype(float)/grid_tot_temp.astype(float) # ici grid_f est à valeurs [0,1]
    # On décale f selon la proportion globale de 1 :
    grid_f = grid_f - proportion # ici grid_f est à valeurs [-1,1]
    grid_f = np.flip(m=grid_f, axis=1)



print(resolutions) # résolution 25x25
resolution_x = resolutions[0]
resolution_y = resolutions[1]


fig     = plt.figure()
ax      = fig.add_subplot(111)
div     = make_axes_locatable(ax)
cax     = div.append_axes('right', '5%', '5%')
x_min,x_max,y_max,y_min = 0,1,1,0
extent  = [x_min,x_max,y_max,y_min]
max_abs_value_f = np.max((np.abs(np.min(grid_f)),np.abs(np.max(grid_f))))
vmin    = -max_abs_value_f
vmax    =  max_abs_value_f
heatmap = ax.imshow(X=grid_f.T,cmap='bwr',extent=extent,alpha=1,vmin=vmin,vmax=vmax,origin='upper')
cbar    = fig.colorbar(heatmap, cax=cax)
tx      = ax.set_title('Phase 1 : rectangle aléatoire')
heatmap.set_clim(vmin, vmax)


min_resolution_rectangle = 2
min_w = min_resolution_rectangle
min_h = min_resolution_rectangle

x_pixel = random.randrange(resolution_x-min_w+1) # 0,1,2,3,...,resolution_x-min_w
y_pixel = random.randrange(resolution_y-min_h+1) # 0,1,2,3,...,resolution_y-min_h
w_pixel = random.randrange(min_w,resolution_x-x_pixel+1) # min_w,...,resolution_x-x_pixel
h_pixel = random.randrange(min_h,resolution_y-y_pixel+1) # min_h,...,resolution_y-y_pixel

new_x0_pixel = x_pixel
new_y0_pixel = y_pixel
new_x1_pixel = x_pixel + w_pixel
new_y1_pixel = y_pixel + h_pixel
new_x0      = new_x0_pixel/resolution_x
new_y0      = new_y0_pixel/resolution_y
new_x1      = new_x1_pixel/resolution_x
new_y1      = new_y1_pixel/resolution_y
new_w_pixel = w_pixel
new_h_pixel = h_pixel
new_w       = new_w_pixel /resolution_y
new_h       = new_h_pixel /resolution_x

x0     = x_pixel/resolution_x
y0     = y_pixel/resolution_y
width  = w_pixel/resolution_x
height = h_pixel/resolution_y
patch = plt.Rectangle(xy=(x0,y0),width=width,height=height,alpha=1,color='red',zorder=2,fill=False,linewidth=4)
line, = ax.plot([], [], lw=3,zorder=0)

def mean_or_sum_function(x,mean_or_sum):
    if mean_or_sum==0:
        return np.mean(x)
    if mean_or_sum==1:
        return np.sum(x)

i = 0
stop_simulation = 0
number_of_simulations = 1

max_value_found     = -np.inf
old_max_value_found = -np.inf

mean_or_sum = 0 # 0:np.mean, 1:np.sum

max_sigma_relatif = 0.12 # σ=12% de la largeur de l'image, donc 3σ=36% de l'image, 4σ=48% de l'image, ce qui donne un bon flou gaussien
max_sigma_pixel = max_sigma_relatif*resolution_x # = 3 pixels
number_of_steps_sigma = 7
step_sigma = 0

color_shift = 6
phase = 0
# phases (0,1) : on met un rectangle aléatoire et on le fait flasher rouge et vert
# phase 2 : on met le flou gaussien progressif
# phases (3,4) : maximiser np.mean(...) et np.max(...) par montée de gradient
# phase 5 : on défloue l'image et on mesure ce qu'on a trouvé
# phases (6,7) : maximiser np.mean(...) et np.max(...) par montée de gradient

def gen():
    global stop_simulation
    global number_of_simulations
    i = 0
    if stop_simulation==number_of_simulations:
        print("Simulation terminée")
    while stop_simulation<number_of_simulations:
        i += 1
        yield i
def init():
    line.set_data([], [])
    return line,patch
def animate(i):
    global phase
    global color_shift
    global max_sigma_pixel
    global number_of_steps_sigma
    global step_sigma
    global stop_simulation
    global max_value_found
    global old_max_value_found
    global mean_or_sum
    global grid_f_filtered
    global min_w
    global min_h
    global x_pixel,y_pixel,w_pixel,h_pixel
    global new_x0,new_y0,new_w,new_h
    global new_x0_pixel,new_x1_pixel,new_y0_pixel,new_y1_pixel

    if phase==0:
        x_pixel = random.randrange(resolution_x-min_w+1) # 0,1,2,3,...,resolution_x-min_w
        y_pixel = random.randrange(resolution_y-min_h+1) # 0,1,2,3,...,resolution_y-min_h
        w_pixel = random.randrange(min_w,resolution_x-x_pixel+1) # min_w,...,resolution_x-x_pixel
        h_pixel = random.randrange(min_h,resolution_y-y_pixel+1) # min_h,...,resolution_y-y_pixel

        new_x0_pixel = x_pixel
        new_y0_pixel = y_pixel
        new_x1_pixel = x_pixel + w_pixel
        new_y1_pixel = y_pixel + h_pixel

        phase=1
        print("phase 1")
        color_shift = 6
        tx.set_text('Phase 1 : rectangle aléatoire')
    elif phase==1:
        if color_shift%2==0:
            patch.set_color('red')
        if color_shift%2==1:
            patch.set_color('green')
        color_shift -= 1
        if color_shift==0:
            color_shift=6
            phase=2
            print("phase 1 -> 2")
    elif phase==2:
        step_sigma += 1
        sigma_filter_pixel = max_sigma_pixel*step_sigma/number_of_steps_sigma
        sigma_filter_relatif = sigma_filter_pixel/resolution_x
        # On met un flou gaussien
        grid_f_filtered = gaussian_filter(np.array(grid_f,dtype=float), sigma=sigma_filter_pixel)
        max_abs_value_f = np.max((np.abs(np.min(grid_f_filtered)),np.abs(np.max(grid_f_filtered))))
        vmin = -max_abs_value_f
        vmax =  max_abs_value_f
        heatmap.set_data(grid_f_filtered.T)
        heatmap.set_clim(vmin, vmax)
        tx.set_text('Phase 2 : σ = {:.2f}%'.format(100*sigma_filter_relatif))
        if step_sigma==number_of_steps_sigma:
            phase=3
            print("phase 2 -> 3")
            tx.set_text('Phase 3 : maximiser np.mean(...)')
    elif phase==3:
        # Pour le rectangle
        pos_rect = patch.get_bbox()
        x0       = pos_rect.x0
        x1       = pos_rect.x1
        y0       = pos_rect.y0
        y1       = pos_rect.y1
        x0_pixel = int(np.round(x0*resolution_x))
        y0_pixel = int(np.round(y0*resolution_y))
        x1_pixel = int(np.round(x1*resolution_x))
        y1_pixel = int(np.round(y1*resolution_y))
        w_pixel  = x1_pixel-x0_pixel
        h_pixel  = y1_pixel-y0_pixel
        new_x0_pixel = x0_pixel
        new_x1_pixel = x1_pixel
        new_y0_pixel = y0_pixel
        new_y1_pixel = y1_pixel

        # On regarde la somme selon toutes les déformations du rectangle
        sums = np.full(shape=9,fill_value=-np.inf) # 0:left, 1:right, 2:top, 3:bottom
        # Coin inférieur gauche :
        if x0_pixel>0:sums[0]              = mean_or_sum_function(x=grid_f_filtered[x0_pixel-1:x1_pixel,y0_pixel:y1_pixel],mean_or_sum=mean_or_sum) # left
        if x0_pixel<x1_pixel-min_w:sums[1] = mean_or_sum_function(x=grid_f_filtered[x0_pixel+1:x1_pixel,y0_pixel:y1_pixel],mean_or_sum=mean_or_sum) # right
        if y0_pixel<y1_pixel-min_h:sums[2] = mean_or_sum_function(x=grid_f_filtered[x0_pixel:x1_pixel,y0_pixel+1:y1_pixel],mean_or_sum=mean_or_sum) # top
        if y0_pixel>0:sums[3]              = mean_or_sum_function(x=grid_f_filtered[x0_pixel:x1_pixel,y0_pixel-1:y1_pixel],mean_or_sum=mean_or_sum) # bottom
        # Coin supérieur droit :
        if x0_pixel<x1_pixel-min_w:sums[4] = mean_or_sum_function(x=grid_f_filtered[x0_pixel:x1_pixel-1,y0_pixel:y1_pixel],mean_or_sum=mean_or_sum) # left
        if x1_pixel<resolution_x:sums[5]   = mean_or_sum_function(x=grid_f_filtered[x0_pixel:x1_pixel+1,y0_pixel:y1_pixel],mean_or_sum=mean_or_sum) # right
        if y1_pixel<resolution_y:sums[6]   = mean_or_sum_function(x=grid_f_filtered[x0_pixel:x1_pixel,y0_pixel:y1_pixel+1],mean_or_sum=mean_or_sum) # top
        if y0_pixel<y1_pixel-min_h:sums[7] = mean_or_sum_function(x=grid_f_filtered[x0_pixel:x1_pixel,y0_pixel:y1_pixel-1],mean_or_sum=mean_or_sum) # bottom
        # La somme sans déplacement :
        sums[8] = old_max_value_found # sans déplacement
        # On regarde ce qu'on fait
        maximum_sum = np.max(sums)
        if maximum_sum>-np.inf:
            do_this = np.argmax(sums) # donne le premier i=0,1,2,3,4,5,6,7 où on a un maximum
            old_max_value_found = sums[do_this]
            if do_this==0:new_x0_pixel,new_y0_pixel=x0_pixel-1,y0_pixel   # coin inférieur gauche va à gauche
            if do_this==1:new_x0_pixel,new_y0_pixel=x0_pixel+1,y0_pixel   # coin inférieur gauche va à droite
            if do_this==2:new_x0_pixel,new_y0_pixel=x0_pixel  ,y0_pixel+1 # coin inférieur gauche va en haut
            if do_this==3:new_x0_pixel,new_y0_pixel=x0_pixel  ,y0_pixel-1 # coin inférieur gauche va en bas
            if do_this==4:new_x1_pixel,new_y1_pixel=x1_pixel-1,y1_pixel   # coin supérieur droit va à gauche
            if do_this==5:new_x1_pixel,new_y1_pixel=x1_pixel+1,y1_pixel   # coin supérieur droit va à droite
            if do_this==6:new_x1_pixel,new_y1_pixel=x1_pixel  ,y1_pixel+1 # coin supérieur droit va en haut
            if do_this==7:new_x1_pixel,new_y1_pixel=x1_pixel  ,y1_pixel-1 # coin supérieur droit va en bas
            if do_this==8:new_x1_pixel,new_y1_pixel=x1_pixel  ,y1_pixel   # on ne bouge pas
            if maximum_sum>max_value_found: # si la somme augmente
                max_value_found=maximum_sum
            else: # si la somme stagne ou diminue
                if mean_or_sum==0: # si on était à np.mean on passe à np.max
                    mean_or_sum=1
                    if step_sigma>0:
                        print("phase 3 -> 4")
                        ax.set_title('Phase 4 : maximiser np.sum(...)')
                    if step_sigma==0:
                        print("phase 6 -> 7")
                        tx.set_text('Phase 7 : maximiser np.max(...)')
                elif mean_or_sum==1:
                    #print(sums)
                    if step_sigma==0:
                        print("Phase 7 -> 1")
                        phase=0
                        max_value_found = -np.inf
                        old_max_value_found = -np.inf
                        stop_simulation += 1 # on augmente le compteur d'arrêt
                    else:                        
                        phase = 5
                        mean_or_sum = 0
                        max_value_found = -np.inf
                        old_max_value_found = -np.inf
                        print("phase 4 -> 5")
                        tx.set_text('Phase 5 : σ = {:.2f}'.format(max_sigma_pixel))
    elif phase==5:
        step_sigma -= 1
        sigma_filter_pixel = max_sigma_pixel*step_sigma/number_of_steps_sigma
        sigma_filter_relatif = sigma_filter_pixel/resolution_x
        # On met un flou gaussien
        grid_f_filtered = gaussian_filter(np.array(grid_f,dtype=float), sigma=sigma_filter_pixel)
        max_abs_value_f = np.max((np.abs(np.min(grid_f_filtered)),np.abs(np.max(grid_f_filtered))))
        vmin = -max_abs_value_f
        vmax =  max_abs_value_f
        heatmap.set_data(grid_f_filtered.T)
        heatmap.set_clim(vmin, vmax)
        tx.set_text('Phase 5 : σ = {:.2f}%'.format(100*sigma_filter_relatif))
        if step_sigma==0:
            """
            phase=0
            print("phase 5 -> 1")
            tx.set_text('Phase 1 : rectangle aléatoire')
            stop_simulation += 1
            """
            phase=3
            print("Phase 5 -> 6")
            tx.set_text('Phase 6 : maximiser np.mean(...)')

    # Ici on met à jour les nouvelles valeurs
    new_w_pixel = new_x1_pixel-new_x0_pixel
    new_h_pixel = new_y1_pixel-new_y0_pixel
    new_x0      = new_x0_pixel/resolution_x
    new_y0      = new_y0_pixel/resolution_y
    new_x1      = new_x1_pixel/resolution_x
    new_y1      = new_y1_pixel/resolution_y
    new_w       = new_w_pixel /resolution_y
    new_h       = new_h_pixel /resolution_x
    # On met à jour le rectangle
    patch.set_bounds(new_x0,new_y0,new_w,new_h)
    ax.add_patch(patch)
    return line,patch

interval = 50 # en millisecondes
anim = FuncAnimation(fig, animate, init_func=init, frames=gen, interval=interval, blit=False)

show_or_gif = 0 # 0 pour afficher, 1 pour faire un gif
if show_or_gif==0: # pour faire un plot animé
    plt.show()
if show_or_gif==1: # pour faire un gif
    gif_file_name = "ile.gif"
    anim.save(gif_file_name, writer='imagemagick')

