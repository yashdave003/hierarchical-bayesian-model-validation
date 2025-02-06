import numpy as np
import pywt
import matplotlib.pyplot as plt
import random
from scipy import stats
import random
import pywt.data
from PIL import Image
import pandas as pd
import seaborn as sns
import seaborn as sns
import os
import pickle




def FullProcess(path, layer = "All", wv = "db1"):
    #Taken from demo prints layers of images under a specific wavelet
    image = Image.open(path).convert('L')
    image= (image-np.mean(image))/np.std(image)
    plt.imshow(image, interpolation="nearest", cmap=plt.cm.gray)
    high_depth_image = pywt.wavedec2(image, wv)
    n = len(high_depth_image)
    if layer == "All":
        for j in range(1, n):
            fig = plt.figure(figsize=(12, 3))
            level_data = high_depth_image[j]
            for i, a in enumerate([i for i in level_data]):
                ax = fig.add_subplot(1, 4, i + 1)
                ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
                ax.set_xticks([])
                ax.set_yticks([])

            fig.tight_layout()
            plt.show()
    elif layer != "None":
        fig = plt.figure(figsize=(12, 3))
        level_data = high_depth_image[layer]
        for i, a in enumerate([i for i in level_data]):
            ax = fig.add_subplot(1, 4, i + 1)
            ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        plt.show()
    return n


def ReturnLayer(image, layer = "Max", seperate = True, print_k = True, wv= "db1"):
    
    high_depth_image = pywt.wavedec2(image, wv)
    k = len(high_depth_image)
    if print_k:
        print(k)
    if layer == "Max":
        level = high_depth_image[k-1]
    else:
        level = high_depth_image[layer-1]
    if seperate:
        return np.array(level)
    else:
        a = level[0]
        for arr in level[1:]:
            #print(a.shape, arr.shape)
            a = np.append(a, arr, axis= 1)
        level = a
        return np.array(level)


def normalize_image(path):
    image = Image.open(path).convert('L')
    image= (image-np.mean(image))/np.std(image)
    return image



def density_of_layer(flat_layers, method = 0.2, diagonal = True, without_diagonal = False, title = ""):
    
    layer_names = ["Horizantal", "Vertical", "Diagonal"]
    
    if(diagonal):
        fig, axes = plt.subplots(1, 2, figsize=(24, 6))
        fig.suptitle(title)
        for i in range(len(flat_layers)):
            axes[1].set_ylim(bottom = 10**-5, top= 10)
            axes[1].set_xlim(left = -3, right= 3)
            sns.kdeplot(ax = axes[0], x = flat_layers[i], bw_method = method, label = layer_names[i])
            sns.kdeplot(ax = axes[1], x = flat_layers[i], bw_method = method, log_scale=[False, True], label = layer_names[i])
            axes[0].legend()
            axes[1].legend()
    if(without_diagonal):
        fig, axes = plt.subplots(1, 2, figsize=(24, 6))
        fig.suptitle(title)
        for i in range(len(flat_layers[:-1])):
            axes[1].set_ylim(bottom = 10**-5, top= 10)
            axes[1].set_xlim(left = -3, right= 3)
            sns.kdeplot(ax = axes[0], x = flat_layers[i], bw_method = method,label = layer_names[i])
            sns.kdeplot(ax = axes[1], x = flat_layers[i], bw_method = method, log_scale=[False, True],label = layer_names[i])
            axes[0].legend()
            axes[1].legend()
            
    return np.cov(flat_layers)


def convert_to_wavelet_basis(folder_dir,  normalized = True, basis="db1"):
    file_list = [os.path.join(folder_dir, filename) for filename in os.listdir(folder_dir)]
    file_names = os.listdir(folder_dir)
    #Setup df Dict
    df_dict = dict()
    image = Image.open(file_list[0]).convert('L')
    first_image = pywt.wavedec2(image, basis)
    layer_len = len(first_image)
    print(str(layer_len) + " layers being used")
    for i in range(layer_len):
        #df = pd.DataFrame(columns=["Image ID", "Orientation", "Data", "Flattened Data"])
        df = pd.DataFrame(columns=["Image ID", "Orientation", "Data"])
        df_dict[i+1] = df
    
    
    #Fill DF DICT
    for k in range(len(file_list)):
        image = Image.open(file_list[k]).convert('L')
        image = np.array(image)
        if normalized:
            std= np.std(image)
            mean = np.mean(image)
            image = (image- mean)/std 
            #image = image * 255
            
        name = file_names[k].split(".")[0]
        transformed = pywt.wavedec2(image, basis)
        #df_dict[1].loc[len(df_dict[1].index)] = [name, "ONELAYER", np.array(transformed[0][0]), np.array(transformed[0][0]).flatten()]
        df_dict[1].loc[len(df_dict[1].index)] = [name, "ONELAYER", np.array(transformed[0][0])]
        direction_names = ['Horizontal detail', 'Vertical detail', 'Diagonal detail']

        for i in range(1, layer_len): 
            for j in range(len(transformed[i])):
                arr = np.array(transformed[i][j])
                df_dict[i+1].loc[len(df_dict[i+1].index)] = [name, direction_names[j], arr]
                #df_dict[i+1].loc[len(df_dict[i+1].index)] = [name, direction_names[j], arr.flatten()]

    return df_dict


def dict_to_pickle(converted_directory, converted, name):
    filename = name
    filename = os.path.join(converted_directory, filename)
    #converted[keys].to_csv(filename, sep=',', index=False, encoding='utf-8')
    with open(filename+".pickle", 'wb') as handle:
        pickle.dump(converted, handle, protocol=pickle.HIGHEST_PROTOCOL)