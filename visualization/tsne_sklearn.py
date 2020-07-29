import os
import cv2
import matplotlib
import numpy as np
from glob import glob
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox



#ref: https://blog.csdn.net/lly1122334/java/article/details/90643054
BASE_DATA_FOLDER = "D:/AcouDigits/journalExtension/AcouDigits_FSL_test"
TEST_DATA_FOLDER = os.path.join(BASE_DATA_FOLDER, "WD_28")
TRAIN_DATA_FOLDER = "D:/AcouDigits/journalExtension/AcouDigits_FSL_train"


def visualize_scatter(data_2d, label_ids, id_to_label_dict, figsize=(28, 28)):
    plt.figure(figsize=figsize)
    plt.grid()

    nb_classes = len(np.unique(label_ids))
    colors = ['k', 'gray','lightcoral','r','chocolate',
            'darkorange','gold','olivedrab','y','darkseagreen',
            'limegreen','g','c','lightskyblue','dodgerblue',
            'royalblue','b','m','deeppink','crimson']

    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    c = colors[label_id],
                    marker='o',
                    label=id_to_label_dict[label_id],)

    # for label_id in np.unique(label_ids):
    #     plt.scatter(data_2d[np.where(label_ids == label_id), 0],
    #                 data_2d[np.where(label_ids == label_id), 1],
    #                 marker='o',
    #                 color= plt.cm.Set1(label_id / float(nb_classes)),
    #                 linewidth='1',
    #                 alpha=0.8,
    #                 label=id_to_label_dict[label_id],)
    plt.legend(loc='best')
    plt.show()

def visualize_scatter_with_images(X_2d_data, images, id_to_label_dict:dict, figsize=(28, 28), image_zoom=1):
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_2d_data, images):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(X_2d_data)
    ax.autoscale()
    plt.show()

def visualize_scatter_3d(label_ids, tsne_result_scaled, id_to_label_dict):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import animation
    fig = plt.figure(figsize=(28, 28))
    ax = fig.add_subplot(111,projection='3d')

    plt.grid()
        
    nb_classes = len(np.unique(label_ids))
        
    for label_id in np.unique(label_ids):
        ax.scatter(tsne_result_scaled[np.where(label_ids == label_id), 0],
                    tsne_result_scaled[np.where(label_ids == label_id), 1],
                    tsne_result_scaled[np.where(label_ids == label_id), 2],
                    alpha=0.8,
                    color= plt.cm.Set1(label_id / float(nb_classes)),
                    marker='o',
                    label=id_to_label_dict[label_id])
    ax.legend(loc='best')
    ax.view_init(25, 45)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-2.5, 2.5)
    plt.show()

def vis():
    images = []
    labels = []

    # test data
    for class_folder_name in os.listdir(TEST_DATA_FOLDER):
        class_folder_path = os.path.join(TEST_DATA_FOLDER, class_folder_name)
        for image_path in glob(os.path.join(class_folder_path, "*.jpg")):
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.flatten()
            images.append(image)
            labels.append(class_folder_name)

    # train data
    for idx, person_folder_name in enumerate(os.listdir(TRAIN_DATA_FOLDER)):
        person_folder_path = os.path.join(TRAIN_DATA_FOLDER, person_folder_name)
        for class_folder_name in os.listdir(person_folder_path):
            for image_path in glob(os.path.join(person_folder_path, class_folder_name, "*.jpg")):
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = image.flatten()
                images.append(image)
                labels.append(str((1)*10+int(class_folder_name)))
                # labels.append(str((idx+1)*10+int(class_folder_name)))

    images = np.array(images)
    labels = np.array(labels)

    images_scaled = StandardScaler().fit_transform(images)

    label_to_id_dict = {v:i for i,v in enumerate(np.unique(labels))}
    id_to_label_dict = {v: k for k, v in label_to_id_dict.items()}
    label_ids = np.array([label_to_id_dict[x] for x in labels])

    pca = PCA(n_components=180)
    pca_result = pca.fit_transform(images_scaled)

    tsne = TSNE(n_components=2, perplexity=40.0)
    tsne_result = tsne.fit_transform(pca_result)
    tsne_result_scaled = StandardScaler().fit_transform(tsne_result)

    # # point-part
    visualize_scatter(tsne_result_scaled, label_ids, id_to_label_dict)

    # # image-part
    # visualize_scatter_with_images(tsne_result_scaled, images = [np.reshape(i, (28, 28)) for i in images], 
    #                             id_to_label_dict = id_to_label_dict, image_zoom=0.7)

    # 3d-part
    # tsne = TSNE(n_components=3)
    # tsne_result = tsne.fit_transform(pca_result)
    # tsne_result_scaled = StandardScaler().fit_transform(tsne_result)
    # visualize_scatter_3d(label_ids, tsne_result_scaled, id_to_label_dict)


if __name__ == "__main__":
    vis()