import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import pymesh
from sklearn.cluster import MeanShift, estimate_bandwidth
import os
import json
import sys

class KeypointExtractor:
    def __init__(self, src, dst=''):
        self.mesh_path = src
        self.save_path = dst
        self.mesh = None
        self.ftr_type = None
        self.vertice, self.normal, self.gaussian = None, None, None
        self.__initialize()

        self.vertice_decomposed, self.normal_decomposed, self.gaussian_decomposed, self.label_decomposed = None, None, None, None
        self.vertice_keypoint, self.normal_keypoint, self.gaussian_keypoint = None, None, None
        self.class_keypoint = None

    def __initialize(self):
        self.__load_mesh()
        self.mesh.add_attribute("vertex_gaussian_curvature")
        self.mesh.add_attribute("vertex_normal")
        self.gaussian = self.mesh.get_attribute("vertex_gaussian_curvature")
        self.normal = self.mesh.get_vertex_attribute("vertex_normal")
        self.vertice = self.mesh.vertices

    def __load_mesh(self):
        mesh = pymesh.load_mesh(self.mesh_path)
        # remove duplicated vertex and return mesh
        mesh, _ = pymesh.remove_duplicated_vertices(mesh, tol=1e-12, importance=None)
        # remove collinear and return mesh
        mesh, _ = pymesh.remove_degenerated_triangles(mesh)
        self.mesh = mesh


    def __meanshift(self, ftr_type):
        # meanshift algorithm is used to divide the vertice into subgroup
        # quantile and n_samples can be set depeding on your needs
        if ftr_type == 'curvature':
            bandwidth = estimate_bandwidth(self.vertice_decomposed, quantile=0.06, n_samples=500)
        elif ftr_type == 'texture':
            bandwidth = estimate_bandwidth(self.vertice_decomposed, quantile=0.06, n_samples=200)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(self.vertice_decomposed)
        labels = ms.labels_
        self.label_decomposed = labels
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        print(f'Estimated number of {ftr_type} clusters = {n_clusters_}')

    def __decompose_mesh(self, ftr_type):
        if ftr_type == 'curvature':
            idx = np.argwhere(self.gaussian > np.mean(self.gaussian) + np.std(self.gaussian)).ravel()
        elif ftr_type == 'texture':
            idx = np.argwhere(self.gaussian < np.mean(self.gaussian)).ravel()
        self.gaussian_decomposed =  self.gaussian[idx]
        self.vertice_decomposed = self.vertice[idx]
        self.normal_decomposed = self.normal[idx]
    
    def __run_meanshift(self, ftr_type):
        # Meanshift: get each vertice's cluster label
        self.__meanshift(ftr_type)
        # return closet normal value's idx of each group
        closest_idxs = self.__get_closest_to_normal_mean()

        # Keypoints: get vertices in each label a.k.a. keypoints
        self.vertice_keypoint = self.vertice_decomposed[closest_idxs]
        self.normal_keypoint = self.normal_decomposed[closest_idxs]
        self.gaussian_keypoint = self.gaussian_decomposed[closest_idxs]
        

    def run(self, ftr_type):
        if ftr_type == 'standard':
            self.ftr_type = ftr_type
            self.__decompose_mesh(ftr_type='curvature')
            self.__run_meanshift(ftr_type='curvature')
            curvature_keypoint, curvature_normal, curvature_gaussian = self.vertice_keypoint.copy(), self.normal_keypoint.copy(), self.gaussian_keypoint.copy()

            self.__decompose_mesh(ftr_type='texture')
            self.__run_meanshift(ftr_type='texture')
            texture_keypoint, texture_normal, texture_gaussian = self.vertice_keypoint.copy(), self.normal_keypoint.copy(), self.gaussian_keypoint.copy()

            self.vertice_keypoint = np.concatenate((curvature_keypoint, texture_keypoint), axis=0)
            self.normal_keypoint = np.concatenate((curvature_normal, texture_normal), axis=0)
            self.gaussian_keypoint = np.concatenate((curvature_gaussian, texture_gaussian), axis=0)

        else:
            self.ftr_type = ftr_type
            self.__decompose_mesh(ftr_type)
            self.__run_meanshift(ftr_type)
            self.label_unique = np.unique(self.label_decomposed)

        print(f'Estimated number of clusters = {self.vertice_keypoint.shape[0]}')

    def visualize_decomposed_mesh(self):
        if self.ftr_type == 'standard':
            print('Warning: Standard method cannot visualize decomposed mesh')
            return

        xs, ys, zs = self.vertice_decomposed[:, 0], self.vertice_decomposed[:, 1], self.vertice_decomposed[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        n_clusters = len(np.unique(self.label_decomposed))
        color_num = [0.37895601, 0.66094761, 0.73378628]
        # print(color_num)
        for i in range(n_clusters):
            color_num = np.random.rand(3,)
            idx = np.where(self.label_decomposed == i)
            x_cluster, y_cluster, z_cluster = xs[idx], ys[idx], zs[idx]
            ax.scatter(x_cluster,y_cluster,z_cluster,color=color_num, label= str('Group ') + str(i), alpha = 0.3)
            ax.scatter(self.vertice_keypoint[i,0],self.vertice_keypoint[i,1], self.vertice_keypoint[i,2], color=color_num, marker='*',s=200)
        ax.legend()
        plt.axis('off')
        plt.show()

    def visualize_mesh(self, keypoint = False):
        idx_sampled = np.random.choice(self.vertice.shape[0], size=7000, replace=False)
        vertice_sampled = self.vertice[idx_sampled]
        xs, ys, zs = vertice_sampled[:, 0], vertice_sampled[:, 1], vertice_sampled[:, 2]
        xs_kp, ys_kp, zs_kp = self.vertice_keypoint[:, 0], self.vertice_keypoint[:, 1], self.vertice_keypoint[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(xs, ys, zs, color='gray', s=2, alpha=0.3)
        if (keypoint):
            ax.scatter(xs_kp, ys_kp, zs_kp, color='red', s=5, alpha=1)
        plt.axis('off')
        plt.show()


    def __get_closest_to_normal_mean(self):
        n_clusters = len(np.unique(self.label_decomposed))
        closet_value_idx_lst = []
        
        for i in range(n_clusters):
            idx = np.where(self.label_decomposed == i)
            normal_cluster = self.normal_decomposed[idx]
            normal_mean = np.mean(normal_cluster, axis=0)
            closet_value_idx = self.__find_nearest(self.normal_decomposed, idx[0], normal_mean)
            closet_value_idx_lst.append(closet_value_idx)
        return closet_value_idx_lst


    def __find_nearest(self, normal, idx, normal_mean):
        min_idx, min_distance = None, None  
        for i in idx:
            dist = LA.norm(normal[i] - normal_mean)
            if min_distance == None or dist < min_distance:
                min_idx = i
                min_distance = dist
        return min_idx

    def __write_back_vertice_keypoint(self):
        np.save(f'{self.save_path}/{self.vertice_keypoint.shape[0]}_vertice.npy', self.vertice_keypoint)

    def __write_back_normal_keypoint(self):
        np.save(f'{self.save_path}/{self.normal_keypoint.shape[0]}_normal.npy', self.normal_keypoint)
    
    def __write_back_class_keypoint(self):
        with open(f"{self.save_path}/{self.vertice_keypoint.shape[0]}_class.txt", 'w') as f:
            for idx in range(len(self.class_keypoint)):
                if idx < (len(self.class_keypoint) - 1):
                    f.write(str(self.class_keypoint[idx]) + '\n')
                else:
                    f.write(str(self.class_keypoint[idx]))

    def __write_back_json(self):
        json_file_name = f'{self.save_path}/{self.vertice_keypoint.shape[0]}_note.json'
        file_exists = os.path.exists(json_file_name)
        if (file_exists):
            os.remove(json_file_name)
        with open(json_file_name, 'w') as f:
            data = {"number": self.vertice_keypoint.shape[0], 'name': self.class_keypoint}
            json.dump(data, f, indent=2)

    def set_save_path(self, dst):
        self.save_path = dst

    def set_class_name(self, prefix):
        keypoint_size = self.vertice_keypoint.shape[0]
        self.class_keypoint = [str(prefix) + f'_{i}' for i in range(keypoint_size)]

    def save(self):
        self.__write_back_vertice_keypoint()
        self.__write_back_normal_keypoint()
        self.__write_back_class_keypoint()
        self.__write_back_json()

if __name__ == '__main__':
    extractor = KeypointExtractor(sys.argv[0])
    extractor.run('standard')
    extractor.set_class_name(prefix='kc')
    # extractor.visualize_decomposed_mesh()
    extractor.visualize_mesh(keypoint=True)
    extractor.set_save_path('../../data/key_component')
    extractor.save()
