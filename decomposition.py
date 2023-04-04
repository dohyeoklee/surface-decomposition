from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.ndimage import gaussian_filter
import time
import argparse
import json
import os

'''
#For debug
basis_size = 4
pop_size = 5
select_size = 3
'''

basis_size = 40
pop_size = 500
select_size = 300

def initialize_population():
	size = int(basis_size*pop_size)
	pi_list = np.random.permutation(np.linspace(0.05,0.0,size,endpoint=False))
	mu_x_list = np.random.permutation(np.linspace(-0.15,0.15,size))
	mu_y_list = np.random.permutation(np.linspace(-0.15,0.15,size))
	a_list = np.random.permutation(np.linspace(0.02,0.0,size,endpoint=False))
	init_pop_list = []
	for i in range(pop_size):
		param_list = []
		for j in range(basis_size):
			idx = i*basis_size + j
			param_list.append([pi_list[idx],mu_x_list[idx],mu_y_list[idx],a_list[idx]])
		init_pop_list.append(param_list)
	return init_pop_list

def ES_update(mesh_X,mesh_Y,Z,population_list):
	fitness_list = []
	for population in population_list:
		Z_bell = get_decomp_Z(mesh_X,mesh_Y,population)
		fitness = 1000 / np.linalg.norm(Z-Z_bell)
		fitness_list.append(fitness)

	print(max(fitness_list))
	fitness_sum = sum(fitness_list)
	prob_list = list(map((lambda x : x/fitness_sum),fitness_list))

	cdf_value = 0
	cdf_list = []
	for prob in prob_list:
		cdf_list.append(cdf_value+prob)
		cdf_value += prob
	cdf_list[-1] = 1.0

	selected = []
	for i in range(select_size):
		rnd = random.random()
		for j,cdf in enumerate(cdf_list):
			if rnd <= cdf:
				selected.append(j)
				break
	new_population_list = []
	selected = np.random.permutation(selected)
	pointer = 0
	for i in range(int(len(selected)/2)):
		pointer += 1
		if pointer == len(selected):
			break
		idx_1 = selected[2*i]
		idx_2 = selected[2*i+1]
		parent_1 = population_list[idx_1]
		parent_2 = population_list[idx_2]
		cpt_1,cpt_2 = np.sort(np.random.randint(4*basis_size,size=2))
		parent_1_arr = sum(parent_1,[])
		parent_2_arr = sum(parent_2,[])
		offspring_1_arr = \
		parent_1_arr[:cpt_1]+parent_2_arr[cpt_1:cpt_2]+parent_1_arr[cpt_2:]
		offspring_2_arr = \
		parent_2_arr[:cpt_1]+parent_1_arr[cpt_1:cpt_2]+parent_2_arr[cpt_2:]
		offspring_1 = np.reshape(np.array(offspring_1_arr),(basis_size,4))
		offspring_2 = np.reshape(np.array(offspring_2_arr),(basis_size,4))
		new_population_list.append(offspring_1.tolist())
		new_population_list.append(offspring_2.tolist())

	fit_arr = np.array(fitness_list)
	ascending_fitness_list = np.sort(fit_arr)[::-1]
	for i in range(pop_size-select_size):
		idx = np.where(fit_arr == ascending_fitness_list[i])
		new_population_list.append(population_list[idx[0][0]])

	new_population_list = mutation(new_population_list)
	return new_population_list

def mutation(population_list):
	size = int(basis_size*pop_size)
	pi_list = np.random.permutation(np.linspace(0.1,0.0,size,endpoint=False))
	mu_x_list = np.random.permutation(np.linspace(-0.2,0.2,size))
	mu_y_list = np.random.permutation(np.linspace(-0.2,0.2,size))
	a_list = np.random.permutation(np.linspace(0.05,0.0,size,endpoint=False))
	pop_mut_idxs = np.random.choice(np.arange(pop_size),int(pop_size*0.1))
	pointer = 0
	for pop_mut_idx in pop_mut_idxs:
		mut_idxs = np.random.choice(np.arange(basis_size),int(basis_size*0.1))
		for mut_idx in mut_idxs:
			population_list[pop_mut_idx][mut_idx] = [pi_list[pointer],mu_x_list[pointer],mu_y_list[pointer],a_list[pointer]]
			pointer += 1
	return population_list

def get_decomp_Z(mesh_X,mesh_Y,param_list):
	Z = []
	for X,Y in zip(mesh_X,mesh_Y):
		for x,y in zip(X,Y):
			z = 0
			for pi,mu_x,mu_y,a in param_list:
				z += bell_shape_func(x,y,pi,mu_x,mu_y,a)
			Z.append(z)
	return np.reshape(np.array(Z),(60,60))

def bell_shape_func(x,y,pi,mu_x,mu_y,a):
	z = pi*1/(2*np.pi*a**2)*1/((1+1/a**2*((x-mu_x)**2+(y-mu_y)**2))**(3/2))
	return z

def lonlat_to_xy(lon_mesh,lat_mesh,lon0,lat0):
	r = 6371000
	X = []
	Y = []
	for lon,lat in zip(lon_mesh.flatten(),lat_mesh.flatten()):
		x = r*np.cos(lat*np.pi/180)*(lon-lon0)*np.pi/180
		y = r*(lat-lat0)*np.pi/180
		X.append(x)
		Y.append(y)
	mesh_X = np.reshape(np.array(X),(60,60))
	mesh_Y = np.reshape(np.array(Y),(60,60))
	return mesh_X,mesh_Y

def normalize_XYZ(mesh_X,mesh_Y,Z):
	c_X = 10/np.linalg.norm(mesh_X)
	mesh_X = c_X*mesh_X
	d_X = np.mean(mesh_X)
	mesh_X = mesh_X - d_X

	c_Y = 10/np.linalg.norm(mesh_Y)
	mesh_Y = c_Y*mesh_Y
	d_Y = np.mean(mesh_Y)
	mesh_Y = mesh_Y - d_Y

	c_Z = 1000/np.linalg.norm(Z)
	Z = c_Z*Z

	norm_param=[c_X,d_X,c_Y,d_Y,c_Z]
	return mesh_X,mesh_Y,Z,norm_param

def smoothing_Z(Z):
	Z = gaussian_filter(Z,sigma=2)
	return Z

def reverse_normalize(param_list,norm_param):
	upscaled_param=[]
	c_X,d_X,c_Y,d_Y,c_Z=norm_param
	for param in param_list:
		pi,mu_x,mu_y,a = param
		mu_x=(mu_x-d_X)/c_X
		mu_y=(mu_y-d_Y)/c_Y
		a=a*(1/c_X**2+1/c_Y**2)**(0.5)
		pi=pi*1/c_Z*(1/c_X**2+1/c_Y**2)
		upscaled_param.append([pi,mu_x,mu_y,a])
	return upscaled_param

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path',type=str,default='./data/ETOPO1_Ice_g_gdal.nc')
	args = parser.parse_args()

	lon0,lon1,lat0,lat1 = 127,128,37,38
	i0 = 60*(lon0+180)+1 #18421
	i1 = 60*(lon1+180)+1 #18481
	j0 = 60*(lat0+90)+1 #7621
	j1 = 60*(lat1+90)+1 #7681

	nc = Dataset(args.data_path)

	lon = nc.variables['lon_map']
	lat = nc.variables['lat_map']
	height =nc.variables['ETOPO_height_mask']

	lon_seoul = lon[i0:i1]
	lat_seoul = lat[j0:j1]
	Z_ori = []
	for j in range(j0,j1):
		Z_ori.append(height[j][i0:i1])
	Z_ori = np.array(Z_ori)

	lon_seoul_mesh,lat_seoul_mesh = np.meshgrid(lon_seoul,lat_seoul)
	mesh_X_ori,mesh_Y_ori = lonlat_to_xy(lon_seoul_mesh,lat_seoul_mesh,lon0,lat0)
	mesh_X,mesh_Y,Z,norm_param = normalize_XYZ(mesh_X_ori,mesh_Y_ori,Z_ori)
	Z = smoothing_Z(Z)
	
	param_list = initialize_population()

	#optim_step = 2 #For debug
	optim_step = 300
	for step in range(optim_step):
		start_time = time.time()
		param_list = ES_update(mesh_X,mesh_Y,Z,param_list)
		dt = time.time() - start_time
		print("step {} complete / time: {}".format(step+1,dt))

	error_list = []
	for population in param_list:
		Z_bell = get_decomp_Z(mesh_X,mesh_Y,population)
		error = np.linalg.norm(Z-Z_bell)
		error_list.append(error)

	min_idx = np.argmin(np.array(error_list))
	result_param = reverse_normalize(param_list[min_idx],norm_param)

	RMSE_error = np.linalg.norm(Z-Z_bell)
	print("RMSE error: {}".format(RMSE_error))

	time_save = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
	save_path = './'+f'Error_{RMSE_error:.3f}'+f'_time_{time_save}'+'.json'
	with open(save_path,'w',encoding='utf-8') as f:
		json.dump(result_param,f,indent=4)

	#For visualize
	
	fig = plt.figure()
	ax_original = fig.add_subplot(121,projection='3d')
	ax_original.plot_surface(mesh_Y_ori,mesh_X_ori,Z_ori)

	ax_decomp = fig.add_subplot(122,projection='3d')
	Z_bell = get_decomp_Z(mesh_X_ori,mesh_Y_ori,result_param)
	ax_decomp.plot_surface(mesh_Y_ori,mesh_X_ori,Z_bell)

	plt.show()
	