# hdf5 wrapper functions for handling typical hdf5 data files written by finite difference solver NGA.

import h5py
import os
import numpy as np


### ====== RFML specific hdf5 functions and wrappers ====== ###
def is_rfml_hdf5(filename):
	"""
		Checks file file_name against conventions of RFML hdf5 data file:
		1. should contain group '/data'
		2. should contain attributes '/data/dimensions','/data/datafield_names'
	"""
	fid = h5py.File(filename, 'r')
	try:
		# Should contain group 'data'
		status = False
		if ('data' in list(fid)):
			# data must contain attributes 'datafield_names' and 'dimensions'
			gid = fid['data']
			attrManager = gid.attrs
			attrList = list(attrManager)
			if ('datafield_names' in attrList and 'dimensions' in attrList):
				status = True
	finally:
		fid.close()
		return status

def hdf5_NGA_config_write(filename, simname, x, y, z, icyl, iper, mask, endian='little-endian'):
	"""
		Wrapper function to write configuration file for NGA simulations in RFML hdf5 convention
		function is structured same as matlab function in RFML matlab repository.

		filename : name (and path) of config file to be written
		simname	 : a long string containing simulation name keywords for init_flow
		x, y, z	 : coordinate arrays of cell faces
		icyl	 : integer flag for the use of cylindrical coordinates
		iper	 : 3 x 1 array for periodicity in the three directions
		mask	 : 2D numpy integer array of shape (x.size - 1, y.size - 1)
		endian	 : 'little-endian' or 'big-endian'
	"""
	fid = h5py.File(filename, 'w')
	try:
		group_id = fid.create_group('/data')
		# Attributes
		#-> simulation name
		write_type = h5py.h5t.C_S1.copy()
		write_type.set_size(64)
		write_type.set_strpad(h5py.h5t.STR_SPACEPAD)
		__oned_attribute_write(group_id, 'simulation_name', write_type, simname)
		#-> parameters
		if(endian=='big-endian'):
			write_type = h5py.h5t.STD_I32BE
		else:
			write_type = h5py.h5t.STD_I32LE
		nx = x.size - 1
		ny = y.size - 1
		nz = z.size - 1
		parameters = np.array([icyl, iper[0], iper[1], iper[2], nx, ny, nz]).astype(write_type)
		__oned_attribute_write(group_id, 'parameters', write_type, parameters)

		# Datasets
		#-> mask : type integer
		__dataset_write(group_id, 'mask', write_type, mask)

		#-> x, y, z : type float
		if(endian=='big-endian'):
			write_type = h5py.h5t.IEEE_F64BE
		else:
			write_type = h5py.h5t.IEEE_F64LE
		__dataset_write(group_id, 'x', write_type, x)
		__dataset_write(group_id, 'y', write_type, y)
		__dataset_write(group_id, 'z', write_type, z)
		__dataset_write(group_id, 'xm', write_type, (x[1:nx] + x[0:nx-1])/2.0)
		__dataset_write(group_id, 'ym', write_type, (y[1:nx] + y[0:nx-1])/2.0)
		__dataset_write(group_id, 'zm', write_type, (z[1:nx] + z[0:nx-1])/2.0)
	finally:
		fid.close()

def hdf5_NGA_data_write(filename, data, t0, dt, endian='little-endian'):
	"""
		Wrapper function to write data file for NGA simulations in RFML hdf5 convention
		function is structured in the same manner as matlab function in the RFML matlab repository.

		filename : name (and path) of data file to be written
		data	 : dictonary with key = datafield name, value = data in numpy ndarray
		t0		 : initial time in time_variables attribute (time_variables[1])
		dt		 : initial time step in time_variables attribute (time_variables[2])
	"""
	fid = h5py.File(filename, 'w')
	try:
		group_id = fid.create_group('/data')
		# Attributes
		#-> datafield_names
		write_type = h5py.h5t.C_S1.copy()
		write_type.set_size(8)
		write_type.set_strpad(h5py.h5t.STR_SPACEPAD)
		datafield_names = list(data.keys())
		__oned_attribute_write(group_id, 'datafield_names', write_type, datafield_names)
		# -> dimensions
		if(endian=='big-endian'):
			write_type = h5py.h5t.STD_I32BE
		else:
			write_type = h5py.h5t.STD_I32LE
		nvars = len(datafield_names)
		data_dims = data[datafield_names[0]].shape
		dimensions = np.append(np.array(data_dims),nvars)
		__oned_attribute_write(group_id, 'dimensions', write_type, dimensions)
		#-> time_variables
		if (endian == 'big-endian'):
			write_type = h5py.h5t.IEEE_F64BE
		else:
			write_type = h5py.h5t.IEEE_F64LE
		tvars = np.array([dt, t0])
		__oned_attribute_write(group_id, 'time_variables', write_type, tvars)

		# datasets
		for ivar in range(0,nvars):
			varname = datafield_names[ivar]
			__dataset_write(group_id, varname, write_type, data[varname])
	finally:
		fid.close()

### ====== Attribute functions ====== ###
def hdf5_read_attribute(filename, parentpath, attribute_name):
	"""
		Returns attribute associated with group or dataset as numpy array

		filename 		: name of hdf5 file
		parentpath 		: path of group or data with which the attribute is associated
		attribute_name 	: name of the attribute

		return type numpy array, since attributes are one dimensional
	"""
	# Returns contents of attribute of a group
	fid = h5py.File(filename, 'r')
	parent = fid.get(parentpath)
	attribute_data = np.array(parent.attrs[attribute_name])
	fid.close()
	return attribute_data


def hdf5_read_variables(filename):
	"""
		Returns variable names in group 'data'
	"""
	if (is_rfml_hdf5(filename)):
		varnames = hdf5_read_attribute(filename, '/data/', 'datafield_names')
		varnames = [x.decode('utf-8') for x in varnames]
		return varnames
	else:
		raise Exception(filename + 'not an RFML hdf5 file')


def hdf5_add_attribute(filename, parentpath, attribute_name, attribute_type, attribute_data):
	"""
		Adds new attribute to the specified location
		
		filename : Name of hdf5 file
		parentpath : path of the parent group or dataset to which the attribute is attached
		attribute_name : (string) name of the attribute, encoded with 'utf-8' while written
		attribute_type : hdf5 type kind or convertible from native python data types
						 must contain all necessary info, such as string type with necessary padding and length
		attribute_data : numpy array with attribute data to be written
	"""
	fid = h5py.File(filename, 'r+')
	try:
		parent = fid.get(parentpath)
		attribute_manager = parent.attrs

		if (attribute_name in list(attribute_manager)):
			raise AttributeError('Attribute already exists! please use hdf5_replace_attribute method instead')
		else:
			__oned_attribute_write(parent, attribute_name, attribute_type, attribute_data)
	except Exception as err:
		print(err)
	finally:
		fid.close()


def hdf5_replace_attribute(filename, parentpath, attribute_name, attribute_data):
	"""
		Adds new attribute to the specified location

		filename : Name of hdf5 file
		parentpath : path of the parent group or dataset to which the attribute is attached
		attribute_name : (string) name of the attribute, encoded with 'utf-8' while written
		attribute_type : hdf5 type kind or convertible from native python data types
						 must contain all necessary info, such as string type with necessary padding and length
		attribute_data : numpy array with attribute data to be written
	"""
	fid = h5py.File(filename, 'r+')
	parent = fid.get(parentpath)
	attribute_manager = parent.attrs

	try:
		if (attribute_name in list(attribute_manager)):
			# attribute type - copy from old attribute
			attribute_id = attribute_manager.get_id(attribute_name)
			attribute_type = attribute_id.get_type()

			# delete old attribute and replace with new one
			del attribute_manager[attribute_name]
			__oned_attribute_write(parent, attribute_name, attribute_type, np.array(attribute_data))

			fid.close()
			__hdf5_reclaim_space(filename)
		else:
			raise AttributeError('Attribute does not exists! please use hdf5_add_attribute method instead')
	except Exception as err:
		fid.close()
		print(err)

### ====== Dataset functions ====== ###
def hdf5_add_dataset(filename, parentpath, dataset_name, data_type, dataset_data):
	"""
		Adds a new dataset to existing hdf5 file
	"""
	assert(is_rfml_hdf5(filename))
	fid = h5py.File(filename, 'r+')
	try:
		# Write dataset
		parent = fid.get(parentpath)
		__dataset_write(parent, dataset_name, data_type, dataset_data)

		# Change attributes 'dimensions' and 'datafield_names'
		variable_names = hdf5_read_variables(filename)
		variable_names = np.append(variable_names, np.array([dataset_name]).astype(type(variable_names)))
		dimensions = hdf5_read_attribute(filename, parentpath, 'dimensions')
		dimensions[3] = dimensions[3] + 1
		fid.close()

		hdf5_replace_attribute(filename, parentpath, 'datafield_names', variable_names)
		hdf5_replace_attribute(filename, parentpath, 'dimensions', dimensions)
	except Exception as err:
		print(err)
		fid.close()

def hdf5_remove_dataset(filename, parentpath, dataset_name):
	"""
		Removes a dataset from hdf5 file.
		Also modifies the attributes 'dimensions' and 'datafield_names'

		filename: path to the hdf5 file
		parentpath: path to group which contains the dataset
		datafield_name: name of the dataset to be remmoved
	"""
	fid = h5py.File(filename,'r+')
	parent = fid.get(parentpath)

	try:
		if(dataset_name in list(parent)):
			del parent[dataset_name]

			# change attributes if rfml_hdf5 file
			if(is_rfml_hdf5(filename)):
				attribute_manager = parent.attrs
				# datafield_names
				dataset_names = attribute_manager['datafield_names']
				dataset_names_str = [x.decode('UTF-8') for x in dataset_names]
				idx = dataset_names_str.index(dataset_name)
				dataset_names = np.delete(dataset_names, idx)
				# dimensions
				dimensions = attribute_manager['dimensions']
				dimensions[3] = dimensions[3] - 1

			fid.close()
			__hdf5_reclaim_space(filename)

			# replace attributes
			if(is_rfml_hdf5(filename)):
				hdf5_replace_attribute(filename, parentpath, 'datafield_names', dataset_names)
				hdf5_replace_attribute(filename, parentpath, 'dimensions', dimensions)

		else:
			raise NameError("dataset " + datafield_name + "does not exist")
	except Exception as err:
		print(err)
		fid.close()

def hdf5_read_dataset(filename, parentpath, dataset_name):
	"""
		Reads a dataset and returns as numpy ndarray

		filename: name of hdf5 file
		parentpath: path of group that contains the dataset
		dataset_name : name of the dataset to be read
	"""
	fid = h5py.File(filename, 'r')
	field = np.array(fid.get(parentpath)[dataset_name])
	fid.close()
	return field

def hdf5_overwrite_dataset(filename, parentpath, dataset_name, new_dataset):
	"""
		Overwrites an existing dataset in hdf5 file with new data
		Does not check for shape consistency

		filename : name of hdf5 file
		parentpath: path of group that contains the dataset
		dataset_name: name of the dataset to be overwritten
		new_dataset: new data to be written
	"""
	fid = h5py.File(filename, 'r+')
	try:
		group_id = fid.get(parentpath)
		data_id = group_id[dataset_name]
		data_id[...] = new_dataset
	finally:
		fid.close()

def hdf5_get_dataset_type(filename, group_path, dataset_name):
	"""
		Returns native hdf5 datatype of a dataset

		filename : name of hdf5 file
		group_path : path inside the file of the containing group ('/' if no group)
		dataset_name : name of the dataset
	"""
	fid = h5py.File(filename, 'r')
	datatype_id = None
	try:
		group_obj_id = fid.get(group_path).id
		dataset_obj_id = h5py.h5d.open(group_obj_id, __convert_name(dataset_name))
		datatype_id = dataset_obj_id.get_type()
	except Exception as err:
		print(err)
	finally:
		fid.close()
		return datatype_id

### === Private functions === ###
def __oned_attribute_write(parent, attribute_name, write_type, attribute_data):
	"""
		Private method that writes attribute to group
		- Uses low level h5py API
		- Access through public methods 'hdf5_NGA_data_write', 'hdf5_NGA_config_write', or 'hdf5_add_attribute'

		parent : high level identifier for parent group/dataset to which attribute is to be attached
		attribute_name : name of the attribute to be written
		write_type : hdf5 type in which attribute is to be written
		attribute_data : attribute data - must be convertible to numpy.ndarray
	"""
	parent_id = parent.id
	dimensions = (np.array(attribute_data).size,)	# tuple
	space_id = h5py.h5s.create_simple(dimensions, dimensions)
	attribute_id = h5py.h5a.create(parent_id, __convert_name(attribute_name), write_type, space_id)
	attribute_id.write(np.array(attribute_data).astype(write_type))
	attribute_id.close()

def __dataset_write(parent, dataset_name, write_type, write_data):
	"""
		Private method to write dataset
		- Uses low level h5py API
		- Accessed by public methods 'hdf5_add_field', 'hdf5_NGA_data_write'

		parent: high level group reference where dataset is to be added.
				Attributes 'datafield_names' and 'dimensions' are not modified inside this function.
		dataset_name : Name of the dataset
		write_type	: data type
		write_data	: data to be written, must be convertible to nparray
	"""
	parent_id = parent.id
	dimensions = np.array(write_data).shape
	space_id = h5py.h5s.create_simple(dimensions, dimensions)
	data_id = h5py.h5d.create(parent_id, __convert_name(dataset_name), write_type, space_id)
	data_id.write(h5py.h5s.ALL, space_id, np.array(write_data).astype(write_type), write_type)


def __hdf5_reclaim_space(filename):
	"""
		Private method that reclaims empty space of hdf5 file created by deleting objects inside
	"""
	# repack to temp file
	cmd = "h5repack -i " + filename + "-o temp.h5 && mv temp.h5 " + filename
	os.system(cmd)

def __convert_name(string_to_convert):
	return bytes(string_to_convert, encoding='utf-8')