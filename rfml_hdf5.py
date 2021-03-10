# hdf5 wrapper functions for handling typical hdf5 data files written by finite difference solver NGA.

import h5py
import os
import numpy as np


### ====== General functions ====== ###
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


### ====== Attribute functions ====== ###
def hdf5_read_attribute(filename, parentpath, attribute_name):
	"""
		Returns attribute associated with group or dataset as numpy array

		filename 		: name of hdf5 file
		parentpath 		: path of group or data with which the attribute is associated
		attribute_name 	: name of the attribute

		return type is list, since attributes are one dimensional
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
	try:
		fid = h5py.File(filename, 'r+')
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
	dimensions = np.array(attribute_data).shape
	space_id = h5py.h5s.create_simple(dimensions, dimensions)
	attribute_id = h5py.h5a.create(parent_id, bytes(attribute_name, encoding='utf-8'), write_type, space_id)
	attribute_id.write(np.array(attribute_data).astype(write_type))
	attribute_id.close()


def __hdf5_reclaim_space(filename):
	"""
		Private method that reclaims empty space of hdf5 file created by deleting objects inside
	"""
	# repack to temp file
	cmd = "h5repack -i " + filename + "-o temp.h5 && mv temp.h5 " + filename
	os.system(cmd)
