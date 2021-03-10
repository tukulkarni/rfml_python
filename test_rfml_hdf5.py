import sys
import unittest
from rfml_hdf5 import *


class TestHdf5Functions(unittest.TestCase):
	sample_file = 'sample_rfml_hdf5.h5'
	generic_file = 'generic_hdf5.h5'
	dataset_names = ['C']
	dataset_dims = (10, 10, 10)

	def setUp(self):
		# Create temporary files for unit testing
		file_id = h5py.h5f.create(bytes(self.sample_file, encoding='utf-8'), h5py.h5f.ACC_TRUNC)
		group_id = h5py.h5g.create(file_id, name=bytes('data', encoding='utf-8'))
		space_id = h5py.h5s.create_simple(self.dataset_dims, self.dataset_dims)

		# data set
		data_id = h5py.h5d.create(group_id, bytes('C', encoding='utf-8'), h5py.h5t.IEEE_F64LE, space_id)
		dataset = np.zeros(self.dataset_dims, np.double)
		data_id.write(h5py.h5s.ALL, h5py.h5s.ALL, dataset)  # all in memory -> all in file
		data_id.close()

		# attributes
		space_id = h5py.h5s.create_simple((4,), (4,))
		attr_dimensions = h5py.h5a.create(group_id, bytes('dimensions', encoding='utf-8'), h5py.h5t.STD_I32LE, space_id)
		attr_dimensions.write(np.append(np.asarray(self.dataset_dims), 1))
		attr_dimensions.close()

		str_type = h5py.h5t.C_S1.copy()
		str_type.set_size(8)
		str_type.set_strpad(h5py.h5t.STR_SPACEPAD)

		space_id = h5py.h5s.create_simple((1,))
		attr_names = h5py.h5a.create(group_id, bytes('datafield_names', encoding='utf-8'), str_type, space_id)
		attr_names.write(np.array([x.ljust(8, ' ') for x in self.dataset_names]).astype(str_type))
		attr_names.close()

		group_id.close()
		file_id.close()

		# create generic hdf5 file without group 'data' or attributes
		file_id = h5py.h5f.create(bytes(self.generic_file, encoding='utf-8'), h5py.h5f.ACC_TRUNC)
		space_id = h5py.h5s.create_simple(self.dataset_dims, self.dataset_dims)
		data_id = h5py.h5d.create(file_id, bytes('C', encoding='utf-8'), h5py.h5t.IEEE_F64LE, space_id)

		dataset = np.zeros(self.dataset_dims, np.double)
		data_id.write(h5py.h5s.ALL, h5py.h5s.ALL, dataset)  # all in memory -> all in file
		data_id.close()
		file_id.close()

	def tearDown(self):
		cmd = "rm " + self.generic_file + " " + self.sample_file
		os.system(cmd)

		# File & I/O tests

	def test_is_rfml_hdf5(self):
		self.assertFalse(is_rfml_hdf5(self.generic_file))
		self.assertTrue(is_rfml_hdf5(self.sample_file))
		self.assertListEqual(hdf5_read_variables(self.sample_file), list(h5py.File(self.sample_file)['data']))

		# Attribute tests

	def test_hdf5_read_attribute(self):
		self.assertListEqual(list(hdf5_read_attribute(self.sample_file, '/data', 'dimensions')), [10, 10, 10, 1])
		self.assertListEqual(list(hdf5_read_attribute(self.sample_file, '/data', 'datafield_names').astype('str')),
							 ['C'])

	def test_hdf5_add_attribute(self):
		hdf5_add_attribute(self.sample_file, '/data', 'time_variables', h5py.h5t.IEEE_F64LE, [1.234E-04, 2.356E-07])
		self.assertTrue(np.array_equal(hdf5_read_attribute(self.sample_file, '/data', 'time_variables'),
									   np.array([1.234E-04, 2.356E-07])))

	def test_hdf5_replace_attribute(self):
		hdf5_add_attribute(self.sample_file, '/data', 'time_variables', h5py.h5t.IEEE_F64LE, [1.234E-04, 2.356E-07])
		hdf5_replace_attribute(self.sample_file, '/data', 'time_variables', [2.468E-04, 4.712E-07])
		self.assertTrue(np.array_equal(hdf5_read_attribute(self.sample_file, '/data', 'time_variables'),
									   np.array([2.468E-04, 4.712E-07])))


if __name__ == '__main__':
	unittest.main()
