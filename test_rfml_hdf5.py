import sys
import unittest
import numpy.testing as npt
from rfml_hdf5 import *


class TestHdf5Functions(unittest.TestCase):
	sample_file = 'sample_rfml_hdf5.h5'
	generic_file = 'generic_hdf5.h5'
	dataset_names = ['C']
	dataset_dims = (10, 10, 10)

	def setUp(self):
		# sample hdf5 file with the RFML structure
		data = {'C': np.zeros(self.dataset_dims)}
		hdf5_NGA_data_write(self.sample_file, data, 0, 2E-07)

		# Manually create generic hdf5 file without group 'data' or required attributes
		file_id = h5py.File(self.generic_file, 'w')
		file_id.create_dataset('C', data=data.get('C'))
		file_id.close()

	def tearDown(self):
		cmd = "rm " + self.generic_file + " " + self.sample_file
		os.system(cmd)

	# === FILE I/O TESTS === #
	def test_is_rfml_hdf5(self):
		self.assertFalse(is_rfml_hdf5(self.generic_file))
		self.assertTrue(is_rfml_hdf5(self.sample_file))
		self.assertListEqual(hdf5_read_variables(self.sample_file), list(h5py.File(self.sample_file)['data']))

	# === ATTRIBUTE TESTS === #
	def test_hdf5_read_attribute(self):
		self.assertListEqual(list(hdf5_read_attribute(self.sample_file, '/data', 'dimensions')), [10, 10, 10, 1])
		self.assertListEqual(list(hdf5_read_attribute(self.sample_file, '/data', 'datafield_names').astype('str')),
							 ['C'])

	def test_hdf5_add_attribute(self):
		hdf5_add_attribute(self.sample_file, '/data', 'temp_attr', h5py.h5t.IEEE_F64LE, [1.234E-04, 2.356E-07])
		self.assertTrue(np.array_equal(hdf5_read_attribute(self.sample_file, '/data', 'temp_attr'),
									   np.array([1.234E-04, 2.356E-07])))

	def test_hdf5_replace_attribute(self):
		hdf5_replace_attribute(self.sample_file, '/data', 'time_variables', [2.468E-04, 4.712E-07])
		self.assertTrue(np.array_equal(hdf5_read_attribute(self.sample_file, '/data', 'time_variables'),
									   np.array([2.468E-04, 4.712E-07])))

	def test_hdf5_add_dataset(self):
		random_dataset = np.random.random_sample(self.dataset_dims)
		hdf5_add_dataset(self.sample_file, '/data', 'T', h5py.h5t.IEEE_F64LE, random_dataset)

		# compare written array
		fid = h5py.File(self.sample_file, 'r')
		read_dataset = np.array(fid.get('/data')['T']).astype(h5py.h5t.IEEE_F64LE)
		npt.assert_array_almost_equal(random_dataset, read_dataset, 1e-15)

		# check attributes 'datafield_names' and 'dimensions'
		self.assertTrue('T' in list(fid['data'].attrs['datafield_names'].astype('str')))
		self.assertTrue('T' in list(fid['data']))
		self.assertEqual((fid['data'].attrs['dimensions'])[3], 2)
		fid.close()

	# === DATASET TESTS === #
	def test_hdf5_remove_dataset(self):
		hdf5_remove_dataset(self.sample_file, '/data', 'C')
		fid = h5py.File(self.sample_file, 'r')
		self.assertTrue('C' not in list(fid['data'].attrs['datafield_names']))
		self.assertTrue('C' not in list(fid['data']))
		self.assertEqual((fid['data'].attrs['dimensions'])[3], 0)
		fid.close()

	def test_hdf5_overwrite_dataset(self):
		new_data = np.random.random_sample(self.dataset_dims)
		new_data = new_data.astype(h5py.h5t.IEEE_F32BE)
		hdf5_overwrite_dataset(self.sample_file, '/data', 'C', new_data)

		# Check that overwritten data is read exactly, even though memory and file types differ
		npt.assert_array_almost_equal(hdf5_read_dataset(self.sample_file, '/data', 'C'), new_data, 16)

		# Check that overwriting has not changed the dataset type from F64LE to F32LE
		new_type = hdf5_get_dataset_type(self.sample_file, 'data', 'C')
		self.assertEqual(new_type, h5py.h5t.IEEE_F64LE)

	# === HDF5 NGA WRAPPER TESTS === #
	def test_hdf5_NGA_config_write(self):
		# write temporary config file
		x = np.linspace(-1.0, 1.0, 6)
		y = np.linspace(-2.0, 2.0, 11)
		z = np.linspace(-4.0, 4.0, 21)
		mask = np.zeros((5, 10)).astype(h5py.h5t.STD_I32LE)
		hdf5_NGA_config_write('temp_config.h5', 'isotropic_turbulence', x, y, z, 0, [1, 1, 1], mask)

		# Test attributes
		npt.assert_array_equal(hdf5_read_attribute('temp_config.h5', '/data', 'simulation_name'),
							   np.array('isotropic_turbulence').astype('|S64'))
		npt.assert_array_equal(hdf5_read_attribute('temp_config.h5', '/data', 'parameters'),
							   np.array([0, 1, 1, 1, x.size - 1, y.size - 1, z.size - 1]))

		# Test datasets
		npt.assert_array_almost_equal(hdf5_read_dataset('temp_config.h5', '/data', 'x'), x, 15)
		npt.assert_array_almost_equal(hdf5_read_dataset('temp_config.h5', '/data', 'y'), y, 15)
		npt.assert_array_almost_equal(hdf5_read_dataset('temp_config.h5', '/data', 'z'), z, 15)
		npt.assert_array_equal(hdf5_read_dataset('temp_config.h5', '/data', 'mask'), mask)

		os.system('rm temp_config.h5')

	def test_hdf5_NGA_data_write(self):
		test_data_file = 'test_NGA_data.h5'
		data = {'U': np.random.random_sample(self.dataset_dims),
				'V': np.random.random_sample(self.dataset_dims),
				'W': np.random.random_sample(self.dataset_dims)}
		hdf5_NGA_data_write(test_data_file, data, 1.0E-04, 2E-07, 'little-endian')

		self.assertTrue(is_rfml_hdf5(test_data_file))

		# Attribute tests
		dimensions = list(self.dataset_dims)
		dimensions.append(3)
		npt.assert_array_almost_equal(hdf5_read_attribute(test_data_file, '/data', 'time_variables'),
										np.array([2E-07, 1.0E-04]).astype(h5py.h5t.IEEE_F64LE), 16)
		self.assertListEqual(list(hdf5_read_attribute(test_data_file, '/data', 'dimensions')), dimensions)
		self.assertListEqual(list(hdf5_read_attribute(test_data_file, '/data', 'datafield_names').astype('str')),
							 list(data.keys()))

		# Dataset tests
		fid = h5py.File(test_data_file, 'r')
		self.assertListEqual(list(fid['data']), ['U', 'V', 'W'])
		npt.assert_array_almost_equal(hdf5_read_dataset(test_data_file, '/data', 'U'), data.get('U'), 16)
		npt.assert_array_almost_equal(hdf5_read_dataset(test_data_file, '/data', 'V'), data.get('V'), 16)
		npt.assert_array_almost_equal(hdf5_read_dataset(test_data_file, '/data', 'W'), data.get('W'), 16)

		fid.close()


if __name__ == '__main__':
	unittest.main()
