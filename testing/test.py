for i in range(1, 5):
	file_name = f'test-{i}.jpeg'
	output_name = f'output-{i}.jpeg'
	m = Masker(file_name, output_name)
	m.mask_aadhar()

