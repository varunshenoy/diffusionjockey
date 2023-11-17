# Run this to generate a processed datafile.
generate_data:
	python dataprep.py

# Run this to turn processed datafile into HF dataset.
prepare:
	python dataset.py

# Start training procedure
train:
	python main.py
