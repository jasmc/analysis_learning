from my_general_variables import *
import my_functions as f

from pathlib import Path
import pandas as pd
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt

protocol_path = r"D:\data\joaquim\test_1105\test_stim control.txt"

camera_path = r"D:\data\joaquim\test_1105\test_two photon sync reader.txt"

# galvo_path = r"E:\data\test\zscan\test1\monacoshutterSat, Oct 28, 2023 6-04-58 PM.dat"
# # r"E:\data\test\zscan\conversion.txt"
# # r"D:\data\joaquim\test_fish\conversion.txt"

images_path = r"D:\data\joaquim\test_1105"



protocol = f.read_protocol(protocol_path)

protocol = protocol.iloc[1:]


camera = pd.read_csv(camera_path, engine='pyarrow', sep=' ', header=0, decimal='.', na_filter=False)

galvo = pd.read_csv(galvo_path, engine='pyarrow', sep='\t', header=4, decimal='.', na_filter=False)

galvo.rename(columns={'time' : abs_time, 'Dev1/ai0' : galvo_value}, inplace=True)

galvo = galvo.iloc[:,[0,1]]

galvo[abs_time] = galvo[abs_time].astype('datetime64[ns]') - pd.Timedelta('1h')

# Calculate unixtime in ms
galvo[abs_time] = galvo[abs_time].astype('int64') / 10**6
# galvo = (galvo[abs_time] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
# galvo = galvo[abs_time].map(pd.Timestamp.timestamp)


# galvo.plot(y='GalvoValue')



camera['AblationValue'].iloc[20000:22000].plot()

# camera = f.highlight_stim_in_data(camera, protocol)


protocol_ = protocol.copy()

protocol_[[beg, end]] = protocol_[[beg, end]].astype('float')

for cs_us in [cs, us, 'PMT_OFF']:

	for beg_end in [beg, end]:

		beg_end_name = ' beg' if beg_end == beg else ' end'

		p = pd.DataFrame(protocol_.loc[protocol_[stim_type]==cs_us, beg_end]).rename(columns={beg_end : abs_time})

		p[cs_us + beg_end_name] = np.arange(1, 1+len(p))

		camera = pd.merge_ordered(camera, p, on=abs_time, how='outer').drop_duplicates(abs_time, keep='first')

del protocol_, p

camera = pd.merge_ordered(camera, galvo, on=abs_time, how='outer').drop_duplicates(abs_time, keep='first')

camera.loc[:,[cs_beg, cs_end, us_beg, us_end, 'PMT_OFF beg', 'PMT_OFF end', 'GalvoValue']] = camera[[cs_beg, cs_end, us_beg, us_end, 'PMT_OFF beg', 'PMT_OFF end', 'GalvoValue']].fillna(0)

del galvo



# camera.plot(x='FrameID', y=abs_time)



# camera = camera.set_index(abs_time)

camera.loc[:,['FrameID', ela_time]] = camera[['FrameID', ela_time]].interpolate(kind='slinear')

# camera = camera.reset_index(drop=True).dropna().drop(columns=[ela_time, abs_time])

camera = camera.drop(columns=[ela_time, abs_time])

# camera['FrameID'] = camera['FrameID'].astype('int64')


#* Fix dtypes.
camera[cols_stim + ['PMT_OFF beg', 'PMT_OFF end']] = camera[cols_stim + ['PMT_OFF beg', 'PMT_OFF end']].astype('Sparse[int16]')

# for stim in cols_stim + ['PMT_OFF beg', 'PMT_OFF end']:
	
# 	camera[stim] = camera.loc[:, stim].astype(pd.api.types.CategoricalDtype(categories=camera[stim].unique(), ordered=True))





#* Pad the image paths
# List all files in the folder
# files = os.listdir(folder_path)
images_paths = [*Path(images_path).glob('*tiff')]

# Regex pattern to find all integer numbers in the file names
pattern = re.compile(r'(\d+)')

# Iterate through each file and rename it
for images_name in images_paths:

	new_image_name = re.sub(pattern, lambda x: x.group(1).zfill(10), str(images_name.stem))
	
	images_name.rename(Path(images_path).joinpath(new_image_name + '.tiff'))





#* Open the images and take the mean
images_paths = [*Path(images_path).glob('*tiff')]

images_mean = [0 for _ in images_paths]

for image_i, image in enumerate(images_paths):
	
	images_mean[image_i] = np.mean(np.array(Image.open(image)))





plt.plot(images_mean)




#! PLOTS



camera['FrameID'] -= camera['FrameID'].iat[0]




with open(galvo_path, 'rb') as file:
    binary_string = file.read()

# print(binary_string)

# with open(r'E:\\data\\test\\zscan\\conversionNEW.txt', 'w') as new_file:
# 	new_file.write(binary_string)
	

text_string = binary_string.decode('')
print(text_string)






camera = camera.fillna(0)


camera[camera['US beg'].notna()]

#! Shape of the galvoValue

camera.plot(x='FrameID', y='GalvoValue', )


camera.plot(x='FrameID', y=[us_beg, us_end, 'PMT_OFF beg', 'PMT_OFF end'], )

camera.plot(y=[us_beg, us_end, cs_beg, cs_end], )


# camera.plot(x='FrameID', y=[ 'PMT_OFF beg', 'PMT_OFF end'], xlim=(31500, 33000))

# camera[abs_time].diff().plot()

camera.plot(x='FrameID', y=[us_beg, us_end],)
			xlim=(31500, 33000))



protocol[end] - protocol[beg]











camera.dtypes



camera.plot()