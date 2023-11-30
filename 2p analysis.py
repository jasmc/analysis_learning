from my_general_variables import *
import my_functions as f

from pathlib import Path
import pandas as pd
import numpy as np
import re

from PIL import Image
import matplotlib.pyplot as plt


import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "plotly_dark"












protocol_path = r"C:\Users\joaqc\Desktop\test_1105\test_stim control.txt"

protocol_path = r"C:\Users\joaqc\Desktop\test_1105\test_stim control.txt"

data_path = r"C:\Users\joaqc\Desktop\test_1105\test_cam.txt"

galvo_path = r"C:\Users\joaqc\Desktop\test_1105\test_two photon sync reader.txt"

images_path = r"C:\Users\joaqc\Desktop\test_1105"



protocol = f.read_protocol(protocol_path)
# pd.read_csv(protocol_path, engine='pyarrow', sep=' ', header=0, decimal='.')
					#    na_filter=False)


# protocol = protocol.iloc[1:]


data = pd.read_csv(data_path, engine='pyarrow', sep=' ', header=0, decimal='.')


galvo = pd.read_csv(galvo_path, engine='pyarrow', sep=' ', header=0, decimal='.')

galvo = galvo.drop(columns='GalvoValue')

galvo = galvo.rename(columns={'AblationValue' : 'GalvoValue'})

galvo = galvo.rename(columns={'FrameID' : 'ID'})

galvo = galvo.drop(columns=['ElapsedTime', 'AbsoluteTime'])

		# galvo_peaks = galvo['GalvoValue'].diff()

		# galvo_peaks[galvo_peaks>1.5]


		# galvo_peaks = galvo_peaks.to_numpy()


		# galvo_peaks = np.where(galvo_peaks>1.5)

		# # galvo_peaks = np.where(galvo_peaks>1.5, 5, 0)




		# np.median(np.diff(galvo_peaks))




# fig = go.Figure()
# fig.add_trace(go.Scattergl(x=np.arange(len(galvo)), y=galvo['GalvoValue'].to_numpy()))
# # fig.add_trace(go.Scattergl(x=np.arange(len(galvo)), y=galvo_peaks))
# fig.add_trace(go.Scattergl(x=np.arange(len(galvo)), y=galvo['GalvoValue'].to_numpy()))


# fig.write_html(r"C:\Users\joaqc\Desktop\test.html")






# galvo['GalvoValue'].iloc[50000:52000].plot()
# galvo_peaks.iloc[50000:52000].plot()

# galvo['GalvoValue'][galvo_peaks>2]



	#! this is for when the galvo signal is saved throgh LabView
	# galvo = pd.read_csv(galvo_path, engine='pyarrow', sep='\t', header=4, decimal='.', na_filter=False)

	# galvo.rename(columns={'time' : abs_time, 'Dev1/ai0' : galvo_value}, inplace=True)

	# galvo = galvo.iloc[:,[0,1]]

	# galvo[abs_time] = galvo[abs_time].astype('datetime64[ns]') - pd.Timedelta('1h')

	# # Calculate unixtime in ms
	# galvo[abs_time] = galvo[abs_time].astype('int64') / 10**6
	# # galvo = (galvo[abs_time] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
	# # galvo = galvo[abs_time].map(pd.Timestamp.timestamp)



# data = f.highlight_stim_in_data(data, protocol)


protocol_ = protocol.copy()

protocol_[[beg, end]] = protocol_[[beg, end]].astype('float')

for cs_us in [cs, us, 'PMT_OFF']:

	for beg_end in [beg, end]:

		beg_end_name = ' beg' if beg_end == beg else ' end'

		p = pd.DataFrame(protocol_.loc[protocol_['Type']==cs_us, beg_end]).rename(columns={beg_end : abs_time})

		p[cs_us + beg_end_name] = np.arange(1, 1+len(p))

		data = pd.merge_ordered(data, p, on=abs_time, how='outer').drop_duplicates(abs_time, keep='first')

del protocol_, p

data = pd.merge_ordered(data, galvo, on='ID', how='outer').drop_duplicates(abs_time, keep='first')



del galvo

data.loc[:,[cs_beg, cs_end, us_beg, us_end, 'PMT_OFF beg', 'PMT_OFF end']] = data[[cs_beg, cs_end, us_beg, us_end, 'PMT_OFF beg', 'PMT_OFF end']].fillna(0)

#* Fix dtypes.
# data[cols_stim[:-2] + ['PMT_OFF beg', 'PMT_OFF end']] = data[cols_stim[:-2] + ['PMT_OFF beg', 'PMT_OFF end']].astype('Sparse[int16]')

# A = data['GalvoValue'] / data[ela_time].diff()
# B = data['GalvoValue'] / data[abs_time].diff()

# fig = go.Figure()
# fig.add_trace(go.Scattergl(x=np.arange(len(A)), y=A.to_numpy()))
# fig.add_trace(go.Scattergl(x=np.arange(len(B)), y=B.to_numpy()))

# A = A.dropna()

# A.median()

data = data.reset_index(drop=True)

galvo_peaks = data['GalvoValue'].diff()

# galvo_peaks[galvo_peaks>1.5]

galvo_peaks = galvo_peaks.to_numpy()

# galvo_peaks.where(galvo_peaks > 1.5, False)

galvo_peaks_index = np.where(galvo_peaks>1.5)[0]

#* Interval between frames (only works for a lot of frames).
interval_between_frames = np.median(np.diff(galvo_peaks_index))

mask = np.diff(galvo_peaks_index) > 600
# (np.diff(galvo_peaks_index) > 340) & (np.diff(galvo_peaks_index) < 360)

C = data.loc[galvo_peaks_index[1:][mask][0]:]
# data.loc[data.index >= galvo_peaks_index[1:][mask][0]]



fig = go.Figure()
fig.add_trace(go.Scattergl(x=data.index, y=galvo_peaks))
fig.add_trace(go.Scattergl(x=data.index, y=data['GalvoValue'].to_numpy()))
fig.add_trace(go.Scattergl(x=C.index, y=C['GalvoValue'].to_numpy()))



# fig.add_trace(go.Scattergl(x=np.arange(len(data)), y=galvo_peaks))
fig.add_trace(go.Scattergl(x=np.arange(len(data)), y=data['GalvoValue'].to_numpy()))


fig.write_html(r"C:\Users\joaqc\Desktop\test.html")



#! Improve this part
galvo_peaks = C['GalvoValue'].diff()
# galvo_peaks[galvo_peaks>1.5]
galvo_peaks = galvo_peaks.to_numpy()
# galvo_peaks.where(galvo_peaks > 1.5, False)
galvo_peaks_index = np.where(galvo_peaks>1.5)[0]

#? Fix the time of all peaks?

D = C.iloc[galvo_peaks_index]

E = np.zeros(len(D))
E[:len(images_mean)] = images_mean

D['Mean of images'] = E


data['Mean of images'] = 0

data.loc[D.index] = D


x = data[abs_time].to_numpy()

fig = go.Figure()
# fig.add_trace(go.Scattergl(x=x, y=D['GalvoValue'].to_numpy()))
fig.add_trace(go.Scattergl(x=x, y=data['Mean of images'].to_numpy()))

fig.add_trace(go.Scattergl(x=x, y=data['PMT_OFF beg'].to_numpy()*200))





# data[abs_time] -= data[abs_time].iloc[0]

# fig = go.Figure()
# fig.add_trace(go.Scattergl(x=data[abs_time].to_numpy(), y=data['GalvoValue'].to_numpy()))
# # fig.add_trace(go.Scattergl(x=np.arange(len(data)), y=galvo_peaks))
# fig.add_trace(go.Scattergl(x=np.arange(len(data)), y=data['GalvoValue'].to_numpy()))









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

number_images = len(images_paths)

images_mean = [0 for _ in images_paths]

for image_i, image in enumerate(images_paths):
	
	images_mean[image_i] = np.mean(np.array(Image.open(image)))






fig = go.Figure()
fig.add_trace(go.Scatter(x=data[abs_time], y=data['GalvoValue'].to_numpy()))
fig.add_trace(go.Scatter(x=data[abs_time], y=galvo_peaks))












galvo.plot(y='GalvoValue')

galvo.iloc[15000:20000].plot(y='GalvoValue')





data = data.iloc[20000:80000]



data['GalvoValue'].iloc[20000:22000].plot()

data.plot(x='ID', y=abs_time)










plt.plot(images_mean)




#! PLOTS



data['ID'] -= data['ID'].iat[0]




# with open(galvo_path, 'rb') as file:
#     binary_string = file.read()

# # print(binary_string)

# # with open(r'E:\\data\\test\\zscan\\conversionNEW.txt', 'w') as new_file:
# # 	new_file.write(binary_string)
	

# text_string = binary_string.decode('')
# print(text_string)






#! Shape of the galvoValue
data.plot('ID', 'GalvoValue')

data.plot(x='ID', y=['GalvoValue', us_beg, us_end] )


plt.plot(data[us_beg], 'green')
plt.plot(data[us_beg], 'red')


data = data[data['ID'].between(8000,45000)]
data.plot(x='ID', y=[us_beg, 'PMT_OFF beg'], )


data.plot(x='ID', y=[us_beg, us_end, cs_beg, cs_end], )


data.plot(x='ID', y=[ 'PMT_OFF beg', 'PMT_OFF end'])

# data[abs_time].diff().plot()

data.plot(x='ID', y=[us_beg, us_end],)



protocol[end] - protocol[beg]











data.dtypes



data.plot()






import struct

galvo_path = Path(r"C:\Users\joaqc\Desktop\monacoshutterSat, Oct 28, 2023 6-04-58 PM.dat")

# galvo_bits = open(galvo_path, 'rb')
# galvo_bits = galvo_bits.read()
# struct.unpack('>i', galvo_bits)





# Reading and decoding data from the file
with open(str(galvo_path), 'rb') as f:
    binary_data = f.read()
    decoded_data = binary_data.decode('ascii', 'ignore')
    print(decoded_data)
