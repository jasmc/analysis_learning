from my_general_variables import *
import my_functions as f

from pathlib import Path
import pandas as pd
import numpy as np
import re

from scipy import interpolate

from PIL import Image, ImageSequence

import matplotlib.pyplot as plt


import plotly.io as pio
import plotly.graph_objects as go

pio.templates.default = "plotly_dark"




number_rows_read = None


pmt_off_beg = 'PMT_OFF beg'
pmt_off_end = 'PMT_OFF end'






protocol_path = r"C:\Users\joaqc\Desktop\test_1\test_1_stim control.txt"

data_path = r"C:\Users\joaqc\Desktop\test_1\test_1_cam.txt"

# galvo_path = r"C:\Users\joaqc\Desktop\test fish 05122023\behavior exp_1\test_1_exp_two photon sync reader.txt"
galvo_path = r"C:\Users\joaqc\Desktop\test_1\signalsfeedback.xls"

images_path = r"C:\Users\joaqc\Desktop\test_1"



protocol = f.read_protocol(protocol_path)
# pd.read_csv(protocol_path, engine='pyarrow', sep=' ', header=0, decimal='.')
					#    na_filter=False)

# protocol = protocol.iloc[1:]


data = pd.read_csv(data_path, engine='c', sep=' ', header=0, decimal='.', nrows=number_rows_read)

data[abs_time] = data[abs_time].astype('float64')


galvo = pd.read_csv(galvo_path, sep='\t', decimal='.', usecols=[0,1], names=[abs_time, 'GalvoValue'], dtype={'GalvoValue':'float64'}, skip_blank_lines=True, skipinitialspace=True, nrows=number_rows_read).dropna(axis=0)


galvo[abs_time] = pd.to_datetime(galvo[abs_time])

# Calculate unixtime in ms
galvo[abs_time] = galvo[abs_time].astype('int64') / 10**6
# galvo = (galvo[abs_time] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
# galvo = galvo[abs_time].map(pd.Timestamp.timestamp)


plt.plot(galvo[galvo_value].iloc[3500:3800])


galvo['1diff'] = galvo[galvo_value].diff()

galvo['2diff'] = galvo['1diff'].diff()

galvo['3'] = 0
galvo['3'].iloc[:-1] = galvo['2diff'][1:]


beg_frames = galvo.loc[(galvo['1diff'] > 0.1) & ((galvo['2diff'] < 0) | (galvo['3'] < 0))]


A = beg_frames[abs_time].diff()

mask = (A > 400) & (A < 550)


galvo['beg'] = np.nan

galvo.loc[beg_frames[mask].index, 'beg'] = 1

# len(beg_frames[mask].index) /2


A[(A > 400) & (A < 550)]
.max()

# number_images * 2





plt.plot(galvo[abs_time].iloc[6000:16420], galvo[galvo_value].iloc[6000:16420], 'k.')
# plt.plot(galvo[abs_time].iloc[6400:6420], galvo['1diff'].iloc[6400:6420], 'g.')
# plt.plot(galvo[abs_time].iloc[6400:6420], galvo['2diff'].iloc[6400:6420], 'r.')

plt.plot(galvo[abs_time].iloc[6400:16420], galvo['beg'].iloc[6400:16420], 'bo')

# data.dropna(subset=['GalvoValue', 'ID'])

space = 1000

# index=815563

for i, index in enumerate(galvo[galvo['beg'].notna()].index):

	fig = go.Figure()
	fig.add_trace(go.Scattergl(x=galvo.loc[index - space : index + space][abs_time], y=galvo.loc[index - space : index + space][galvo_value].to_numpy()))
	# fig.add_trace(go.Scattergl(x=galvo[::2].loc[index - space : index + space][abs_time], y=galvo.loc[index - space : index + space][galvo_value][::2].to_numpy()))
	fig.add_trace(go.Scattergl(x=galvo.loc[index - space : index + space][abs_time], y=galvo.loc[index - space : index + space]['1diff'].to_numpy()))
	fig.add_trace(go.Scattergl(x=galvo.loc[index - space : index + space][abs_time], y=galvo.loc[index - space : index + space]['2diff'].to_numpy()))
	fig.add_trace(go.Scattergl(x=galvo.loc[index - space : index + space][abs_time], y=galvo.loc[index - space : index + space]['3'].to_numpy()))

	# fig.add_trace(go.Scattergl(x=galvo.loc[index - space : index + space][abs_time][:-1], y=galvo.loc[index - space : index + space]['2diff'].to_numpy()[1:]))
	# fig.add_trace(go.Scattergl(x=galvo.loc[index - space : index + space][abs_time], y=galvo.loc[index - space : index + space]['2diff'].diff().to_numpy()))
	fig.add_trace(go.Scattergl(x=galvo.loc[index - space : index + space][abs_time], y=galvo.loc[index - space : index + space]['beg'].to_numpy()))

	fig.show()

	break

	# plt.plot(galvo.loc[index - space : index + space][abs_time], galvo.loc[index - space : index + space][galvo_value], 'k.')
	# plt.plot(galvo.loc[index - space : index + space][abs_time], galvo.loc[index - space : index + space]['1diff'], 'g.')
	# plt.plot(galvo.loc[index - space : index + space][abs_time], galvo.loc[index - space : index + space]['2diff'], 'r.')

	# plt.plot(galvo.loc[index - space : index + space][abs_time], galvo.loc[index - space : index + space]['beg'], 'bo')

	# plt.show()



# galvo_peaks = galvo['GalvoValue'].diff()

# # galvo_peaks[galvo_peaks>1.5]


# galvo_peaks = galvo_peaks.to_numpy()


# galvo_peaks = np.where(galvo_peaks>1.5)

# galvo_peaks = np.where(galvo_peaks>1.5, 5, 0)



# galvo.plot(abs_time, 'GalvoValue')

# galvo = galvo.drop(columns='GalvoValue')

# galvo = galvo.rename(columns={'AblationValue' : 'GalvoValue'})

# galvo = galvo.rename(columns={'FrameID' : 'ID'})

# galvo = galvo.drop(columns=['ElapsedTime', 'AbsoluteTime'])




		# np.median(np.diff(galvo_peaks))




# fig = go.Figure()
# fig.add_trace(go.Scattergl(x=np.arange(len(galvo)), y=galvo['GalvoValue'].to_numpy()))
# fig.add_trace(go.Scattergl(x=np.arange(len(galvo)), y=galvo_peaks))
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

# del protocol_, p


#!
DATA = data.copy()

# plt.plot(data[us_beg], 'ko')

data = pd.merge_ordered(data, galvo, on=abs_time, how='outer').drop_duplicates(abs_time, keep='first')

data.loc[data[pmt_off_beg].notna(), pmt_off_beg] = 1
data.loc[data[pmt_off_end].notna(), pmt_off_end] = 1

data.loc[data[us_beg].notna(), us_beg] = 1
data.loc[data[us_end].notna(), us_end] = 1




interp_function = interpolate.interp1d(galvo[abs_time], galvo[galvo_value], kind='slinear', axis=0, assume_sorted=True, bounds_error=False)

data[galvo_value] = interp_function(data[abs_time])



# del galvo


		# index_pmt_beg = np.zeros(len(data[data[pmt_off_beg].notna()]))
		# # len(index_pmt_beg)


		# for index in data.loc[data[pmt_off_beg].notna(), abs_time]:

		# 	data_sub = data.loc[data[abs_time].between(index - 1000, index + 3000)]
			
			
		# 	plt.plot(data_sub[abs_time], data_sub[pmt_off_beg] + 1, 'ko')
		# 	plt.plot(data_sub[abs_time], data_sub[pmt_off_end] + 1, 'go')
		# 	plt.plot(data_sub[abs_time], data_sub[us_beg] + 1, '.')
		# 	plt.plot(data_sub[abs_time], data_sub[us_end] + 1, '.')
		# 	plt.plot(data_sub[abs_time], data_sub[galvo_value])



		# 	print(data_sub.loc[data_sub[pmt_off_end].notna(), abs_time].to_numpy() - data_sub.loc[data_sub[pmt_off_beg].notna(), abs_time].to_numpy(), data_sub.loc[data_sub[us_beg].notna(), abs_time].to_numpy() - data_sub.loc[data_sub[pmt_off_beg].notna(), abs_time].to_numpy(), data_sub.loc[data_sub[pmt_off_end].notna(), abs_time].to_numpy() - data_sub.loc[data_sub[us_end].notna(), abs_time].to_numpy())

		# 	break






A = data[galvo_value].diff()



A[A>2]


data.iloc[5000:][galvo_value].plot()


len(images_mean)


plt.plot(A.iloc[5000:10000], 'k.')
plt.plot(data[galvo_value].iloc[5000:10000], 'g.')



plt.plot(images_mean)


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
# images_paths = [*Path(images_path).glob('*tiff')]

# number_images = len(images_paths)

# images_mean = [0 for _ in images_paths]

# for image_i, image in enumerate(images_paths):
	
# 	images_mean[image_i] = np.sum(np.array(Image.open(image)))
# 	# [240:260, 240:260])


# images_mean = np.array(images_mean, dtype='int')


images_paths = images_path + r"\test_1green.tif"

im = Image.open(images_paths)

images_mean = []

# ImageSequence.all_frames(im, np.mean)

try:
	for frame in ImageSequence.Iterator(im):
		
		images_mean.append(np.sum(frame))
except:
	pass

plt.plot(images_mean)



images_mean = np.array(images_mean, dtype='int')




len(images_mean)



data['image'] = 0




B = data.loc[data['beg'].notna(), 'image'].iloc[::2].index

data.loc[B, 'image'] = images_mean[1:-1]



len(images_mean[1:-1])














data.loc[:,[cs_beg, cs_end, us_beg, us_end, 'PMT_OFF beg', 'PMT_OFF end']] = data[[cs_beg, cs_end, us_beg, us_end, 'PMT_OFF beg', 'PMT_OFF end']].fillna(0)


plt.plot(data[abs_time], data['PMT_OFF beg'], 'ko')
plt.plot(data[abs_time], data[galvo_value])

# data = data.dropna(subset='ID')




data.dtypes






#* Fix dtypes.
data[cols_stim + ['PMT_OFF beg', 'PMT_OFF end']] = data[cols_stim + ['PMT_OFF beg', 'PMT_OFF end']].astype('Sparse[int16]')

data['ID'] = data['ID'].astype('int')









plt.plot(data['image'])
plt.plot(data['PMT_OFF end'])
plt.plot(images_mean*(-1/20000))

plt.plot(images_mean)
plt.plot(A)


A = np.diff(images_mean)
# *(-1/20000)

len(A[A<-50000])


plt.plot(data[abs_time], y=data[galvo_value])
plt.plot(data[abs_time], y=data[pmt_off_beg])
plt.plot(data[abs_time], y=data[pmt_off_end])


space = 10000

for index in data[data[us_beg]>0].index:
	
	data_sub = data.loc[index-space:index+space]

	fig = go.Figure()
	fig.add_trace(go.Scattergl(x=data_sub[abs_time], y=data_sub[galvo_value].to_numpy()))
	fig.add_trace(go.Scattergl(x=data_sub[abs_time], y=data_sub['image'].to_numpy()))
	fig.add_trace(go.Scattergl(x=data_sub[abs_time], y=data_sub[pmt_off_beg].to_numpy()))
	fig.add_trace(go.Scattergl(x=data_sub[abs_time], y=data_sub[pmt_off_end].to_numpy()))


	break

fig.write_html(r"C:\Users\joaqc\Desktop\test.html")






#region

need to find the beginning of each imaging frame









# B = data['GalvoValue'] / data[abs_time].diff()
# A = data['GalvoValue'] / data[ela_time].diff()

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


#endregion











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
