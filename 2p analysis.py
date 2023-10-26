
from my_general_variables import *

import my_functions as f

import pandas as pd
from pathlib import Path
# import scipy as 



protocol_path = r"C:\Users\joaqc\Desktop\test_fish_2p\testWithImagingstim control.txt"

camera_path = r"C:\Users\joaqc\Desktop\test_fish_2p\testWithImagingcam.txt"

galvo_path = r"C:\Users\joaqc\Desktop\test_fish_2p\test.txt"



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



# camera.plot(x='ID', y=abs_time)



# camera = camera.set_index(abs_time)

camera.loc[:,['ID', ela_time]] = camera[['ID', ela_time]].interpolate(kind='slinear')

# camera = camera.reset_index(drop=True).dropna().drop(columns=[ela_time, abs_time])

camera = camera.drop(columns=[ela_time, abs_time])

# camera['ID'] = camera['ID'].astype('int64')


#* Fix dtypes.
camera[cols_stim + ['PMT_OFF beg', 'PMT_OFF end']] = camera[cols_stim + ['PMT_OFF beg', 'PMT_OFF end']].astype('Sparse[int16]')

# for stim in cols_stim + ['PMT_OFF beg', 'PMT_OFF end']:
	
# 	camera[stim] = camera.loc[:, stim].astype(pd.api.types.CategoricalDtype(categories=camera[stim].unique(), ordered=True))




#! Open the images and take the mean














#! PLOTS



camera['ID'] -= camera['ID'].iat[0]




















#! Shape of the galvoValue

camera.plot(x='ID', y='GalvoValue', )


camera.plot(x='ID', y=[us_beg, us_end, 'PMT_OFF beg', 'PMT_OFF end'], )

camera.plot(x='ID', y=[us_beg, us_end, cs_beg, cs_end], )


camera.plot(x='ID', y=[ 'PMT_OFF beg', 'PMT_OFF end'], xlim=(31500, 33000))

# camera[abs_time].diff().plot()

camera.plot(x='ID', y=[us_beg, us_end],)
			xlim=(31500, 33000))



protocol[end] - protocol[beg]











camera.dtypes



camera.plot()