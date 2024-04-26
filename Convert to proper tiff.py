import os
from pathlib import Path

import numpy as np
import tifffile

#!!!!!!!!!!!! NEEDS TO BE TESTED WHEN THE IMAGES ARE NOT SQUARE




#!!!!!!!!!!!!!!!! bug. duplicates frames



data_folder = r"F:\Pilot studies\2023_11_2-P"
fish_name = r"20240220_01_delay_2p-1_mitfaMinusMinus,elavl3H2BCaMP6s_7dpf"

data_folder = Path(data_folder) / fish_name

the_plane_path = data_folder / 'imaging' / (fish_name + '_green.tif')
image_path = data_folder / 'imaging' / (fish_name + '.tif')

image_number = 0  # count number of frames
imagesperfile = 1000

with open(the_plane_path, 'rb') as LabView_tif:
    byte_order = np.fromfile(LabView_tif, dtype=np.uint16, count=1)[0]
    # print('byteorder. this is not accurate?')
    # print(byteorder.byteswap())

    arbitrary = np.fromfile(LabView_tif, dtype=np.uint16, count=1)[0]
    # print('arbitrary should be 42')
    # print(arbitrary.byteswap())

    IGD1off = np.fromfile(LabView_tif, dtype=np.uint32, count=1)[0]
    # print('1st offset')
    # print(IGD1off.byteswap())

    number_fields = np.fromfile(LabView_tif, dtype=np.uint16, count=1)[0].byteswap()
    # print('num fields')
    # print(numfields)

    tag = np.fromfile(LabView_tif, dtype=np.uint16, count=1)[0]
    # print('first tag image height should be 256')
    # print(tag.byteswap())

    field_data = np.fromfile(LabView_tif, dtype=np.uint8, count=6)

    # do not swap?
    w = np.fromfile(LabView_tif, dtype=np.uint32, count=1)[0].byteswap()
    # print('Height')
    # print(h)

    tag = np.fromfile(LabView_tif, dtype=np.uint16, count=1)[0]
    # print('second tag image wifth should be 257')
    # print(tag.byteswap())

    field_data = np.fromfile(LabView_tif, dtype=np.uint8, count=6)

    h = np.fromfile(LabView_tif, dtype=np.uint32, count=1)[0].byteswap()
    # print('Width')
    # print(w)

    for n in range(number_fields - 2):
        np.fromfile(LabView_tif, dtype=np.uint8, count=12)

    next_offset = np.fromfile(LabView_tif, dtype=np.uint32, count=1)[0].byteswap()
    # print('next offset')
    # print(nextOffset)

    # skip resolution data
    np.fromfile(LabView_tif, dtype=np.uint8, count=16)
    # resdata=np.fromfile(fid, dtype=np.uint8, count=16)
    # print(resdata)
    # fread(fid,16);


    number_pixels = h * w

#???????????????????
    expected_header_length=number_fields*12+2+4+16


    image = np.fromfile(LabView_tif, dtype=np.uint16, count=number_pixels).reshape(h, w)
    # .T
    # print('image: ', nextImage)
    # print('image shape: ', nextImage.shape)
    # print('image dtype: ', nextImage.dtype)
    # B = uint16(reshape(fread(fid,numpixels,'uint16'),h,w)');
    
    
    image_path = os.path.join(data_folder/ 'imaging' , f'splitfiles{image_number // imagesperfile:04d}.tif')
    

    tif = tifffile.TiffWriter(image_path, mode='w')
    tif.save(image)

    # imwrite(image_path, B, append=True)
    # cv2.imwrite(image_path, B, [cv2.IMWRITE_TIFF_COMPRESSION, 0])
    # imwrite(B,fullfile(data_folder,strcat('splitfiles0000.tif')),'TIF','Compression','none','WriteMode','overwrite')  

    # Save the image in TIFF format
    # img.save('output.tif')


    skip_header = np.fromfile(LabView_tif, dtype=np.uint8, count=expected_header_length)
    # print(skipheader)

    while True:
        
        if len(skip_header) == expected_header_length:

            next_data = np.fromfile(LabView_tif, dtype=np.uint16, count=number_pixels)
            
            if len(next_data) == number_pixels:
                
                image_number += 1
                image = next_data.reshape((h, w)).byteswap()
                # .T
                
                tif = tifffile.TiffWriter(image_path, mode='r+')
                tif.save(image)

                if image_number % imagesperfile == 0:
                    # mode = 'w'
                    image_path = os.path.join(data_folder/ 'imaging' , f'splitfiles{image_number // imagesperfile:04d}.tif')
                    tif = tifffile.TiffWriter(image_path, mode='w')
                    tif.save(image)
                    tif.close()
                    tif = tifffile.TiffWriter(image_path, mode='r+')

                else:
                    # mode = None
                    # tif = tifffile.TiffWriter(image_path, mode='r+')
                    tif.save(image)

                print(f'Wrote image {image_number} to {image_path}') if image_number % 1000 == 0 else None

                # with open(image_path, 'ab') as split_file:
                #     imwrite(split_file, nextImage)
                #     # nextImage.tofile(split_file)

            skip_header = np.fromfile(LabView_tif, dtype=np.uint8, count=expected_header_length)

        else:
            break
            
# count = 0

# while True:
#     if len(np.fromfile(LabView_tif, dtype=np.uint16, count=number_pixels)) == number_pixels

#     if len(np.fromfile(LabView_tif, dtype=np.uint8, count=expected_header_length)) == expected_header_length

#     count += 1

# tif.close()

