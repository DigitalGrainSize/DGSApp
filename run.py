# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2022, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import streamlit as st
from PIL import Image
import os, io, glob
import zipfile
import pandas as pd
from dgs import *

import matplotlib.pyplot as plt
import numpy as np
from stqdm import stqdm

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=RuntimeWarning)

#============================================================
# =========================================================
def do_compute(images_list): 
    maxscale=5
    verbose=True 
    x=0
    resolution=1

    ALL_RES = []
    for k in stqdm(range(len(images_list))):
        st.session_state.img_idx=k

        infile = images_list[k]
        print(infile)
        # infile=io.BytesIO(infile.getvalue())
        data_out = dgs(infile, 1,verbose,  maxscale, x)
        ALL_RES.append(data_out)

        input_img  = np.array(Image.open(infile), dtype=np.uint8)

        # plt.clf()
        plt.imshow(input_img,cmap='gray')
        plt.plot([50, 50+data_out['mean grain size']], [50, 50], 'r')
        try:
            plt.xlim(0,200)
            plt.ylim(0,200)
        except:
            pass
        plt.axis("off")
        plt.savefig(infile.name.replace('.','_dgs_out.')+'_detail.png', dpi=300, bbox_inches='tight')
        plt.close()

    outfile = images_list[0].name+'_'+images_list[-1].name+'_dgs_out'

    ## parse out dict into three separate dictionaries
    S = {}; P = {}; F = {}
    counter = 0
    for data_out in ALL_RES:
        stats = dict(list(data_out.items())[:7])
        percentiles = dict(list(data_out.items())[7:9])
        freqs_bins = dict(list(data_out.items())[9:])

        if resolution!=1:
            freqs_bins['grain size bins']*=resolution
            percentiles['percentile_values']*=resolution

            for k in stats.keys():
                stats[k] = stats[k]*resolution

        S[images_list[counter].name] = stats.items()
        P[images_list[counter].name] = percentiles
        F[images_list[counter].name] = freqs_bins
        counter += 1

    # convert into stats (rows) versus images (columns)
    tmp = list(S.keys())
    d = {tmp[0]: [k[1] for k in list(S[tmp[0]])]}
    for k in range(1,len(tmp)):
        d.update( {tmp[k]: [k[1] for k in list(S[tmp[k]])]} )

    dat = pd.DataFrame(data=d, index = ['param_x', 'param_maxscale', 'median grain size', 'mean grain size', 'grain size sorting', 'grain size skewness', 'grain size kurtosis'])
    dat.to_csv(outfile+'_stats.csv')
    
    # convert into percentiles (rows) versus images (columns)
    tmp = list(P.keys())
    dd = {tmp[0]: P[tmp[0]]['percentile_values']}
    for k in range(1,len(tmp)):
        dd.update( {tmp[k]: P[tmp[k]]['percentile_values'] } )

    prcs=pd.DataFrame(data=dd, index = P[tmp[0]]['percentiles'])
    prcs.to_csv(outfile+'_percentiles.csv')

    # write each to csv file
    pd.DataFrame.from_dict(F).to_csv(outfile+'_freqs_bins.csv')

    counter = 0
    cols = ['r','g','b','m','c','k','y'][:len(F)]
    for f in F:
        try:
            plt.plot(F[f]['grain size bins'], F[f]['grain size frequencies'],cols[counter], lw=2, label=images_list[counter].name.split(os.sep)[-1])
        except:
            pass
        counter += 1
    plt.legend(fontsize=6)

    plt.xlabel('Grain Size (pixels)')

    plt.ylabel('Frequency')
    #plt.show()
    plt.savefig(outfile+'_psd.png', dpi=300, bbox_inches='tight')
    plt.close('all')

# =========================================================
def rm_thumbnails():
    try:
        for k in glob.glob('*dgs_out*'):
            os.remove(k)
    except:
        pass

def create_zip():
    with zipfile.ZipFile('dgs_results.zip', mode="w") as archive:
        for k in glob.glob("*dgs_out*"):
            archive.write(k)
    
    with open('dgs_results.zip','rb') as f:
        g=io.BytesIO(f.read()) 
    os.remove('dgs_results.zip')
    rm_thumbnails()
    return g

def compute_button():
    do_compute(images_list)
    st.balloons()

# =========================================================
# ================draw page ==============

st.set_page_config(
     page_title="DGS online",
     page_icon="🖖",
     layout="centered",
     initial_sidebar_state="collapsed",
     menu_items={
         'Get Help': None,
         'Report a bug': None,
         'About': "DGS your images!"
     }
 )

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

uploaded_files = st.file_uploader("Upload files. Works with jpg, tif, and png format files only", accept_multiple_files=True)
for uploaded_file in uploaded_files:
     bytes_data = uploaded_file.read()
images_list=uploaded_files

# Initialize app states
if 'img_idx' not in st.session_state:
    st.session_state.img_idx=0

if images_list==[]:
    image= Image.open("./assets/logo_cropped_trans.png")
else:
    if st.session_state.img_idx>=len(images_list):
        image = Image.open("./assets/logo_cropped_trans.png")
    else:
        image = Image.open(images_list[st.session_state.img_idx])

st.image("./assets/logo_cropped_trans.png")
# st.title("Digital Grain Size - Online")
st.markdown("by [Daniel Buscombe](https://github.com/dbuscombe-usgs), [Marda Science](https://www.mardascience.com/).\
See [github page](https://github.com/dbuscombe-usgs/pyDGS) for code and docs. \
Uses a modification of the algorithm detailed [in this paper](https://github.com/dbuscombe-usgs/pyDGS/blob/master/ref/Buscombe_2013_sedimentology_10.1111-sed.12049.pdf). \
[Citations](https://scholar.google.com/scholar?cluster=2040301200121912861&hl=en&as_sdt=0,5) appreciated. \
This implementation attempts to estimate appropriate hyperparameters ('x' and 'maxscale') automatically. DGS logo by Matthew Reimer, North Arrow Research. \
Your images are not stored. Larger imagery takes longer time. Multiply your grain sizes in pixels by a scaling in millimetre (or microns) per pixel for results in mm (or microns). This scaling might vary per image. \
Please use [github issues](https://github.com/dbuscombe-usgs/pyDGS/issues) for feedback, thanks! ")


col1,col2,col3,col4=st.columns(4)
with col1:
    st.button(label="Compute Grain Size Distributions",key="compute_button",on_click=compute_button)

with col4:
    st.download_button(
     label="Download zipped folder of results",
     data=create_zip(),
     file_name= 'dgs_results.zip', 
 )

#