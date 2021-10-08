import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import efficientnet.keras as efn
import streamlit as st
import SessionState
from skimage.transform import resize
import skimage
import skimage.filters
import reportgenerator
import style
from keras.models import Model, load_model
st.set_option('deprecation.showPyplotGlobalUse', False)

model = load_model('classifier.h5')


st.markdown(
    f"""
<style>
    .reportview-container .main .block-container{{
        max-width: {1000}px;
        padding-top: {5}rem;
        padding-right: {0}rem;
        padding-left: {0}rem;
        padding-bottom: {0}rem;
    }}
    .reportview-container .main {{     

    }}   
    [data-testid="stImage"] img {{
        margin: 0 auto;
        max-width: 500px;
    }}
</style>
""",
    unsafe_allow_html=True,
)

# main panel
logo = Image.open('dss_logo.png')
st.image(logo, width=None)
style.display_app_header(main_txt='Gleason Score Prediction for Prostate Cancer',
                         sub_txt='The intensity of prostate cancer metastasis in using artificial intelligence', is_sidebar=False)

# session state
ss = SessionState.get(page='home', run_model=False)


st.markdown('**Upload biopsy image to analyze**')
st.write('')
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg'])

med_opinion_list = ["The cancer cells look like healthy cells and PSA levels are low. However, cancer in this early stage is usually slow growing.",
                    "Well differentiated cells and PSA levels are medium. This stage also includes larger tumors found only in the prostate, as long as the cancer cells are still well differentiated. ",
                    "Moderately diffentiated cells and the PSA level is medium. The tumor is found only inside the prostate, and it may be large enough to be felt during DRE.",
                    "Moderately or poorly diffentiated cells and the PSA level is medium. The tumor is found only inside the prostate, and it may be large enough to be felt during DRE.",
                    "Poorly diffentiated cells. The cancer has spread beyond the outer layer of the prostate into nearby tissues. It may also have spread to the seminal vesicles. The PSA level is high.",
                    "Poorly diffentiated cells. The tumor has grown outside of the prostate gland and may have invaded nearby structures, such as the bladder or rectum.",
                    "Poorly diffentiated cells. The cancer cells across the tumor are poorly differentiated, meaning they look very different from healthy cells.",
                    "Poorly diffentiated cells. The cancer has spread to the regional lymph nodes.",
                    "Poorly diffentiated cells. The cancer has spread to distant lymph nodes, other parts of the body, or to the bones.",
                    ]

if uploaded_file is not None:
    # uploaded_file.read()
    image = Image.open(uploaded_file)
    st.image(image, caption='Biopsy image', use_column_width=True)
    im_resized = image.resize((224, 224))
    im_resized = resize(np.asarray(im_resized), (224, 224, 3))

    # grid section
    col1, col2, col3 = st.columns(3)
    col1.header('Resized Image')
    col1.image(im_resized, caption='Biopsy image', use_column_width=False)
    with col2:
        st.header('Gray Image')
        gray_image = skimage.color.rgb2gray(im_resized)
        st.image(gray_image, caption='preprocessed image',
                 use_column_width=False)

    with col3:
        st.header('Spotted Pattern')
        # sigma = float(sys.argv[2])
        gray_image = skimage.color.rgb2gray(im_resized)
        blur = skimage.filters.gaussian(gray_image, sigma=1.5)
        # perform adaptive thresholding
        t = skimage.filters.threshold_otsu(blur)
        mask = blur > t
        sel = np.zeros_like(im_resized)
        sel[mask] = im_resized[mask]
        st.image(sel, caption='preprocessed image', use_column_width=False)

    preds = model.predict(np.expand_dims(im_resized, 0))
    data = (preds[0]*100).round(2)
    isup_data = [data[0], data[1], data[2], data[3],
                 data[4]+data[5]+data[6], data[7]+data[8]+data[9]]
    gleason_label = ['0+0', '3+3', '3+4', '4+3',
                     '4+4', '3+5', '5+3', '4+5', '5+4', '5+5']
    gleason_colors = ['yellowgreen', 'red', 'gold', 'lightskyblue',
                      'cyan', 'lightcoral', 'blue', 'pink', 'darkgreen', 'yellow']
    isup_label = ['0', '1', '2', '3', '4', '5']
    isup_colors = ['gold', 'lightskyblue', 'cyan', 'lightcoral', 'blue']

    col1, col2, = st.columns(2)
    with col1:
        reportgenerator.visualize_confidence_level(data, label=gleason_label, ylabel='GleasonScore Pattern Scale',
                                                   title='GleasonScore Prediction ')
    with col2:
        reportgenerator.pieChart(data, label=gleason_label, colors=gleason_colors,
                                 title='GleasonScore Prediction Distribution', startangle=120)

    col1, col2, = st.columns(2)
    with col1:
        reportgenerator.pieChart(isup_data, label=isup_label, colors=isup_colors,
                                 title='ISUP Pattern Scale Prediction Distribution', startangle=45)
    with col2:
        reportgenerator.visualize_confidence_level(isup_data, label=isup_label, ylabel='ISUP Pattern Scale',
                                                   title='ISUP Prediction')

    opinion = list(data).index(max(list(data)))
    style.display_app_header(main_txt='Medical Report Proposition:',
                             sub_txt=med_opinion_list[opinion], is_sidebar=False)
