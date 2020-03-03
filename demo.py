import streamlit as st
from score import Score
import cv2
import numpy as np


@st.cache(allow_output_mutation=True)
def init():
    # init model
    score = Score(model_data_path='./data/initModel.data', mean_data_path='./data/mean_AADB.txt')

    return score


def main():
    score = init()

    st.title('Aesthetics Ranking Demo')

    image_file = st.file_uploader(label='Upload an image.')
    image_show = st.empty()
    image_score = st.empty()

    if image_file is not None:
        img = cv2.imdecode(np.asarray(bytearray(image_file.read())), cv2.IMREAD_COLOR)
        image_show.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=400)
        result = score.score_one_image(img)
        image_score.markdown('The aesthetics socre: `%f`.' % result)


if __name__ == '__main__':
    main()
