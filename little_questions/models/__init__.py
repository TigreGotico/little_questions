import logging
import pickle
from os.path import join, isfile

import nltk
import requests
from xdg import BaseDirectory as XDG

try:
    import JarbasModelZoo as _JarbasModelZoo
    from JarbasModelZoo import LOG
except ImportError:
    _JarbasModelZoo = None
    LOG = logging.getLogger(__name__)

MODEL2URL = {
    "questions52_EN":
        "https://github.com/OpenJarbas/little_questions/releases/download/0.7.0a1/questions52_svm_EN_0.7.0a1.pkl",
    "questions6_EN":
        "https://github.com/OpenJarbas/little_questions/releases/download/0.7.0a1/questions6_svm_EN_0.7.0a1.pkl",
    "questions52_ES":
        "https://github.com/OpenJarbas/little_questions/releases/download/0.7.0a1/questions52_svm_ES_googtx_0.7.0a1.pkl",
    "questions6_ES":
        "https://github.com/OpenJarbas/little_questions/releases/download/0.7.0a1/questions6_svm_ES_googtx_0.7.0a1.pkl",
    "questions52_CA":
        "https://github.com/OpenJarbas/little_questions/releases/download/0.7.0a1/questions52_svm_CA_googtx_0.7.0a1.pkl",
    "questions6_CA":
        "https://github.com/OpenJarbas/little_questions/releases/download/0.7.0a1/questions6_svm_CA_googtx_0.7.0a1.pkl",
    "questions52_PT":
        "https://github.com/OpenJarbas/little_questions/releases/download/0.7.0a1/questions52_svm_PT_googtx_0.7.0a1.pkl",
    "questions6_PT":
        "https://github.com/OpenJarbas/little_questions/releases/download/0.7.0a1/questions6_svm_PT_googtx_0.7.0a1.pkl",
    "questions52_FR":
        "https://github.com/OpenJarbas/little_questions/releases/download/0.7.0a1/questions52_svm_FR_googtx_0.7.0a1.pkl",
    "questions6_FR":
        "https://github.com/OpenJarbas/little_questions/releases/download/0.7.0a1/questions6_svm_FR_googtx_0.7.0a1.pkl",
    "questions52_DE":
        "https://github.com/OpenJarbas/little_questions/releases/download/0.7.0a1/questions52_svm_DE_googtx_0.7.0a1.pkl",
    "questions6_DE":
        "https://github.com/OpenJarbas/little_questions/releases/download/0.7.0a1/questions6_svm_DE_googtx_0.7.0a1.pkl",
    "questions52_IT":
        "https://github.com/OpenJarbas/little_questions/releases/download/0.7.0a1/questions52_svm_IT_googtx_0.7.0a1.pkl",
    "questions6_IT":
        "https://github.com/OpenJarbas/little_questions/releases/download/0.7.0a1/questions6_svm_IT_googtx_0.7.0a1.pkl",
}


def download(model_id, force=False):
    if model_id in MODEL2URL:
        url = MODEL2URL[model_id]
        model_id = url.split("/")[-1].replace(".pkl", "")
    else:
        raise ValueError("invalid model_id")
    path = join(XDG.save_data_path("little_questions"), model_id + ".pkl")
    if isfile(path) and not force:
        LOG.info("Already downloaded " + model_id)
        return
    LOG.info("downloading " + model_id)
    LOG.info(url)
    LOG.info("this might take a while...")
    with open(path, "wb") as f:
        f.write(requests.get(url).content)
    return path


def load_model(model_id):
    if isfile(model_id):
        path = model_id
    else:
        path = join(XDG.save_data_path("little_questions"), model_id + ".pkl")
        LOG.debug("loading: " + path)
        if not isfile(path):
            path = download(model_id)
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


LANG2MODEL = {
    "en": join(XDG.xdg_data_home, "little_questions",
               "questions52_svm_EN_0.7.0a1.pkl"),
    "en_small": join(XDG.xdg_data_home, "little_questions",
                     "questions6_svm_EN_0.7.0a1.pkl"),
    "es": join(XDG.xdg_data_home, "little_questions",
               "questions52_svm_ES_googtx_0.7.0a1.pkl"),
    "es_small": join(XDG.xdg_data_home, "little_questions",
                     "questions6_svm_ES_googtx_0.7.0a1.pkl"),
    "es_tagger": join(XDG.xdg_data_home, "JarbasModelZoo",
                      "nltk_cess_esp_udep_brill_tagger.pkl"),
    "ca": join(XDG.xdg_data_home, "little_questions",
               "questions52_svm_CA_googtx_0.7.0a1"),
    "ca_small": join(XDG.xdg_data_home, "little_questions",
                     "questions6_svm_CA_googtx_0.7.0a1"),
    "ca_tagger": join(XDG.xdg_data_home, "JarbasModelZoo",
                      "nltk_cess_cat_udep_brill_tagger.pkl"),
    "fr": join(XDG.xdg_data_home, "little_questions",
               "questions52_svm_FR_googtx_0.7.0a1"),
    "fr_small": join(XDG.xdg_data_home, "little_questions",
                     "questions6_svm_FR_googtx_0.7.0a1"),
    "it": join(XDG.xdg_data_home, "little_questions",
               "questions52_svm_IT_googtx_0.7.0a1"),
    "it_small": join(XDG.xdg_data_home, "little_questions",
                     "questions6_svm_IT_googtx_0.7.0a1"),
    "de": join(XDG.xdg_data_home, "little_questions",
               "questions52_svm_DE_googtx_0.7.0a1"),
    "de_small": join(XDG.xdg_data_home, "little_questions",
                     "questions6_svm_DE_googtx_0.7.0a1"),
    "pt": join(XDG.xdg_data_home, "little_questions",
               "questions52_svm_PT_googtx_0.7.0a1.pkl"),
    "pt_small": join(XDG.xdg_data_home, "little_questions",
                     "questions6_svm_PT_googtx_0.7.0a1.pkl"),
    "pt_tagger": join(XDG.xdg_data_home, "JarbasModelZoo",
                      "nltk_floresta_macmorpho_brill_tagger.pkl")
}


# download resources
def download_en():
    download("questions52_EN")
    download("questions6_EN")
    # TODO check directly if file exists for speed
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)


def download_pt():
    download("questions52_PT")
    download("questions6_PT")
    if _JarbasModelZoo is not None:
        _JarbasModelZoo.download("nltk_floresta_macmorpho_brill_tagger")


def download_es():
    download("questions52_ES")
    download("questions6_ES")
    if _JarbasModelZoo is not None:
        _JarbasModelZoo.download("nltk_cess_esp_udep_brill_tagger")


def download_fr():
    download("questions52_FR")
    download("questions6_FR")


def download_it():
    download("questions52_IT")
    download("questions6_IT")


def download_de():
    download("questions52_DE")
    download("questions6_DE")


def download_ca():
    download("questions52_CA")
    download("questions6_CA")
    if _JarbasModelZoo is not None:
        _JarbasModelZoo.download("nltk_cess_cat_udep_brill_tagger")


# get path to downloaded models
def get_model_path(model="en"):
    if model.startswith("http"):
        # TODO download
        raise NotImplementedError("downloading models from url not supported")
    elif isfile(model):
        # user provided a full path
        return model
    if model in LANG2MODEL:
        # default models, download nltk data if needed
        if model.startswith("en"):
            download_en()
        elif model.startswith("es"):
            download_es()
        elif model.startswith("pt"):
            download_pt()
        elif model.startswith("ca"):
            download_ca()
        elif model.startswith("fr"):
            download_fr()
        elif model.startswith("de"):
            download_de()
        elif model.startswith("it"):
            download_it()
        return LANG2MODEL.get(model)
    raise ValueError("unknown model")
