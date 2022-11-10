# -*- coding: utf-8 -*-

"""A module for downloading from gdrive."""

import gdown
import requests


def download_file_from_google_drive(id, destination, use="gdown"):
    """

    Parameters
    ----------
    id : str
    google file id of the file to be downloaded

    destination : str
    set the name of the file to download

    use:
    options {gdown , requests}
    choose between the alternatives.

    Description
    ------------
    A method to download file from g drive. The file Id is to be passed as a parameter.

    """
    if use == "request":
        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        def save_response_content(response, destination):
            CHUNK_SIZE = 32768

            with open(destination, "wb") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)

        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(URL, params={'id': id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, destination)

    if use == "gdown":
        gdown.download(id=id, output=destination, quiet=False)