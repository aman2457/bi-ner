# Core Pkgs
import wikipedia
import random


# NLP Pkgs
import spacy_streamlit
import spacy
import streamlit as st
from spacy_streamlit import visualize_ner

# loading the model
nlp = spacy.load('en_core_web_sm')

# utility function to fetch data from wikipedia


def wikiExtract(keyword):
    keyword2 = 'Wikipedia is a free online encyclopedia, created and edited by volunteers around the world and hosted by the Wikimedia Foundation.'
    try:
        keyword1 = wikipedia.page(keyword)
        return str(keyword1.content)

    # catching the error Disambuguation and returning a random page.
    except wikipedia.DisambiguationError as e:
        s = random.choice(e.options)
        keyword1 = wikipedia.page(s)
        return str(keyword1.content)

    # catching all other exception and returning the demo text of wikipedia
    except (wikipedia.exceptions.HTTPTimeoutError,
            wikipedia.exceptions.PageError,
            wikipedia.exceptions.RedirectError,
            wikipedia.exceptions.WikipediaException):
        return str(keyword2)


def main():
    st.set_page_config(page_title='BITask-AmanKumar', page_icon=None,
                       layout='centered', initial_sidebar_state='auto')
    st.title("Named Entity Recognition on scrapped data form wikipedia")
    st.subheader(
        "Enter the keyword you want to search on wikipedia and perform NER")

    # getting the keyword from user
    input = st.text_area("Enter the keyword", " ")
    st.text("Wait.. Ner is getting performed(2-15 seconds)")
    st.text("If you are still seeing the wikipedia information below then one these may be the possible reasons-\n1- Servers times out\n2.No Wikipedia matched with keyword\n3.A page title unexpectedly resolves to a redirect\n ")

    # calling the utility function to fetch the data
    raw = wikiExtract(input)

    # performing the ner on the text and stroing it in docx
    docx = nlp(raw)

    # visulaizing the text with NER
    spacy_streamlit.visualize_ner(
        docx, labels=nlp.get_pipe('ner').labels)


# calling the main function
if __name__ == '__main__':
    main()
