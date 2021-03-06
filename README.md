# Streamlit app to Perform NER on Wikipedia text

## [Live Project Url](https://afternoon-shore-15753.herokuapp.com/)


### The requirment of the project are as follows :- 
- Scrap data using wikipedia API to perform NER
- Perform Named Entity Recognition on scrapped data and extract entities like city, person, organisation, Date, Geographical Entity, Product etc.
- Display annotated text in Streamlit App.

### Language Used
- Python

### Library, Packaged and API Used
- Wikipedia (API)
- Streamlit (Library)
- Spacy_streamlit (Package)
- Spacy (Library)

### Steps to access the live project
- Go to url (https://afternoon-shore-15753.herokuapp.com/)
![image](https://user-images.githubusercontent.com/54279054/114305866-90b03a00-9af7-11eb-9e02-48009f784185.png)

- Enter any keyword in the textbox area on which you want to perform the NER
 ![image](https://user-images.githubusercontent.com/54279054/114305949-da008980-9af7-11eb-9054-c4a5c6e29f5b.png)

- And just move your cursor out from the text area

- If entered keyword matches with any wikipedia page title you will see the output below.
 ![image](https://user-images.githubusercontent.com/54279054/114306023-1fbd5200-9af8-11eb-8406-eb406469acf4.png)

### How to install the project
#### Open VScode and open a terminal inside it and run the following steps
1. Clone this repository using the code below.
    - ```git clone https://github.com/aman2457/bi-ner.git```
  
2. Install the required package and libraries using command.
   - ```pip install -r requirements.txt```

3. Now run the below command in cli to open the app.
   - ```streamlit run app.py```


### Some information regarding the app.
1. The app fetch the text from wikipedia which matches with the user's keyword. If multiple pages found with same keyword then a random page is choosen.
2. The extracted text then loaded into a NLP model which peform NER.
3. After the NER the ouptut is feeded into a spacy_streamlit.visualize_ner funtion of streamlit_spacy which visualize the text based on NER.

### Caution :- 
1. If you are not getting output on the given keyword there may be folliwing reason:- 
    - Server get time out
    - No Wikipedia page matched with keyword
    - A page title unexpectedly resolves to a redirect
