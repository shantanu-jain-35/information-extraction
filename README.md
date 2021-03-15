# information-extraction
Developing an Information Extraction system, which fetches relational tuples from the unstructured documents on the Internet, based on a user seed query.

# Team Members

Phan Anh Nguyen (pn2363)

Shantanu Lalitkumar Jain (slj2142)

# Project Structure
- /src
    - iterativeSetExpansion.py
    - example_relations.py
    - spanbert.py
    - spacy_help_functions.py
    - relations.txt
    - /pretrained_spanbert
    	- config.json
		- pytorch_model.bin
    - /pytorch_pretrained_bert
    	- init.py
	- file_utils.py
	- modeling.py
	- optimization.py
	- tokenization.py
- /transcripts
    - transcript_2_0.7_bill_gates_microsoft_10.txt
    - /additional
    	- transcript1_1_0.7_mark_zuckerberg_harvard_10.txt
		- transcript2_2_0.7_sundar_pichai_google_10.txt
		- transcript3_3_0.7_megan_repinoe_redding_2.txt
		- transcript4_4_0.7_bill_gates_microsoft_10.txt
- requirements.txt
- README.md

# Environment
- Python Version: 3.6.9
- NLTK: 3.5

# Instructions
Please follow the below mentioned instructions for running our program in the VM:
1. Go to the project folder
```
cd proj2/
```
2. Install the dependencies,
```
pip3 install -r requirements.txt
```
3. Run the driver program with the required arguments (mentioned below)
**NOTE**: Remember to add the query in quotes for multi word query. Otherwise the system would consider only the first word as query.
```
python3 src/iterativeSetExpansion.py <google api key> <google engine id> <r> <t> "<query>" <k>
```

# Project design

Our product code contains a main driver called iterativeSetExpansion.py. It utilizes helper functions provided in spacy_helper_functions.py. The flow of our program works as follows: 

1. The user starts the program with a search term from the command line and specified relation, desired precision level, as well as number of tuples to return.
2. Our driver will then call the Google Custom Search API on the provided query terms. Here we use the [Google API Client Discovery](https://developers.google.com/discovery) service to call the API and collect the result documents.
3. We attempt to access html content from each link above using beautifulsoup4. We perform preprocessing where we extract specific sections from the bs4 soup content, as well as remove several special characters, before passing the document to spaCy for tokenization.
4. Each sentence is then processed using helper functions to extract potential candidate pairs. Out of identified candidate pairs, only those that match the format required for the specified relation are passed through to spanbert for processing.
5. Predicted relations are then checked for confidence threshold, whether it matches specified relation, and whether the relation already exists. 
6. As per the spec, at the end of each round of urls, we check to see if k-relations have been extracted. If not, we attempt to find the next query based on the relation with the highest confidence identified that has not been used for querying. If no such tuple exists, the program exits.
7. If k-relations have been extracted, we return all extracted relations.

# Step-3 methodology

+ In order to make sure we do not re-process previously seen URLs, we keep a set of all previous processed URLS. All new URLs obtained from the search API is matched against this set before processing
+ We removed all the sections with tags: script, headers and noscript. Since, we are mostly interested in the body of the webpage (which contains majority of the content), we observed removing this sections reduced the number of sentences to parse, and improved the relations extracted.
+ We also removed leading and trailing whitespaces, along with any citations which might interfere with the model. This allowed us to capture additional information within the 20,000 character limit. This was especially useful for extracting relations from wikipedia-based articles that contained many references.
+ Other preprocessing include, splitting multiple headlines into each line, and removing any extra new lines that might be present due to the structure of html.
+ Before calling spanbert predict on generated candidate pairs, we filtered out candidates that did not match the subject-object structure that the specified relation calls for. (I.e: If relation 4 was chosen, only candidate pairs where subj = ORGANIZATION and objc = PERSON are passed into spanbert).
+ To prevent duplicates, we check each generated relation from spanbert against the existing set of extracted relations. If the relation extracted by spanbert has higher confidence than the existing relation, we update that confidence, otherwise we ignore.

# Handling of Non-HTML files

For the purpose of our project, we have decided to ignore the non-html files. Therefore, any document set retrieved from the search engine, which is not a html file, will be ignored and **not** considered for precision and query expansion calculations. 


# References
+ Preprocessing Inspiration: https://stackoverflow.com/a/24968429
+ https://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text