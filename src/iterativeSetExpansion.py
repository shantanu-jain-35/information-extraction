"""
Module for extracting structured top-k tuples from an unstructured database on the web using
Iterative Set Expansion
"""

import sys
import math
import string
import spacy
import requests
import nltk
import re
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from collections import defaultdict, OrderedDict
from spanbert import SpanBERT
from spacy_help_functions import get_entities, create_entity_pairs
from googleapiclient.discovery import build



class IterativeSetExpansion:
    """
    Contains the entire codebase for performing top-k tuple extraction using iterative set expansion
    """
    def __init__(self, r, k, q, t, client_key, engine_key):
        """
        Constructor
        @params:
            :r: int, [1-4] indicating the relation whose tuples are to be extracted
            :k: int, top-k tuples to be extracted
            :t: float, a threshold confidence for each extracted relation
            :q: string, query to be given in each iteration
            :client_key: string, the JSON API Key
            :engine_key: string, the Google API ID
            :X: dict, ((subj, obj) --> (relation, confidence)), the set of tuples extracted
            :relationships: dict, the predefined relations that we are interested in
            :model: SpanBERT, the pre-trained span bert model
            :entities_of_interest: list, of all the entities that we are looking to extract
            :rel_prerequisites: dict(), the entity pairs that we are looking for in a sentence
            :previous_queries: list, the set of all the queries given
            :visited_urls: list, of all the urls visited
        """
        self.r = r
        self.k = k
        self.t = t
        self.q = q
        self.client_key = client_key
        self.engine_key = engine_key
        self.X = dict()
        self.relationships = {1:"per:schools_attended", 
                              2:"per:employee_of", 
                              3:"per:cities_of_residence", 
                              4:"org:top_members/employees"}
        # self.bs_filtered_sections = ['noscript', 'header', 'html', 'meta', 'head', 'input', 'script', 'style']
        self.model = SpanBERT(pretrained_dir="./src/pretrained_spanbert")
        self.entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
        self.rel_prerequisites = {1:{"Subj":{"PERSON"}, "Obj":{"ORGANIZATION"}}, 
                                  2:{"Subj":{"PERSON"}, "Obj":{"ORGANIZATION"}},
                                  3:{"Subj":{"PERSON"}, "Obj":{"DATE","LOCATION","STATE_OR_PROVINCE","COUNTRY"}},
                                  4:{"Subj":{"ORGANIZATION"}, "Obj":{"PERSON"}}}
        self.previous_queries = [q]
        self.visited_urls = []
    
    def extract_structured_tuples(self):
        """
        The main function which performs the steps to extract top-k tuples from the databse of web.
        @params:
            :None
        @returns:
            X: dict(), a set of extracted tuples.
        """
        num_iterations = 0
        while len(self.X.keys()) < self.k:
            print("Parameters:")
            print("Client Key\t=\t", self.client_key)
            print("Engine Key\t=\t", self.engine_key)
            print("Relation  \t=\t", self.relationships[self.r])
            print("Threshold \t=\t", str(self.t))
            print("Query     \t=\t", self.q)
            print("# of Tuples\t=\t", str(self.k))
            print("========== Iteration: {} - Query: {} ==========".format(num_iterations+1, self.q))

            # Fetching a list of ten urls from the google api
            google_service = build("customsearch", "v1", developerKey=self.client_key)
            result_set = google_service.cse().list(q=self.q, cx=self.engine_key).execute()
            document_set = result_set['items']

            # For each item, extracting relations from the url
            for document_index in range(len(document_set)):
                url = document_set[document_index]['link']
                print("URL ({}/10): {}".format(document_index+1, url))
                if url in self.visited_urls:
                    print("The url has already been seen. Ignoring this.")
                    continue
                self.visited_urls.append(url)
                print("\tFetching text from url...")
                extracted_text = self.fetch_text_from_url(url)
                if extracted_text == None:
                    continue
                tokenized_documents = self.tokenize_documents(extracted_text)
                # Checking format of texts
                # with open(f'./extracted_texts/{document_index}.txt', 'w+') as f:
                #     f.write(str(extracted_text))
                #     f.write("\n\n\n\n=====================================================================================================\n\n\n\n")
                #     f.write(str(tokenized_documents))
                #     f.close()
                self.extract_relations(tokenized_documents)

            # Sort X as per the confidence thresholds
            self.X = {k: v for k, v in sorted(self.X.items(), key=lambda item: item[1][1], reverse=True)}
            num_iterations += 1
            
            print("================== ALL RELATIONS for {} ({}) ==================".format(
                self.relationships[self.r],
                len(self.X.keys())))
            for key in self.X.keys():
                print("Confidence: {} \t\t | Subject: {} \t\t | Object: {}".format(
                    self.X[key],
                    key[0],
                    key[1]
                ))

            # Checking the termination criteria
            if len(self.X.keys()) < self.k:
                continue_search = self.generate_queries()
                if not continue_search:
                    print("ISE has stalled, halting search and returning all found tuples:")
                    break
            
        return self.X
    
    def fetch_text_from_url(self, url):
        """
        Fetches text from webpages, and filters them to pass to spacy
        @params:
            :url: the webpage url from which the text will be extracted
        @returns:
            filtered_text: the webpage text after it has been filtered.
        """
        try:
            # adding a timeout of 5 seconds
            results = requests.get(url, timeout=5).text
        except Exception as err:
            print("Timeout occurred while fetching the contents of the url. Continuing...")
            return None
        soup = BeautifulSoup(results, 'lxml')

        # text filtering
        for script in soup(["script", "noscript", "header"]):
            script.extract()
        text = soup.get_text(separator='\n')
        lines = (re.sub('\[(.*?)\]','[]',line) for line in text.splitlines())
        # code inspired from: https://stackoverflow.com/a/24968429
        # break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # drop blank lines
        filtered_text = ' '.join(chunk for chunk in chunks if chunk)
        # keeping only the first 20000 characters
        if len(filtered_text) > 20000:
            filtered_text = filtered_text[:20000]
        print("\tWebpage Length (num characters): {}".format(len(filtered_text)))
        return filtered_text

    def tokenize_documents(self, text):
        """
        Tokenizes the documents into sentences using spacy
        @params:
            :text: the webpage content obtained from beautiful soup 
        @returns:
            document: the tokenized document after passing through spacy
        """
        print("\tAnnotating the webpage using spacy...")
        nlp = spacy.load("en_core_web_lg")
        nlp.add_pipe("sentencizer", before="parser")
        document = nlp(text)
        return document

    def extract_relations(self, document):
        """
        Extracts relations from each sentence (if any) by using pre-trained span bert model
        @params:
            :document: the tokenized document from spacy
        @returns:
            :None
        """
        num_sentences = len(list(document.sents))
        print("\tExtracted {} sentences...".format(len(list(document.sents))))
        print("\tProcessing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...")
        sentence_counter = 0
        for sentence in document.sents:
            sentence_counter += 1
            if sentence_counter % 5 == 0:
                print("\tProcessed {} / {}  sentences...".format(sentence_counter, num_sentences))
            candidate_pairs = self.create_candidate_pairs(sentence)
            if len(candidate_pairs) == 0:
                continue
            relation_predictions = self.model.predict(candidate_pairs)
            self.check_threshold_and_generate_relations(sentence, candidate_pairs, relation_predictions)
    
    def create_candidate_pairs(self, sentence):
        """
        Generates and filters potential candidate pairs
        @params:
            :document: the tokenized sentence from spacy
        @returns:
            :Candidate potential pairs
        """
        candidate_pairs = []
        sentence_entity_pairs = create_entity_pairs(sentence, self.entities_of_interest)
        for entity_pair in sentence_entity_pairs:
            if entity_pair[1][1] in self.rel_prerequisites[self.r]["Subj"] and entity_pair[2][1] in self.rel_prerequisites[self.r]["Obj"]:
                candidate_pairs.append({"tokens": entity_pair[0], "subj": entity_pair[1], "obj": entity_pair[2]})  # e1=Subject, e2=Object
            if entity_pair[2][1] in self.rel_prerequisites[self.r]["Subj"] and entity_pair[1][1] in self.rel_prerequisites[self.r]["Obj"]:
                candidate_pairs.append({"tokens": entity_pair[0], "subj": entity_pair[2], "obj": entity_pair[1]})  # e1=Object, e2=Subject            
        
        candidate_pairs = [p for p in candidate_pairs if not p["subj"][1] in ["DATE", "LOCATION"]]  # ignore subject entities with date/location type
        return candidate_pairs

    def check_threshold_and_generate_relations(self, sentence, candidate_pairs, relation_predictions):
        for example, prediction in list(zip(candidate_pairs, relation_predictions)):
            entity = tuple((example['subj'][0], example['obj'][0]))
            # check if it matches the input relationship
            if prediction[0] == self.relationships[self.r]:
                print("\n\t\t=== Extracted Relation ===")
                print("\t\tSentence: {}".format(sentence.text.strip('\t')))
                print("\t\tConfidence: {} ; Subject: {} ; Object: {} ;".format(
                    prediction[1],
                    example['subj'][0],
                    example['obj'][0]
                    ))
                if prediction[1] >= self.t:
                # adding to the extracted tuples list
                # X --> ((subj, obj) --> (relation, confidence))
                    if entity in self.X.keys():
                        if self.X[entity][1] <= prediction[1]:
                            # old tuple found with higher confidence
                            print("\t\tNew tuple found with higher confidence. Replacing the older one.")
                            self.X[entity] = prediction
                        else:
                            print("\t\tDuplicate with lower confidence than existing record. Ignoring this.")
                    else:
                        print("\t\tAdding to set of extracted relations")
                        self.X[entity] = prediction
                else:
                    print("Confidence is lower than threshold confidence. Ignoring this.")
                print("\t\t==========================\n")
        
        return

    def generate_queries(self):
        continue_search = False
        for entity in self.X.keys():
            new_query = entity[0]+" "+entity[1]
            if new_query not in self.previous_queries:
                self.previous_queries.append(new_query)
                self.q = new_query
                # If we find a usable query, continue search
                continue_search = True
                break
        return continue_search
            
def main():
    API_KEY = sys.argv[1]
    ENGINE_ID = sys.argv[2]
    RELATION = int(sys.argv[3])
    THRESHOLD = float(sys.argv[4])
    QUERY = sys.argv[5]
    N_RESULTS = int(sys.argv[6])
    session = IterativeSetExpansion(RELATION, N_RESULTS, QUERY, THRESHOLD, API_KEY, ENGINE_ID)
    result_set = session.extract_structured_tuples()
    print("Final number of extracted tuples: {}".format(len(result_set.keys())))

if __name__ == "__main__":
    main()