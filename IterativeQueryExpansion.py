import sys
import math
import string
import spacy
import requests
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from collections import defaultdict, OrderedDict
from spanbert import SpanBERT
from spacy_help_functions import get_entities, create_entity_pairs
from googleapiclient.discovery import build

API_KEY = sys.argv[1]
ENGINE_ID = sys.argv[2]
RELATION = int(sys.argv[3])
THRESHOLD = float(sys.argv[4])
QUERY = sys.argv[5]
N_RESULTS = int(sys.argv[6])
        

class IterativeQueryExpansion():
    """
    # Class Params: 
        - self.r: Specified relation value between 1-4
        - self.q: initial query extracted from command line
        - self.prev_queries: stored previous queries from all iterations. Initial value contains just processed self.q
        - self.k: number of results to be returned at the end of query expansion
        - self.entities_of_interest: Entities of interest to be passed as parameters for spaCy to extract
        - self.rel_prerequisites: structure of sub-obj tuple that is recognized by spanBERT for predicting specified relation
        - self.relationships: structure of extracted relations from spanBERT that we want to extract
        - self.bs_filtered_sections: beautiful soup sections to be filtered out during preprocessing
        - self.model: loaded pre-trained spanBERT model
    """ 
    def __init__(self):
        self.r = RELATION
        self.t = THRESHOLD
        self.q = QUERY
        self.prev_queries = {QUERY.title().replace(" ","_")}
        self.k = N_RESULTS
        self.X = dict() # Initialize empty tuples
        self.entities_of_interest = ["ORGANIZATION", "PERSON", "LOCATION", "CITY", "STATE_OR_PROVINCE", "COUNTRY"]
        self.rel_prerequisites = {1:{"Subj":{"PERSON"}, "Obj":{"ORGANIZATION"}}, 
                                  2:{"Subj":{"PERSON"}, "Obj":{"ORGANIZATION"}},
                                  3:{"Subj":{"PERSON"}, "Obj":{"DATE","LOCATION","STATE_OR_PROVINCE","COUNTRY"}},
                                  4:{"Subj":{"ORGANIZATION"}, "Obj":{"PERSON"}},}
        self.relationships = {1:"per:schools_attended", 
                              2:"per:employee_of", 
                              3:"per:cities_of_residence", 
                              4:"org:top_members/employees"}
        self.bs_filtered_sections = {'noscript', 'header', 'html', 'meta', 'head', 'input', 'script'}
        # Load pre-trained SpanBERT model
        self.model = SpanBERT("./pretrained_spanbert") 

    """
    Do: Main driver for iterative query expansion
    Input: None
    Output: Top-K spanBERT predicted tuples
    """    
    def main(self):
        # # Setup spaCy
        nlp = spacy.load("en_core_web_lg")

        iter_count = 0
        # While we don't have top-k results, iterate search 
        while len(self.X.keys()) < self.k:
            print(f"Starting iteration {iter_count}...")
            URL = 'https://www.googleapis.com/customsearch/v1?key=' + API_KEY + '&cx=' + ENGINE_ID + '&q=' + self.q
            searchResults = requests.get(url = URL).json()
            
            #
            # googleService = build("customsearch", "v1", developerKey=API_KEY)
            # searchResults = googleService.cse().list(q=self.q, cx=ENGINE_ID).execute()
            # print(searchResults)

            count = 0
            # Get HTML content for each page in results:
            for index in range(len(searchResults['items'])):
                # try:
                link = searchResults['items'][index]['link']
                print(f"\nURL ( {index+1} / 10): {link}")
                # print("\tFetching text from url ...")
                linkResults = requests.get(link).text
                soup = BeautifulSoup(linkResults,'lxml')

                # Cut text accordingly if needed
                text = soup.findAll(text=True)
                output = ''

                # Filter out unwanted sections (Source: https://matix.io/extract-text-from-webpage-using-beautifulsoup-and-python/):
                # TODO: Fine-tune filtered sections to optimize extracted texts
                for t in text:
                    if t.parent.name not in self.bs_filtered_sections:
                        output += '{}'.format(t)
                text = output
                if len(text) > 20000:
                    text = text[:20000]
                # print(text)
                
                with open(f'{count}.txt', 'w+') as f:
                    f.write(text)
                count += 1

                #Generate tokenized sentences using spaCy
                doc = nlp(text)
                sent_counter = 1
                for sentence in doc.sents:
                    sent_counter += 0
                    if sent_counter%5 == 0:
                        print(f"\tProcessing sentence {sent_counter}")
                    self.generate_relations(sentence)

                # except:
                #     print(f"Error with parsing/fetching URL. Continuing...")
                #     continue
            
            # At the end of every search iteration, resort relations by highest confidence
            self.X = {k: v for k, v in sorted(self.X.items(), key=lambda item: item[1][1], reverse=True)}
            # Increment iter_count
            iter_count += 1
            # Append to query if X not enough
            if len(self.X.keys()) < self.k:
                self.appendQuery()
            print(f"========== Total number of relations extracted: {len(self.X.keys())} ==========")
            print(f"========== Extracted relations: ==========")
            for key in self.X.keys():                
                print(f"\tEntity Pair: {key}, Predicted Relation and Confidence: {self.X[key]}")
            print(f"========== Next Query: {self.q} ==========")

        return self.X

    
    """
    Do: Uses spaCy and spanBERT to extract predictions for a given sentence input
    Input: Tokenized sentence from spaCy
    Output: None, automatically updates self.X
    """  
    def generate_relations(self,sentence):
        # print("\n\nProcessing sentence: {}".format(sentence))
        # print("Tokenized sentence: {}".format([token.text for token in sentence]))
        ents = get_entities(sentence, self.entities_of_interest)
        # print("spaCy extracted entities: {}".format(ents))
        
        # create entity pairs
        candidate_pairs = []
        sentence_entity_pairs = create_entity_pairs(sentence, self.entities_of_interest)
        for ep in sentence_entity_pairs:
            if ep[1][1] in self.rel_prerequisites[self.r]["Subj"] and ep[2][1] in self.rel_prerequisites[self.r]["Obj"]:
                candidate_pairs.append({"tokens": ep[0], "subj": ep[1], "obj": ep[2]})  # e1=Subject, e2=Object
            if ep[2][1] in self.rel_prerequisites[self.r]["Subj"] and ep[1][1] in self.rel_prerequisites[self.r]["Obj"]:
                candidate_pairs.append({"tokens": ep[0], "subj": ep[2], "obj": ep[1]})  # e1=Object, e2=Subject

        # Classify Relations for all Candidate Entity Pairs using SpanBERT
        candidate_pairs = [p for p in candidate_pairs if not p["subj"][1] in ["DATE", "LOCATION"]]  # ignore subject entities with date/location type

        # If no entity pairs identified, return
        if len(candidate_pairs) == 0:
            # print("No candidate pairs found... moving on to next sentence")
            return

        # print("Candidate entity pairs:")
        # for p in candidate_pairs:
            # print("Subject: {}\tObject: {}".format(p["subj"][0:2], p["obj"][0:2]))
        # print("Applying SpanBERT for each of the {} candidate pairs. This should take some time...".format(len(candidate_pairs)))
        relation_preds = self.model.predict(candidate_pairs)  # get predictions: list of (relation, confidence) pairs
        results = []

        # Generate Extracted Relations
        for ex, pred in list(zip(candidate_pairs, relation_preds)):
            entry = tuple((ex["subj"][0], ex["obj"][0]))
            # Filter by inputted relationship type
            if pred[0] == self.relationships[self.r] and pred[1] >= self.t:
                print("\n\tExtracted relation:")
                print("\t\tSubject: {}\tObject: {}\tRelation: {}\tConfidence: {:.2f}".format(ex["subj"][0], ex["obj"][0], pred[0], pred[1]))
                # Add directly to current list of extracted tuples
                if self.X.get(entry, None):
                    if self.X[entry][1] < pred[1]:
                        print(f"\t\t\tFor entry {entry}, found higher confidence of {pred[1]} compared to previous value of {self.X[entry][1]}. Overwriting...\n")
                        self.X[entry] = pred    
                else:
                    print(f"\t\t\tNo existing entry, adding tuple {entry} with confidence {pred[1]} to set X\n")
                    self.X[entry] = pred
            # else:
            #     print("\t\t\tWrong relation or Confidence of this extracted pair too low, ignoring...")

        return
    
    """
    Do: Appends highest-confidence tuple to query for next iteration of search
    Input: None
    Output: None
    """  
    def appendQuery(self):
        # TODO: Test to see if validaiton is working 100%? Are there cases where duplicate queries get through??
        # Check for duplicates
        for entry in self.X.keys():
            query = entry[0].replace(" ","_") + "_" + entry[1].replace(" ","_")
            # print("Query is: ", query)
            if query not in self.prev_queries:
                self.prev_queries.add(query)
                self.q += " " + query.replace("_"," ")
                # print("Updated query is: ", self.q)
                break
        return

if __name__ == "__main__":

    session = IterativeQueryExpansion()
    results = session.main()
    print(f"Final number of extracted tuples: {len(results.keys())}")

