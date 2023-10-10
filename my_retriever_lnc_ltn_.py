from collections import Counter
import numpy as np
import math
import time


class Retrieve:
    
    # Create new Retrieve object storing index and term weighting
    # scheme. (You can extend this method, as required.)
    def __init__(self, index, term_weighting):
        self.start_time = time.time()
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.collection_idf = self.compute_inverse_term_frequency()
        self.doc_v_l = self.compute_document_vectors_and_length()
        
        
        
                    
                
        
    def compute_number_of_documents(self):
        self.doc_ids = set() 
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)
    
    # Converts a query into a dictionary of term counts.
    def compute_dictionary_of_term_counts(self, query):  
        # query's terms counted by Counter, represented in a dictionary
        dict_term_counts = dict(Counter(query))
        return dict_term_counts
    
    def compute_document_vectors_and_length(self):
        doc_vectors = {}
        doc_lengths = {}
        for term in self.index:
            
            # An array that only contains all vectors of a document
            # Used for more efficient np operations for convenience
            lengths_array = []
            
            
            for doc_id in self.index[term]:
                if term in self.index:
                    if self.term_weighting == "tfidf" :
                        freq = 1 + math.log(self.index[term][doc_id])
                        freq = freq*self.collection_idf[term]
                    elif term in self.collection_idf:
                        freq = 1 + math.log(self.index[term][doc_id])
                    
                    if not (doc_id in doc_vectors):
                        doc_vectors[doc_id] = {}
                        
                    
                    

                lengths_array.append(freq)  
                doc_lengths[doc_id] = np.array(lengths_array)
                doc_lengths[doc_id] = np.sqrt(np.sum(doc_lengths[doc_id]*doc_lengths[doc_id]))
                doc_vectors[doc_id][term] = freq/doc_lengths[doc_id]
        return (doc_vectors, doc_lengths)
    
    # Gets the IDF of each term in the collection.
    def compute_inverse_term_frequency(self):
        dict_idf = {}
        # Loops over all terms in the collection to determine doc. freq.
        # Also calculates IDF value with each pass, and adds them to a dict. 
        for term in self.index:
            document_freq = len(self.index[term])
            idf = math.log(self.num_docs/document_freq)
            dict_idf[term] = idf  
        print (dict_idf)
        return dict_idf
    
    def compute_query_vectors_and_length(self, query):
        query_vectors = {}
        query_lengths = []
        dict_term_counts = self.compute_dictionary_of_term_counts(query) 
        for term in dict_term_counts:
            freq = 0
            
            if term in self.index:
                if self.term_weighting == "tfidf":
                    freq = 1 + math.log(dict_term_counts[term])
                    freq = freq*self.collection_idf[term]
                elif term in self.collection_idf:
                    freq = 1 + math.log(dict_term_counts[term])
                    
                if term in query_vectors:
                    query_vectors[term] += freq
                else:
                    query_vectors[term] = freq 
                    
                query_lengths.append(freq)
                
        query_lengths = np.array(query_lengths)
        query_lengths = np.sqrt(np.sum(query_lengths*query_lengths))
        return (query_vectors, query_lengths)

                
    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        query_v_l = self.compute_query_vectors_and_length(query)
        
        
        query_vectors = query_v_l[0]
        query_lengths = query_v_l[1]
        cosine_similarities = {}
        for doc_id in range(1, self.num_docs+1):
            qidi_numerator = 0
            length_doc_denominator = 0
            
            if self.term_weighting == "tf" or self.term_weighting == "tfidf":
                doc_vectors = self.doc_v_l[0]
                doc_lengths = self.doc_v_l[1]
                # length_doc_denominator = np.sqrt(np.sum(doc_lengths[doc_id]*doc_lengths[doc_id]))
                
            # for term in query:
            #     if term in self.index.keys() and doc_id in self.index[term].keys():
            #         query_numerator = query_vectors[term]
            #         doc_numerator = doc_vectors[doc_id][term]
            #         qidi_numerator += query_numerator*doc_numerator
                    
            # cosine_similarities[doc_id] = qidi_numerator/length_doc_denominator
            
            for term in query:
                if term in self.index.keys() and doc_id in self.index[term].keys():
                    query_numerator = query_vectors[term]
                    doc_numerator = doc_vectors[doc_id][term]
                    qidi_numerator += query_numerator*doc_numerator
                    
            cosine_similarities[doc_id] = qidi_numerator
            
            
                    
                    
                    
                    
        top_10 = sorted(cosine_similarities, key = cosine_similarities.get, reverse = True)[:10]
        
        return top_10