from collections import Counter
import numpy as np
import math

class Retrieve:
    
    # Create new Retrieve object storing index and term weighting
    # scheme. (You can extend this method, as required.)
    def __init__(self, index, term_weighting):
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
    
    # Maps all vectors and lengths of documents into 2 separate dictionaries
    # for each.
    def compute_document_vectors_and_length(self):
        
        # Stores all vectors of a document. Key = doc_id. Value = another
        # dictionary with term as key, and frequency as value.
        doc_vectors = {}
        
        # Same as above. But value stores a list of unlabelled vectors 
        doc_vectors_unlabelled = {}
        
        for term in self.index:
            
            
            # Find all vectors (term freq) of a document and apply log of tf.
            # (Does not apply to binary)
            for doc_id in self.index[term]:
                    
                # Apply log normalisation to tf and multiply by idf for tfidf
                if self.term_weighting == "tfidf" :
                    freq = 1 + math.log(self.index[term][doc_id])
                    freq = freq*self.collection_idf[term]
                # Only apply log normalisation to tf for tf
                elif self.term_weighting == "tf":
                    freq = 1 + math.log(self.index[term][doc_id])
                # Term appears in document, so 1 for binary mode
                elif self.term_weighting == "binary":
                    freq = 1
                
                # Creates an empty dictionary for the doc_id key
                # Will return a KeyError error if not done
                if not (doc_id in doc_vectors):
                    doc_vectors[doc_id] = {}
                    doc_vectors_unlabelled[doc_id] = []
                
                # Attaches tf to both dictionary
                doc_vectors[doc_id][term] = freq
                doc_vectors_unlabelled[doc_id].append(freq)
                    
        
        # Return the dictionaries as a tuple for easy access
        return (doc_vectors, doc_vectors_unlabelled)
    
    # Gets the IDF of each term in the collection.
    def compute_inverse_term_frequency(self):
        dict_idf = {}
        # Loops over all terms in the collection to determine doc. freq.
        # Also calculates IDF value with each pass, and adds them to a dict. 
        for term in self.index:
            document_freq = len(self.index[term])
            idf = math.log(self.num_docs/document_freq)
            dict_idf[term] = idf  
        return dict_idf
    
    # Maps all query vectors into a dictionary.
    def compute_query_vectors(self, query):
        query_vectors = {}
         
        for term in self.dict_term_counts:
            freq = 0
            
            if term in self.index:
                if self.term_weighting == "tfidf":
                    freq = 1 + math.log(self.dict_term_counts[term])
                    freq = freq*self.collection_idf[term]
                elif self.term_weighting == "tf":
                    freq = 1 + math.log(self.dict_term_counts[term])
                elif self.term_weighting == "binary":
                    freq = 1
                    
                query_vectors[term] = freq 
                    
        return query_vectors

                
    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        
        self.dict_term_counts = self.compute_dictionary_of_term_counts(query)
        query_vectors = self.compute_query_vectors(query)
        cosine_similarities = {}
        
        # Loops through the document set for a single query to perform
        # cosine similarity calculations.
        for doc_id in range(1, self.num_docs+1):
            
            # Cosine similarity numerator
            qidi_numerator = 0
            
            # Cosine similarity denominator
            # Length of query excluded because it's the same for all docs.
            length_doc_denominator = 0
            
            doc_vectors = self.doc_v_l[0]
            doc_lengths = self.doc_v_l[1]
            
            # Calculates Euclidean distance with np array manipulations
            l_a_np = np.array(doc_lengths[doc_id])
            length_doc_denominator = np.sqrt(np.sum(l_a_np*l_a_np))
            
            # Iterate through all unique terms in query
            for term in self.dict_term_counts:
                
                # If term exists in index, and doc_id also exists, and
                # the document vector for the current term is not zero...
                if term in self.index.keys() and \
                doc_id in self.index[term].keys() and \
                doc_vectors[doc_id][term] != 0:
                    
                    # ...add the result of the doc vector and the query vector
                    # for the current term into the numerator
                    doc_numerator = doc_vectors[doc_id][term]
                    query_numerator = query_vectors[term]
                    qidi_numerator += query_numerator*doc_numerator
            
            # After iterating through the query's terms, calculate cosine
            # similarity for that document and add it to a dictionary
            cos_sim = qidi_numerator/length_doc_denominator
            if cos_sim > 0:
                cosine_similarities[doc_id] = cos_sim
            
            
                    
                    
                    
        results = sorted(cosine_similarities, key = cosine_similarities.get, reverse = True)[:10]
        return results