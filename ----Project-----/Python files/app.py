from typing import Any, Dict, List
import pandas as pd
import  numpy as np
import mlflow
from prefect import task, flow
from tqdm import tqdm, tqdm_notebook

from sklearn.model_selection import train_test_split        
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
import gensim
import gensim.downloader as api
from gensim.models import KeyedVectors
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from pickle import load,dump
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV



# Global variables
token_words = []
clean_text_stem = []
len_stem = []
clean_text_lemm = []
len_lemm = []
pos_tag = [] 

@task
def load_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    dat = data[0:100]
    return dat

@task
def remove_null(data: pd.Series) -> pd.Series:
    data = data.dropna()
    return data  
    
@task
def preprocess(x:Any) -> pd.DataFrame:
    x = str(x).lower()  # Lowering all text to covert all of them to there base form
    # Replacing commonly use words or numbers like 1,000 to 1k and 1,000,000 to 1m and currency symbol to there respective names and many other symbols to there name
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " is")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will").replace("@","at")
    
    # Renaming 1000 to 1k and 1000000 to 1m (the onces which may not be seperated with commans)
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    #Remove any special character like [= , ' ; "" ']
    pattern = re.compile('\W')
    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)
    # Removing HTML tags
    x = BeautifulSoup(x)
    x = x.get_text()
    # tokenize into words
    x = x.split()
    # remove stop words                
    x = [t for t in x if t not in stopwords.words("english")]          
    return pd.Series(" ".join(x))

@task
def feature_engineering(data : pd.DataFrame) -> pd.DataFrame:
    ## count the number of times each question in question1 and  question2 has repeated
    data['freq_qid1'] = data.groupby('qid1')['qid1'].transform('count') 
    data['freq_qid2'] = data.groupby('qid2')['qid2'].transform('count')
    
    ## Finding the length of question1 and question2 after the preprocessing is completed
    data['q1len'] = data['question1'].str.len()
    data['q2len'] = data['question2'].str.len()

    ## counting the number of words in question1 and question2
    data['q1_n_words'] = data['question1'].apply(lambda row: len(str(row).split(" ")))
    data['q2_n_words'] = data['question2'].apply(lambda row: len(str(row).split(" ")))

    def normalized_word_Common(row):
        w1 = set(map(lambda word: str(word).lower().strip(), str(row['question1']).split(" ")))
        w2 = set(map(lambda word: str(word).lower().strip(), str(row['question2']).split(" ")))    
        return 1.0 * len(w1 & w2)
    data['word_Common'] = data.apply(normalized_word_Common, axis=1)

    def normalized_word_Total(row):
        w1 = set(map(lambda word: str(word).lower().strip(), str(row['question1']).split(" ")))
        w2 = set(map(lambda word: str(word).lower().strip(), str(row['question2']).split(" ")))    
        return 1.0 * (len(w1) + len(w2))
    data['word_Total'] = data.apply(normalized_word_Total, axis=1)

    def normalized_word_share(row):
        w1 = set(map(lambda word: str(word).lower().strip(), str(row['question1']).split(" ")))
        w2 = set(map(lambda word: str(word).lower().strip(), str(row['question2']).split(" ")))    
        return 1.0 * len(w1 & w2)/(len(w1) + len(w2))
    data['word_share'] = data.apply(normalized_word_share, axis=1)

    data['freq_q1+q2'] = data['freq_qid1']+data['freq_qid2']
    data['freq_q1-q2'] = abs(data['freq_qid1']-data['freq_qid2'])
    
    # Apply Natural logarithm to the word_share column
    data['log_word_share'] = data['word_share'].apply(lambda x: np.log10(x))
        
    return data


@task
def text_preprocess(row:pd.Series):
    #tokenization of the questions
    token = str(row).split()
    token_words.append(token)
    # stemming the questions
    stem = [stemmer.stem(word) for word in token]
    clean_text_stem.append(" ".join(stem))
    len_stem.append(len(stem))
    # lemmitization
    lemm = [lemmatizer.lemmatize(word) for word in token]
    clean_text_lemm.append(' '.join(lemm))
    len_lemm.append(len(lemm))
    # applying parts of speech 
    pos = [nltk.pos_tag([word]) for word in lemm]
    pos_tag.append(pos)
    
@task 
def add_preprocessed_data1(data :pd.DataFrame) -> pd.DataFrame:
    data['preproc_ques1_token'] = token_words
    data['preproc_ques1_stem'] = clean_text_stem
    data['preproc_ques1_stem_len'] = len_stem
    data['preproc_ques1_lemm'] = clean_text_lemm
    data['preproc_ques1_lemm_len'] = len_lemm
    data['preproc_ques1_pos'] = pos_tag
   
    ## let us clear the content of list
    token_words.clear()
    clean_text_stem.clear()
    len_stem.clear()
    clean_text_lemm.clear()
    len_lemm.clear()
    pos_tag.clear()
    return data

@task 
def add_preprocessed_data2(data :pd.DataFrame) -> pd.DataFrame:
    data['preproc_ques2_token'] = token_words
    data['preproc_ques2_stem'] = clean_text_stem
    data['preproc_ques2_stem_len'] = len_stem
    data['preproc_ques2_lemm'] = clean_text_lemm
    data['preproc_ques2_lemm_len'] = len_lemm
    data['preproc_ques2_pos'] = pos_tag
   
    ## let us clear the content of list
    token_words.clear()
    clean_text_stem.clear()
    len_stem.clear()
    clean_text_lemm.clear()
    len_lemm.clear()
    pos_tag.clear()
    return data

@task
def document_vector(doc, keyed_vectors):
    # Remove out-of-vocabulary words. Create document vectors by averaging word vectors
    vocab_tokens = [word for word in doc if word in keyed_vectors.index_to_key]
    return np.mean(keyed_vectors.__getitem__(vocab_tokens), axis=0)

@task
def get_smote(inputs: pd.DataFrame,outputs: pd.DataFrame) -> Any:
    smote = SMOTE()
    x_smote,y_smote = smote.fit_resample(inputs,outputs)
    return {'x':x_smote,'y':y_smote}

@task
def split_data(input_: pd.DataFrame, output_: pd.Series, test_data_ratio: float) -> Dict[str, Any]:
    X_tr, X_te, y_tr, y_te = train_test_split(input_, output_, test_size=test_data_ratio, random_state=0)
    return {'X_TRAIN': X_tr, 'Y_TRAIN': y_tr, 'X_TEST': X_te, 'Y_TEST': y_te}

@task
def get_scaler(data: pd.DataFrame) -> Any:
    # scaling the numerical features
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler

@task
def rescale_data(data: pd.DataFrame, scaler: Any) -> pd.DataFrame:    
    # scaling the numerical features
    # column names are (annoyingly) lost after Scaling
    # (i.e. the DataFrame is converted to a numpy ndarray)
    data_rescaled = pd.DataFrame(scaler.transform(data), 
                                columns = data.columns, 
                                index = data.index)
    return data_rescaled

@task
def find_best_model(X_train: pd.DataFrame, y_train: pd.Series, estimator: Any, parameters: Dict) -> Any:
    # Enabling automatic MLflow logging for scikit-learn runs
    mlflow.sklearn.autolog(max_tuning_runs=None)

    with mlflow.start_run():        
        clf = GridSearchCV(
            estimator=estimator, 
            param_grid=parameters, 
            scoring='accuracy',
            cv=5,
            return_train_score=True,
            verbose=1
        )
        clf.fit(X_train, y_train)
        
        # Disabling autologging
        mlflow.sklearn.autolog(disable=True)
        
        return clf

@flow
def main(path:str):
    
    mlflow.set_tracking_uri("sqlite:///quora.db")
    mlflow.set_experiment("Quora Question Pair Similarity")
    
    # Define Parameters
    TARGET_COL = 'is_duplicate'
    TEST_data_RATIO = 0.2
    data_PATH = path
    SAFE_DIV = 0.0001
    UNWANTED_COL = ['id', 'qid1', 'qid2', 'log_word_share']  

    # Download necessary nltk packs
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    wv = api.load('glove-twitter-50')
    dump(wv,open('./embeddings/wv.pkl','wb'))
    wv=load(open('./embeddings/wv.pkl','rb'))    

    # Load the data
    data = load_data(path=data_PATH)
    
    # Null values 
    data = remove_null(data)
    
    # Applying preprocess
    data["question1"] = data["question1"].apply(preprocess)
    data["question2"] = data["question2"].apply(preprocess)
    
    # Basic Feature extraction
    data = feature_engineering(data)

    # Applying the tokenization, lemmitization, stemming to the preprocess_question1 column
    data['question1'].apply(lambda x: text_preprocess(x))
    data = add_preprocessed_data1(data)
    data['question2'].apply(lambda x: text_preprocess(x))
    data = add_preprocessed_data2(data)

    # applying the pretrained glove (glove-twitter-50) on  preproc_ques1_token column and preproc_ques2_lemm column
    data['doc_vec_glove_q1'] = data.preproc_ques1_token.apply(lambda x : document_vector(x, wv))
    data['doc_vec_glove_q1'] = data.preproc_ques2_token.apply(lambda x : document_vector(x, wv))
    
    # Identify Target Variable
    numeric_data = data.select_dtypes(include=[np.number])
    numeric_data = numeric_data.drop(UNWANTED_COL, axis = 1)
    inputs = numeric_data.drop([TARGET_COL], axis = 1)
    outputs = numeric_data[TARGET_COL]
    
    # Smote analysis on data
    smote_data = get_smote(inputs, outputs)

    # Split the data into Train and Test
    train_test_dict = split_data(smote_data['x'], smote_data['y'], test_data_ratio=TEST_data_RATIO)

    # Rescaling Train and Test data
    scaler = get_scaler(train_test_dict['X_TRAIN'])
    scaled_train_data = rescale_data(data=train_test_dict['X_TRAIN'], scaler=scaler)
    scaled_test_data = rescale_data(data=train_test_dict['X_TEST'], scaler=scaler)
        
    # Creating a model
    estimator = RandomForestClassifier(random_state=42)
    param_grid = [{'n_estimators': [200, 500],'max_features': ['sqrt', 'log2'],'max_depth' : [4,5,6,7,8],'criterion' :['gini', 'entropy']}]
    
    classifier = find_best_model(X_train=scaled_train_data, y_train=train_test_dict['Y_TRAIN'] ,estimator=estimator , parameters=param_grid)
    print(classifier.best_params_)
    print(classifier.score(scaled_test_data, train_test_dict['Y_TEST']))

# Run the main function
main(path='./data/train.csv')