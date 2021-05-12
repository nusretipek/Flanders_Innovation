# Import Packages
from IPython.display import clear_output
import pandas as pd
import requests
from requests.utils import requote_uri
from fake_useragent import UserAgent
from lxml import html
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
from textblob import TextBlob
from langdetect import detect
import re
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pickle
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix ,accuracy_score, classification_report
from sklearn.naive_bayes import BernoulliNB, GaussianNB, ComplementNB
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
import gensim
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

def fix_http(URL):
    if URL != '':
        if ('http' in URL) & (URL[-1:] == '/'):
            return URL
        elif ('http' in URL) & (URL[-1:] != '/'):
            return URL + '/'
        elif ('http' not in URL) & (URL[-1:] == '/'):
            return 'http://' + URL
        else:
            return 'http://' + URL + '/'

ua = UserAgent()

def get_html(URL, Timeout):
    header = {'User-Agent': str(ua.random)}
    try:
        page = requests.get(URL, timeout=Timeout, headers=header)
    except:
        return None
    return page.text

def get_html_ssl(URL, Timeout):
    header = {'User-Agent': str(ua.random)}
    try:
        page = requests.get(URL, timeout=Timeout, headers=header, verify=False)
    except:
        return None
    return page.text

def scrape_urls(CSV_file, URL_column, clean=False, output_file=None):
    requests.packages.urllib3.disable_warnings()
    tqdm(disable=True, total=0)
    if len(tqdm._instances) > 0:
        while len(tqdm._instances) > 0:
            tqdm._instances.pop().close()
        clear_output(wait=True)
    df = pd.read_csv(CSV_file)
    if (clean):
        df = df[pd.isnull(df[URL_column]) != True]
    html_texts = []
    for url in tqdm(df[URL_column].tolist(), total=len(df[URL_column])):
        text = get_html(fix_http(url), 10)
        if(text is None):
            text = get_html_ssl(fix_http(url), 10)
        html_texts.append(text)
    df['Raw_HTML'] = html_texts
    if (output_file is None):
        df.to_csv("scraped_html_file.csv", index=False)
    else:
        df.to_csv(output_file, index=False)
    return "Successfully Completed!"

def visible_texts(soup):
    re_spaces = re.compile(r'\s{3,}')
    text = ' '.join([s for s in soup.strings if s.parent.name not in ('style', 'script', 'head', 'title')])
    return re_spaces.sub(' ', text)

def language_detector(page):
    try:
        soup = BeautifulSoup(page, 'html.parser')
        for tag in soup.find_all('div', id=re.compile(r'(cook)|(popup)')):
            tag.decompose()
        for tag in soup.find_all('div', class_=re.compile(r'(cook)|(popup)')):
            tag.decompose()
        body_text = visible_texts(BeautifulSoup(visible_texts(soup), 'html.parser'))
        if len(soup.find_all('frame')) > 0:
            frame_text = ''
            for f in soup.find_all('frame'):
                frame_request = requests.get(f['src'])
                frame_soup =  BeautifulSoup(frame_request.content, 'html.parser')
                frame_text = frame_text + ' ' + visible_texts(BeautifulSoup(visible_texts(frame_soup), 'html.parser'))
            body_text = body_text + frame_text
        return detect(body_text)
    except:
        return 'unknown'

def detect_language(CSV_file = 'scraped_html_file.csv', HTML_column = 'Raw_HTML'):
    tqdm(disable=True, total=0)
    if len(tqdm._instances) > 0:
        while len(tqdm._instances) > 0:
            tqdm._instances.pop().close()
        clear_output(wait=True)
    df = pd.read_csv(CSV_file)
    languages = []
    counter = 0
    for html in tqdm(df[HTML_column].tolist(), total=len(df[HTML_column])):
        languages.append(language_detector(html))
        counter += 1
        if(counter % 100 == 0):
            time.sleep(30)
    df['Languages'] = languages
    df.to_csv(CSV_file, index=False)
    return "Successfully Completed!"

def language_detector2(URL):
    try:
        page = requests.get(URL, timeout=10)
        soup = BeautifulSoup(page.content, 'html.parser')
        for tag in soup.find_all('div', id=re.compile(r'(cook)|(popup)')):
            tag.decompose()
        for tag in soup.find_all('div', class_=re.compile(r'(cook)|(popup)')):
            tag.decompose()
        body_text = visible_texts(BeautifulSoup(visible_texts(soup), 'html.parser'))
        if len(soup.find_all('frame')) > 0:
            frame_text = ''
            for f in soup.find_all('frame'):
                frame_request = requests.get(f['src'])
                frame_soup =  BeautifulSoup(frame_request.content, 'html.parser')
                frame_text = frame_text + ' ' + visible_texts(BeautifulSoup(visible_texts(frame_soup), 'html.parser'))
            body_text = body_text + frame_text
        return len(body_text.split()), detect(body_text)
    except:
        return 0, 'unknown'

def language_switcher(URL, lang_code):
    success_boolean = False
    try:
        page = requests.get(URL)
    except:
        return success_boolean, ''
    soup = BeautifulSoup(page.text, 'html.parser')
    returned_list = soup.find_all(hreflang=re.compile(lang_code), href=True)
    if (len(returned_list) == 0):
        returned_list = soup.find_all(href=True)
        for item in returned_list:
            lower_string = str(item.text).lower()
            if (any(['nl' == word for word in lower_string.split()])):
                success_boolean = True
                new_page = item['href']
                if ('http' not in item['href']):
                    new_page = URL + item['href'].strip('.')
                if language_detector2(new_page)[1] == 'nl':
                    return success_boolean, new_page
        for item in returned_list:
            lower_string = str(item['href']).lower()
            if (lower_string.find('nl') != -1):
                success_boolean = True
                new_page = item['href']
                if ('http' not in item['href']):
                    new_page = URL + item['href'].strip('.')
                if language_detector2(new_page)[1] == 'nl':
                    return success_boolean, new_page
        return success_boolean, ''
    elif (len(returned_list) == 1):
        success_boolean = True
        new_page = returned_list[0]['href']
        if ('http' not in returned_list[0]['href']):
            new_page = URL + returned_list[0]['href'].strip('.')
        if language_detector2(new_page)[1] == 'nl':
            return success_boolean, new_page
    elif (len(returned_list) > 1):
        success_boolean = True
        for item in returned_list:
            new_page = item['href']
            if (item['href'].find('be') != -1):
                if ('http' not in item['href']):
                    new_page = URL + item['href'].strip('.')
                if language_detector2(new_page)[1] == 'nl':
                    return success_boolean, new_page
        new_page = returned_list[0]['href']
        if ('http' not in returned_list[0]['href']):
            new_page = URL + returned_list[0]['href'].strip('.')
        if language_detector2(new_page)[1] == 'nl':
            return success_boolean, new_page
    else:
        return success_boolean, ''

def crawl_contact_page(URL, Base_URL, request_page):
    new_pages = []
    soup_crawl = BeautifulSoup(request_page.text, 'html.parser')
    returned_list = soup_crawl.find_all(href=True)
    for item in returned_list:
        lower_href_text = ''.join(str(item.text).lower().strip())
        if ('cont' in lower_href_text):
            if ('www' in item['href']):
                new_pages.append(item['href'])
            else:
                new_page = Base_URL + item['href'].strip('.')
                new_pages.append(new_page)
    return list(set(new_pages))

def crawl_location_page(URL, Base_URL, request_page):
    new_pages = []
    soup_crawl = BeautifulSoup(request_page.text, 'html.parser')
    returned_list = soup_crawl.find_all(href=True)
    for item in returned_list:
        lower_href_text = ''.join(str(item.text).lower().strip())
        if (('vest' in lower_href_text) | ('loc' in lower_href_text)):
            if ('www' in item['href']):
                new_pages.append(item['href'])
            else:
                new_page = Base_URL + item['href'].strip('.')
                new_pages.append(new_page)
    return list(set(new_pages))

def validate_zip(URL, Base_URL, zip_1, zip_2):
    page = requests.get(URL)
    contact_pages = crawl_contact_page(URL, Base_URL, page)
    location_pages = crawl_location_page(URL, Base_URL, page)
    total_pages = contact_pages + location_pages
    print(total_pages)
    soup = BeautifulSoup(page.text, 'lxml')
    [s.decompose() for s in soup('script')]
    all_text = ' '.join(re.sub(r'\n', ' ', soup.get_text()).split())
    numeric_text = re.findall(r'\d+', all_text)
    if (any([str(zip_1) == number for number in numeric_text]) |
            any([str(zip_2) == number for number in numeric_text])):
        return True
    elif (len(total_pages) != 0):
        for new_page in total_pages:
            time.sleep(3)
            page = requests.get(new_page)
            soup = BeautifulSoup(page.text, 'lxml')
            [s.decompose() for s in soup('script')]
            all_text = ' '.join(re.sub(r'\n', ' ', soup.get_text()).split())
            numeric_text = re.findall(r'\d+', all_text)
            if (any([str(zip_1) == number for number in numeric_text]) |
                    any([str(zip_2) == number for number in numeric_text])):
                return True
    return False

def validate_street(URL, Base_URL, street_raw):
    page = requests.get(URL)
    contact_pages = crawl_contact_page(URL, Base_URL, page)
    location_pages = crawl_location_page(URL, Base_URL, page)
    total_pages = contact_pages + location_pages
    print(total_pages)
    soup = BeautifulSoup(page.text, 'lxml')
    [s.decompose() for s in soup('script')]
    all_text = ' '.join(re.sub(r'\n', ' ', soup.get_text()).split())
    street_raw_temp = re.sub(r'\d+', '', street_raw).strip()
    final_street = re.sub('[\(\[].*?[\)\]]', '', street_raw_temp)
    if (final_street in all_text):
        return True
    elif (len(total_pages) != 0):
        for new_page in total_pages:
            time.sleep(3)
            page = requests.get(new_page)
            soup = BeautifulSoup(page.text, 'lxml')
            [s.decompose() for s in soup('script')]
            all_text = ' '.join(re.sub(r'\n', ' ', soup.get_text()).split())
            if (final_street in all_text):
                return True
    return False

def extract_url_from_email(Email):
    try:
        return (re.findall(r'@([A-Za-z.]+)', Email)[0]).strip()
    except:
        return ''

# Input is 4 columns; cur_email,cur_web,email,web columns
def assign_primary_URL(cur_web, cur_email, web, email):
    if not (pd.isnull(cur_web)):
        return fix_http(cur_web)
    elif not (pd.isnull(cur_email)):
        return fix_http(extract_url_from_email(cur_email))
    elif not (pd.isnull(web)):
        return fix_http(web)
    elif not (pd.isnull(email)):
        return fix_http(extract_url_from_email(email))
    else:
        return ''

def get_status_code(URL):
    try:
        return requests.get(URL, timeout=10).status_code
    except:
        return 0

def get_NL_URL(URL, status_code):
    try:
        if status_code == 200:
            if language_detector(URL)[1] != 'nl':
                success_code, new_url = language_switcher(URL, 'nl')
                if success_code & (new_url != ''):
                    return new_url
        return URL
    except:
        return URL

def switch_language(CSV_file = 'scraped_html_file.csv', language_column = 'Languages', URL_column = 'URL'):
    requests.packages.urllib3.disable_warnings()
    df = pd.read_csv(CSV_file)
    tqdm(disable=True, total=0)
    if len(tqdm._instances) > 0:
        while len(tqdm._instances) > 0:
            tqdm._instances.pop().close()
        clear_output(wait=True)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if((df.loc[index, language_column] not in ['nl', 'en']) and (pd.isnull(df.loc[index, 'Raw_HTML']) == False)):
            if (language_switcher(df.loc[index, URL_column], 'nl') is not None):
                success_code, new_url = language_switcher(df.loc[index, URL_column], 'nl')
                if success_code & (new_url != ''):
                    df.loc[index, URL_column] = new_url
                    df.loc[index, 'Raw_HTML'] = get_html_ssl(fix_http(df.loc[index, URL_column]), 10)
                    df.loc[index, language_column] = language_detector(df.loc[index, 'Raw_HTML'])
    df.to_csv(CSV_file, index=False)
    return "Successfully Completed!"

def filter_get_language_distribution(CSV_file = 'scraped_html_file.csv', Language_column = 'Languages'):
    df = pd.read_csv(CSV_file)
    print(df.Languages.value_counts())
    df = df[df[Language_column].isin(['nl', 'en'])]
    df.to_csv(CSV_file, index=False)
    return "Successfully Completed!"

# Clean Text + Get if any other frames
def clean_pop_cookie_frame(raw_text):
    soup = BeautifulSoup(raw_text, 'html.parser')
    for tag in soup.find_all('div', id=re.compile(r'(cook)|(popup)')):
        tag.decompose()
    for tag in soup.find_all('div', class_=re.compile(r'(cook)|(popup)')):
        tag.decompose()
    body_text = visible_texts(BeautifulSoup(visible_texts(soup), 'html.parser'))
    if len(soup.find_all('frame')) > 0:
        frame_text = ''
        for f in soup.find_all('frame'):
            try:
                frame_request = requests.get(f['src'], timeout=10)
                frame_soup = BeautifulSoup(frame_request.content, 'html.parser')
                frame_text = frame_text + ' ' + visible_texts(BeautifulSoup(visible_texts(frame_soup), 'html.parser'))
            except:
                frame_text = ''
        body_text = body_text + frame_text
    return body_text.strip()

def lower_punct_number_clean(text, lower_bound_letter_length):
    temp_text = re.sub('[^A-Za-z ]+', '', text)
    temp_text = ' '.join([i for i in temp_text.split() if len(i) >= lower_bound_letter_length])
    return temp_text.lower().strip()

english_stopwords = stopwords.words('english')
dutch_stopwords = stopwords.words('dutch')

def remove_stopwords(text, lang):
    if (lang == 'nl'):
        temp_text = ' '.join([word for word in text.split() if word not in dutch_stopwords])
        return ' '.join([word for word in temp_text.split() if word not in english_stopwords])
    elif (lang == 'en'):
        return ' '.join([word for word in text.split() if word not in english_stopwords])
    else:
        return None

english_stemmer = SnowballStemmer(language='english')
dutch_stemmer = SnowballStemmer(language='dutch')

def stem_text(text, lang):
    if (text == None):
        return None
    elif (lang == 'nl'):
        return ' '.join([dutch_stemmer.stem(word) for word in text.split()])
    elif (lang == 'en'):
        return ' '.join([english_stemmer.stem(word) for word in text.split()])
    else:
        return None

def count_words(text):
    if (text == None):
        return None
    else:
        return len(text.split())

def HTML_to_text(CSV_file = 'scraped_html_file.csv', HTML_column = 'Raw_HTML'):
    df = pd.read_csv(CSV_file)
    tqdm(disable=True, total=0)
    if len(tqdm._instances) > 0:
        while len(tqdm._instances) > 0:
            tqdm._instances.pop().close()
        clear_output(wait=True)
    df['Clean_Text'] = ''
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        df.loc[index, 'Clean_Text'] = lower_punct_number_clean(clean_pop_cookie_frame(df.loc[index, HTML_column]), 3)
        df.loc[index, 'Clean_Text'] = remove_stopwords(df.loc[index, 'Clean_Text'], df.loc[index, 'Languages'])
    df['word_count'] = df.Clean_Text.apply(lambda z: count_words(z))
    df = df[df['word_count'] > 20]
    df = df.drop('word_count', axis=1)
    df.to_csv(CSV_file, index=False)
    return "Successfully Completed!"

def csv_to_dataframe(CSV_file = 'scraped_html_file.csv', key_column = 'Ondernemingsnummer', text_column='Clean_Text', label_column=None, language = False):
    df = pd.read_csv(CSV_file)
    if(label_column is None):
        #print("Warning: No label data Entered, unsupervised learning techniques are unavailable; only Prediction!")
        if(language):
            return pd.DataFrame({'key': df[key_column], 'text': df[text_column], 'Languages': df['Languages']})
        else:
            return pd.DataFrame({'key': df[key_column], 'text': df[text_column]})
    else:
        if(language):
            return pd.DataFrame({'key': df[key_column], 'text': df[text_column], 'label': df[label_column], 'Languages': df['Languages']})
        else:
            return pd.DataFrame({'key': df[key_column],'text': df[text_column],'label': df[label_column]})

def train_doc_embeddings(dataset, size, iteration, window, min_count):
    def tokenize_text(text):
        tokens = []
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                if len(word) < 2:
                    continue
                tokens.append(word.lower())
        return tokens
    dataset['inno'] = dataset.label.apply(lambda x: 'innovative' if x == 1 else 'not innovative')
    tagged_documents = dataset.apply(lambda x: TaggedDocument(words=tokenize_text(x['text']), tags=str(random.randint(0,1000000))), axis=1)
    model = Doc2Vec(tagged_documents, vector_size=size, window=window, min_count=min_count, workers=4, epochs = iteration)
    d2v_list = []
    for i in range(len(tagged_documents)):
        d2v_list.append(np.array(model.infer_vector(tagged_documents[i].words)).tolist())
    dataset['d2v_list'] = d2v_list
    dataset = dataset.drop('inno', axis = 1)
    return dataset

def oversample_category_d2v(dataset, innovation_category, total_size, seed_value, embeddings = False, language = False):
    dataset = dataset.reset_index()
    index_list = list(dataset[dataset['label'] == innovation_category].index)
    np.random.seed(seed_value)
    choices = np.random.choice(index_list, size=total_size-len(index_list), replace=True)
    key = []
    text = []
    innovation = []
    embeddings_list = []
    language_list = []
    for i in choices:
        key.append(dataset.iloc[i]['key'])
        text.append(dataset.iloc[i]['text'])
        innovation.append(dataset.iloc[i]['label'])
        if(embeddings):
            embeddings_list.append(dataset.iloc[i]['d2v_list'])
        if(language):
            language_list.append(dataset.iloc[i]['Languages'])
    if(embeddings and not language):
        df = pd.DataFrame({'key': key,'text': text, 'label': innovation, 'd2v_list': embeddings_list})
    elif (not embeddings and language):
        df = pd.DataFrame({'key': key,'text': text, 'label': innovation, 'Languages': language_list})
    else:
        df = pd.DataFrame({'key': key,'text': text, 'label': innovation, 'Languages': language_list, 'd2v_list': embeddings_list})
    temp_df = pd.concat([dataset, df])
    return temp_df.reset_index()

def TFIDF_prebuild_sets_d2v(dataset, seed, fraction, feature_count, oversample = False, size = 0, label_oversample = 1, embeddings = False,
                            language = False, d2v = False, verbose = True):
    if(oversample):
        dataset = oversample_category_d2v(dataset, label_oversample, size, 22494, d2v, language)
    TFIDF_vectorizer =TfidfVectorizer(max_features = feature_count)
    full_TFIDF = TFIDF_vectorizer.fit_transform(dataset['text'])
    temp_df = pd.DataFrame(full_TFIDF.toarray(), columns = TFIDF_vectorizer.get_feature_names())
    temp_df.insert(loc=0, column = 'label', value = dataset['label'])
    temp_df.insert(loc=1, column = 'key', value = dataset['key'])
    if(embeddings):
        temp_df.insert(loc=2, column = 'doc_vector', value = dataset['doc_vector'])
    if(language):
        temp_df.insert(loc=3, column = 'Languages', value = dataset['Languages'])
        temp_df['Languages'] = temp_df.Languages.apply(lambda z: 1 if (z == 'en') else 0)
    if(d2v):
        frame = dataset['d2v_list'].apply(pd.Series)
        temp_df = pd.merge(temp_df, frame, left_index=True, right_index=True)
    temp_df = temp_df.sample(frac=1, random_state = seed).drop_duplicates(['key'])
    temp_df = temp_df.drop('key', axis = 1)
    temp_df = temp_df.fillna(0)
    train_temp = temp_df.sample(frac = fraction, random_state = seed)
    test_temp = temp_df.drop(train_temp.index)
    train_predictors = train_temp.drop('label', axis = 1)
    test_redictors = test_temp.drop('label', axis = 1)
    if(verbose):
        print(TFIDF_vectorizer.get_feature_names())
    return train_predictors, train_temp['label'], test_redictors, test_temp['label']

def train_model(clf, Train_X, train_y, Test_X, test_y, save_model = False, model_name=None):
    model = clf
    model.fit(Train_X, train_y)
    score = model.score(Train_X, train_y)
    train_pred = model.predict_proba(Train_X)[:,1]
    test_pred = model.predict_proba(Test_X)[:,1]
    print("Training Accuracy: ".ljust(20), accuracy_score(train_y, model.predict(Train_X)))
    print("Test Accuracy: ".ljust(20), accuracy_score(test_y, model.predict(Test_X)))
    print("Classification Report of the Test Set: \n ")
    print(classification_report(test_y, model.predict(Test_X)))
    #Plot the predicted probabilities from the model - Train Set
    fig,ax = plt.subplots(1,2,figsize=(18,6))
    #Plot 1
    ax[0].hist(train_pred[train_y==0], bins=50, label='Label = 0', alpha=0.8)
    ax[0].hist(train_pred[train_y==1], bins=50, label='Label = 1', alpha=0.8, color='g')
    ax[0].set_title('Train Dataset Predicted Probabilities Histogram', fontsize=15)
    ax[0].set_xlabel('Label Probability', fontsize=15)
    ax[0].set_ylabel('Count', fontsize=15)
    ax[0].legend(fontsize=15)
    ax[0].tick_params(axis='both', labelsize=15, pad=5)
    #Plot 2
    ax[1].hist(test_pred[test_y==0], bins=50, label='Label = 0', alpha=0.8)
    ax[1].hist(test_pred[test_y==1], bins=50, label='Label = 1', alpha=0.8, color='g')
    ax[1].set_title('Test Dataset Predicted Probabilities Histogram', fontsize=15)
    ax[1].set_xlabel('Label Probability', fontsize=15)
    ax[1].set_ylabel('Count', fontsize=15)
    ax[1].legend(fontsize=15)
    ax[1].tick_params(axis='both', labelsize=15, pad=5)
    plt.show()
    if(save_model):
        if(model_name is None):
            pickle.dump(model, open('saved_model.pkl', 'wb'))
        else:
            pickle.dump(model, open(model_name, 'wb'))

def train_doc_embeddings_prediction(dataset, size, iteration, window, min_count):
    def tokenize_text(text):
        tokens = []
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                if len(word) < 2:
                    continue
                tokens.append(word.lower())
        return tokens
    tagged_documents = dataset.apply(lambda x: TaggedDocument(words=tokenize_text(x['text']), tags='innovative'), axis=1)
    model = Doc2Vec(tagged_documents, vector_size=size, window=window, min_count=min_count, workers=4, epochs = iteration)
    d2v_list = []
    for i in range(len(tagged_documents)):
        d2v_list.append(np.array(model.infer_vector(tagged_documents[i].words)).tolist())
    dataset['d2v_list'] = d2v_list
    return dataset

def make_prediction(saved_model, dataset_name = None, language = True, verbose = False,):
    dataset = csv_to_dataframe(dataset_name, language=language)
    dataset['Languages'] = dataset.Languages.apply(lambda z: 1 if (z == 'en') else 0)
    dataset = train_doc_embeddings_prediction(dataset,50,5,10,3)
    TFIDF_vectorizer =TfidfVectorizer(max_features = 300)
    full_TFIDF = TFIDF_vectorizer.fit_transform(dataset['text'])
    temp_df = pd.DataFrame(full_TFIDF.toarray(), columns = TFIDF_vectorizer.get_feature_names())
    temp_df.insert(loc=1, column = 'key', value = dataset['key'])
    temp_df.insert(loc=1, column = 'Languages', value = dataset['Languages'])
    frame = dataset['d2v_list'].apply(pd.Series)
    temp_df = pd.merge(temp_df, frame, left_index=True, right_index=True)
    temp_df = temp_df.sample(frac=1, random_state = 22494).drop_duplicates(['key'])
    temp_df = temp_df.fillna(0)
    if(verbose):
        print(TFIDF_vectorizer.get_feature_names())
    loaded_model = pickle.load(open(saved_model, 'rb'))
    pred = loaded_model.predict_proba(temp_df.drop('key', axis = 1))[:, 1]
    temp_df['predictions_proba'] = pred
    temp_df['predictions'] = temp_df.predictions_proba.apply(lambda z: 1 if(z>0.5) else 0)
    temp_df = temp_df[['key', 'predictions_proba', 'predictions']]
    return temp_df

def get_model_coefficients(saved_model, Train_X, size = 20, mask = True):
	loaded_model = pickle.load(open(saved_model, 'rb'))
	coefs = pd.DataFrame(zip(Train_X.columns, np.transpose(loaded_model.coef_.tolist()[0])), columns=['features', 'coef'])
	if(mask):
		coefs = coefs[coefs['features'] != 'doc_vector']
		coefs['coef_bool'] = coefs.features.apply(lambda z: 1 if (str(z).isdigit()) else 0)
		coefs = coefs[coefs['coef_bool'] != 1]
		coefs = coefs.drop('coef_bool', axis = 1)
	negatives = coefs.sort_values(by=['coef'], ascending=True).head(size).reset_index(drop=True)
	positives = coefs.sort_values(by=['coef'], ascending=False).head(size).reset_index(drop=True)
	table_f = negatives.merge(positives, right_index=True, left_index=True)
	table_f.columns = ['Negative Words', 'Coefficient', 'Positive Words', 'Coefficient']
	return table_f