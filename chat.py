import pandas as pd
import boto3
import json
from nltk.tokenize import word_tokenize
import os
import collections
from symspellpy import SymSpell, Verbosity
from itertools import chain
import inflect
from six import string_types
import requests
from itertools import combinations
import uuid
import nltk
from flask_wtf import Form
from wtforms import TextField, IntegerField, TextAreaField, SubmitField, RadioField,SelectField
from wtforms import validators, ValidationError
from flask import Flask, render_template, request, flash , url_for , redirect , session
import requests
import pymysql
import uuid
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet_ic')
nltk.download('wordnet')
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

brown_ic = wordnet_ic.ic('ic-brown.dat')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')
engine = inflect.engine()
app = Flask(__name__,template_folder='/home/ubuntu/ubuntu/old_machine/template/directory')
app.config.update({
    'SECRET_KEY': 'keyboard kittens',
})
lex_client = boto3.client('lex-runtime', region_name='us-east-1')
comprehend = boto3.client(service_name='comprehend', region_name='us-west-2')

class GetDataForm(Form):
    sentence = TextField("Sentence",[validators.Required("Please enter your query.")])
    submit = SubmitField("Send")
class GetDataForm_2(Form):
    sentence2 = TextAreaField("Submitted Category")
    submit = SubmitField("Send")
    submit2 = SubmitField("Home")
def containsNonAscii(s):
    return any(ord(i)>127 for i in s)
def convert(data):
    if isinstance(data, string_types):
        return str(data)
    elif isinstance(data, collections.Mapping):
        return dict(map(convert, data.items()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert, data))
    else:
        return data
def get_concept_net(inp_str):
    product = []
    slots = get_slots(inp_str,lex_client)
    if slots is not None:
        product.append([value for key, value in slots.items() if key in ['product','slotOne','product_s','product_d']])
    product_2 = list(chain.from_iterable(product))
    product_2 =[x for x in product_2 if x is not None]
    product_2_p = [engine.plural(x) for x in product_2]
    product_2 = product_2 + product_2_p
    syntax_dict = get_syntax(inp_str,comprehend)
    syntax_dict = convert(syntax_dict)
    nouns = [key for key, value in syntax_dict.items() if value == 'NOUN']
    nouns_p = [engine.plural(x) for x in nouns]
    nouns = nouns + nouns_p
    tokens = word_tokenize(inp_str)
    tokens_p = [engine.plural(x) for x in tokens]
    tokens = tokens+tokens_p
    labels = []
    labels2 = {key:[] for key in labels}
    no_pro = []
    if len(product_2)==0 & len(nouns)==0:
        no_pro = tokens
    for i in tokens:
        if i in product_2 or i in nouns or i in no_pro:
            response = requests.get('http://api.conceptnet.io/c/en/'+i+'?offset=0&limit=1000')
            obj = response.json()
            for edge in obj['edges']:
                if edge['rel']['label'] not in labels:
                    labels.append(edge['rel']['label'])
                if edge['rel']['label'] not in list(labels2.keys()):
                    labels2[edge['rel']['label']] = []
            for edge in obj['edges']:
                if (edge['start']['label'] not in labels2[edge['rel']['label']]) & (not containsNonAscii(edge['start']['label'])):
                    labels2[edge['rel']['label']].append(edge['start']['label'])
                if (edge['end']['label'] not in labels2[edge['rel']['label']]) & (not containsNonAscii(edge['end']['label'])):
                    labels2[edge['rel']['label']].append(edge['end']['label'])
    return labels2['RelatedTo']
def get_concept_net_service(inp_str):
    bigrams = ["_".join(a) for a in combinations(word_tokenize(inp_str), 2)]
    labels = []
    labels2 = {key:[] for key in labels}
    for gram in bigrams:
        response = requests.get('http://api.conceptnet.io/c/en/'+gram+'?offset=0&limit=1000')
        obj = response.json()
        for edge in obj['edges']:
            if edge['rel']['label'] not in labels:
                    labels.append(edge['rel']['label'])
    
        for edge in obj['edges']:
            if edge['rel']['label'] not in labels:
                labels.append(edge['rel']['label'])
            if edge['rel']['label'] not in list(labels2.keys()):
                labels2[edge['rel']['label']] = []
        for edge in obj['edges']:
            if (edge['start']['label'] not in labels2[edge['rel']['label']]) & (not containsNonAscii(edge['start']['label'])):
                labels2[edge['rel']['label']].append(edge['start']['label'])
    if 'CapableOf' in labels:
        return labels2['CapableOf']
    else:
        return ''
def get_slots(sent,lex_client):
    json_data = json.dumps(boto3.client('lex-runtime', region_name='us-east-1').post_text(botName='Get_Services',botAlias='Test',userId=uuid.uuid4().hex,inputText=sent), sort_keys=True, indent=4)
    parsed_json = json.loads(json_data)
    pos_dict = {}
    if('slots' in parsed_json.keys()):
        if(not all(v is None for v in parsed_json['slots'].values())):
            pos_dict = convert(parsed_json['slots'].copy())
        return pos_dict

def get_syntax(sent,comprehend):
    json_data = json.dumps(comprehend.detect_syntax(Text=sent, LanguageCode='en'), sort_keys=True, indent=4)
    pos_dict = {}
    parsed_json = json.loads(json_data)  
    if(len(parsed_json['SyntaxTokens']) > 0):
        for i in range(0,len(parsed_json['SyntaxTokens'])):
            value = parsed_json['SyntaxTokens'][i]['PartOfSpeech']['Tag']
            key = parsed_json['SyntaxTokens'][i]['Text']
            pos_dict[key] = value

    return pos_dict
def abv_resolver(string):
    out=''
    if(string in list(dict_df['Abbv_term'])):
        out=dict_df[dict_df['Abbv_term']==string]['Full_term'].values[0]
    else:
        out=string
    return out
def get_entities(sent,comprehend):
    json_data = json.dumps(comprehend.detect_entities(Text=sent, LanguageCode='en'), sort_keys=True, indent=4)
    entities_dict = {}
    parsed_json = json.loads(json_data)
    if(len(parsed_json['Entities']) > 0):
        parsed_json['Entities'][0]['Type']
        for i in range(0,len(parsed_json['Entities'])):
            key = parsed_json['Entities'][i]['Text']
            value = parsed_json['Entities'][i]['Type']
            entities_dict[key] = value
    return entities_dict
def resolver(inp_str):
    product = []
    company = []
    location = []
    slots = get_slots(inp_str,lex_client)
    named_entities = get_entities(inp_str,comprehend)
    #print(slots)
    if slots is not None:
        product.append([value for key, value in slots.items() if key in ['product','slotOne','product_s','product_d']])
        company.append([value for key, value in slots.items() if key in ['companies','companies_n']])
        location.append([value for key, value in slots.items() if key in ['location','city','locn','location_l','state','addrs_s','addrs_p']])
    if named_entities is not None:
        company.append([key for key, value in named_entities.items() if value in ['ORGANIZATION']])
    company_2 = list(chain.from_iterable(company))
    product_2 = list(chain.from_iterable(product))
    location_2 = list(chain.from_iterable(location))
    product_2 =[x for x in product_2 if x is not None]
    location_2 = [x for x in location_2 if x is not None]
    company_2 = [x for x in company_2 if x is not None]
    syntax_dict = get_syntax(inp_str,comprehend)
    syntax_dict = convert(syntax_dict)
    vbs = [key for key, value in syntax_dict.items() if value == 'VERB']
    Noun_Q1_Q2_P  = [engine.plural(x) for x in product_2 ]
    Noun_Q1_Q2_P  = [str(x) for x in Noun_Q1_Q2_P ]
    Noun_Q1_Q2 = product_2 + Noun_Q1_Q2_P
    Noun_Q1_Q2 = [word_tokenize(i) for i in Noun_Q1_Q2]
    Noun_Q1_Q2 = list(chain.from_iterable(Noun_Q1_Q2))
    match_df = df_classifieds[(df_classifieds['Qualifier'].isin(Noun_Q1_Q2)) | (df_classifieds['Noun'].isin(Noun_Q1_Q2)) | (df_classifieds['Qualifier2'].isin(Noun_Q1_Q2)) ]
    best_match = ''
    concept_net_related_service = []
    concept_net_related_syn_1 = ''
    concept_net_related_syn_2 = ''
    if 'fix' in word_tokenize(inp_str):
        inp_2 = inp_str.replace('fix','repair')
        differ_res = differentiator(inp_2)
    else:
        differ_res = differentiator(inp_str)
    if 'fix' in word_tokenize(inp_str):
        inp_2 = inp_str.replace('fix','repair')
        concept_net_related_syn_1 = get_concept_net_service(inp_2)
        inp_3 = inp_str.replace('fix','service')
        concept_net_related_syn_2 = get_concept_net_service(inp_3)
    if len(match_df) ==0:
        if differ_res=='Service':
            concept_net_related = get_concept_net(inp_str)
            #concept_net_related_service = get_concept_net_service(inp_str)
            #print(concept_net_related_service)
            match_df = df_classifieds[(df_classifieds['Qualifier'].isin(concept_net_related_service))|(df_classifieds['Qualifier'].isin(concept_net_related)) | (df_classifieds['Noun'].isin(concept_net_related_service)) | (df_classifieds['Noun'].isin(concept_net_related)) | (df_classifieds['Qualifier2'].isin(concept_net_related_service)) | (df_classifieds['Qualifier2'].isin(concept_net_related))]
        else:
            concept_net_related = get_concept_net(inp_str)
            match_df = df_classifieds[(df_classifieds['Qualifier'].isin(concept_net_related)) | (df_classifieds['Noun'].isin(concept_net_related)) | (df_classifieds['Qualifier2'].isin(concept_net_related)) ]
        h_score = 0
        for i in match_df['Sub-Category to fit SearchTempest']:
            score1 = sentence_similarity(' '.join(Noun_Q1_Q2),str(i))
            if h_score < score1:
                best_match = i
                h_score = score1
        #print(best_match)
    if best_match != '':
        match_df = match_df[(match_df['Sub-Category to fit SearchTempest'] == best_match)]
    
    if differentiator(inp_str)=='Service':
        match_df = match_df[(match_df['Category'].isin(['services','jobs','unknown','gigs',None]))]
    else:
        match_df = match_df[(~match_df['Category'].isin(['services','jobs','gigs']))]
    res_df = match_df[(match_df['Verb'].isin(vbs))]
    resultant_category = ''
    
    if(len(res_df)!= 0):
        if len(company) > 0 or len(location) > 0  :
            resultant_category = res_df['Category']+ " " +res_df['Sub-Category to fit SearchTempest'] + " " + ''.join(company_2) + " " + ''.join(location_2)
        else:
            resultant_category = res_df['Category']+ " " +res_df['Sub-Category to fit SearchTempest']

    else:
        
        if len(company) > 0 or len(location) > 0  :
            resultant_category = match_df['Category']+ " " +match_df['Sub-Category to fit SearchTempest'] + " " + ''.join(company_2) + " " + ''.join(location_2)
        else:
            resultant_category = match_df['Category']+ " " +match_df['Sub-Category to fit SearchTempest']
    if differ_res == 'Service':
        if len(resultant_category!=0):
            resultant_category = resultant_category +  'can be done by ' + ','.join(get_concept_net_service(inp_str))+','.join(concept_net_related_syn_1)+','.join(concept_net_related_syn_2)
        else:
            resultant_category = 'Can be done by ' + ','.join(get_concept_net_service(inp_str))+','.join(concept_net_related_syn_1)+','.join(concept_net_related_syn_2)
    return resultant_category
def differentiator(inp_str):
    bigrams = ["_".join(a) for a in combinations(word_tokenize(inp_str), 2)]
    labels = []
    for gram in bigrams:
        response = requests.get('http://api.conceptnet.io/c/en/'+gram+'?offset=0&limit=1000')
        obj = response.json()
        for edge in obj['edges']:
            if edge['rel']['label'] not in labels:
                    labels.append(edge['rel']['label'])
    #print(labels)
    sentences = ["buy","service","sale"]
    focus_sentence = inp_str
    buy_score = 0
    service_score =0
    syntax_dict = get_syntax(inp_str,comprehend)
    syntax_dict = convert(syntax_dict)
    vbs = [key for key, value in syntax_dict.items() if value == 'VERB']
    noun = [key for key, value in syntax_dict.items() if value == 'NOUN']
    vbs = vbs + noun
    if len(vbs) != 0:
        for i in vbs:
            for sentence in sentences:
                #print("Similarity(\"%s\", \"%s\") = %s" % (focus_sentence, sentence, sentence_similarity(focus_sentence, sentence)+sentence_similarity(sentence, focus_sentence)))
                if sentence == "buy" or sentence == "sale":
                    buy_score = buy_score + sentence_similarity(i, sentence)+sentence_similarity(sentence, i)
                else:
                    service_score = service_score + sentence_similarity(i, sentence)+sentence_similarity(sentence, i)
    
    if buy_score > service_score:
        return "Buy"
    else:
        return "Service"
    
    
    if 'CapableOf' in labels:
        if service_score > buy_score :
            return("Service")
        else:
            return("Purchase")
    else:
        if service_score > buy_score :
            return("Service")
        else:
            return("Purchase")

def get_key_phrases(sent,comprehend):
    json_data = json.dumps(comprehend.detect_key_phrases(Text=sent, LanguageCode='en'), sort_keys=True, indent=4)
    pos_dict = {}
    parsed_json = json.loads(json_data)
    if(len(parsed_json['KeyPhrases']) > 0):
        for i in range(0,len(parsed_json['KeyPhrases'])):
            key = parsed_json['KeyPhrases'][i]['Text']
            value = parsed_json['KeyPhrases'][i]['Score']
            pos_dict[key] = value

    return pos_dict
def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
 
    if tag.startswith('V'):
        return 'v'
 
    if tag.startswith('J'):
        return 'a'
 
    if tag.startswith('R'):
        return 'r'
 
    return None
 
def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
 
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None
def sentence_similarity(sentence1, sentence2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))
    #print(sentence1)
    #print(sentence2)
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]
    #print(synsets1)
    #print(synsets2)
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss ]
    synsets2 = [ss for ss in synsets2 if ss ]
    
    #print(synsets1)
    #print(synsets2)
    score, count = 0.0, 0
    best_score = 0
    # For each word in the first sentence
    if len(synsets1) != 0:
        for synset in synsets1:
            if len([x for x in  [synset.jcn_similarity(ss,ic=brown_ic) for ss in synsets2 if synset.pos() == ss.pos() and synset.pos() in ['n','v']] if x is not None])>0:
            # Get the similarity value of the most similar word in the other sentence
                best_score = max([x for x in  [synset.jcn_similarity(ss,ic=semcor_ic) for ss in synsets2 if synset.pos() == ss.pos() and synset.pos() in ['n','v']] if x is not None])
 
            # Check that the similarity could have been computed
            if best_score is not None:
                score += best_score
                count += 1
 
        # Average the values
        score /= count
    return score
    
pos_dict = {}
parsed_json = json.loads(open('old_machine/myslang.json').read())
Abbv_term = list(parsed_json.keys())
Full_term = list(parsed_json.values())
acronyms_df = pd.DataFrame({'Abbv_term':Abbv_term,'Full_term':Full_term})
acronyms_df_2 = pd.read_csv('old_machine/craiglist.csv',sep = '|',header = None,names = ['Abbv_term','Full_term'])
acronyms_df_2.loc[0,'Abbv_term'] = 'AFAIK'
acronyms_df_2 = acronyms_df_2[~acronyms_df_2['Abbv_term'].isin(acronyms_df['Abbv_term'])]
frames = [acronyms_df,acronyms_df_2]
dict_df = pd.concat(frames)
max_edit_distance_dictionary = 2
prefix_length = 7
sym_spell = SymSpell(max_edit_distance_dictionary, prefix_length)
dictionary_path = os.path.join("old_machine/en_50k.txt")
term_index = 0
count_index = 1
if not sym_spell.load_dictionary(dictionary_path, term_index, count_index):
    print("Dictionary file not found") 
df_classifieds = pd.read_csv('old_machine/comprehend/Classifieds.csv')

sent = ''
inf_cat = ''
def get_results(sent):
    inp_str = sent
    tokens = word_tokenize(inp_str)
    for i,tok in enumerate(tokens):
        resolved = abv_resolver(tok)
        tokens[i] = resolved
    inp_str = " ".join(tokens)
    max_edit_distance_lookup = 2
    suggestion_verbosity = Verbosity.CLOSEST
    suggestions = sym_spell.lookup_compound(inp_str,max_edit_distance_lookup)
    inp_str = suggestions[0].term
    named_entities = get_entities(inp_str,comprehend)
    slots = get_slots(inp_str,lex_client)
    res = resolver(inp_str)        
    return res
@app.route('/', methods = ['GET', 'POST'])
def contact():
    form = GetDataForm(request.form)
    
    if request.method == 'POST':
        sentence=request.form.get('sentence')
        sent = sentence
        if form.validate():
            max_edit_distance_lookup = 2
            suggestion_verbosity = Verbosity.CLOSEST
            suggestions = sym_spell.lookup_compound(sentence,max_edit_distance_lookup)
            inp_str_symspell = suggestions[0].term
            res = get_results(sentence)
            lex_res = get_slots(sentence,lex_client)
            comprehend_res = get_entities(sentence,comprehend)
            comprehend_syn = get_syntax(sentence,comprehend)
            concept_net_res = get_concept_net(sentence)
            if(res is not None and  len(res)>0):
                print('Hii')
                print(res)
                res = ''.join(res.iloc[0])
                session['inf_cat'] = res
                session['message'] = sent
                flash('Message:- ' + sent)
                flash('Inferred Category :- ' + res)
                flash("Lex:- "+str(lex_res))
                flash("Comprehend:- "+str(comprehend_res))
                flash("SymSpell:- "+str(inp_str_symspell))
                flash(comprehend_syn)
                flash("Concept_Net:- "+str(concept_net_res[0]))
                flash("Concept_Net_IS_A:- "+str(concept_net_res[1]))
            else:
                session['inf_cat'] = ''
                session['message'] = sent
                flash('Result :- ' + 'Add to the master file')
                flash('Inferred Category :- ' + '')
                flash(lex_res)
                flash(comprehend_res)
                flash(comprehend_syn)
                flash(concept_net_res)
        return redirect('http://52.90.174.141:9000/submit')
    return render_template('flask.html', form = form)
host="sumithapauroramysql20181211.cladr7eisf0t.us-east-1.rds.amazonaws.com"
port=3306
dbname="Lex_Training"
user="SumithaP"
password="lotus12345"


@app.route('/submit', methods = ['GET', 'POST'])
def submit_category():
    form = GetDataForm_2(request.form)
    print(request.form.get('submit2'))
    if request.method == 'POST':
        conn = pymysql.connect(host, user=user,port=port,passwd=password, db=dbname)
        cur = conn.cursor()
        sentence2 =request.form.get('sentence2')
        flash('Message:- ' + session['message'])
        flash('Inferred Category :- ' + session['inf_cat'])
        flash('Submitted Category :-' + sentence2)
        cur.execute('INSERT INTO reinforcement (message,inferred_category,submitted_category) VALUES (%s, %s, %s)', (session['message'], session['inf_cat'],sentence2 ))
        conn.commit()
        cur.close()
        if request.form.get('submit2') == 'Home':
            session.clear()
            return redirect('http://52.90.174.141:9000')
    return render_template('flask_2.html', form = form)


if __name__ == '__main__':
    app.run()



