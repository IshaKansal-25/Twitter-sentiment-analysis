# load string cleaning modules
import re,string

from nltk.tag import pos_tag        # load sentence tagger
from nltk.stem.snowball import SnowballStemmer          # remove -s,-es,-ing etc suffuxes from word
from nltk.stem.wordnet import WordNetLemmatizer         # find root word, it need to know part of speech of word in order to find root word

#function to clean grammar
def clean_grammer(value:list,
                  stemmer_=SnowballStemmer('english',ignore_stopwords=False),
                  lemmatizer_=WordNetLemmatizer()):
  # Valid options for parts of speech in WordNetLemmatizer are 'n' for nouns, 'v' for verbs, 'a' for adjectives, 'r' for adverbs and 's' for satellite adjectives
  """
  POS tag - meaning -> lemmatizer 

  J - adjective -> a
  N - noun -> n
  R - adverb -> r
  V - verb -> v
  """

  clean_value=list()        # list of clean sentences

  for word,pos in pos_tag(value):
    if stemmer_:word=stemmer_.stem(word)              # apply stemmer
    elif pos.startwith('J'):word=lemmatizer_.lemmatize(word,'a')
    elif pos.startwith('R'):word=lemmatizer_.lemmatize(word,'r')
    elif pos.startwith('V'):word=lemmatizer_.lemmatize(word,'v')
    # elif pos.startwith('N'):word=lemmatizer_.lemmatize(word,'n')
    else: word=lemmatizer_.lemmatize(word,'n')

    clean_value.append(word)          # append clean words to list of clean words
  
  return clean_value


from nltk.tokenize import word_tokenize           # load word tokenize
from nltk.corpus import stopwords                 # load stopwords(helping words or words that don't add meaning to sentence)

def clean(value:str,
          tags2remove:list=['http[\S]+','@[\S]+','&amp[\S]+'],            # \S - string 
          tag2convert:dict={' positive ':[':)',';)',':-)',':p',':D'],
                            ' negative ':[':(',';(',':-(']},
          cleaner:str='[^a-zA-Z ]',
          stopwords:list=stopwords.words(fileids='english'),
          minimum_word_length:int=3,
          call_clean_grammer:bool=True,
          clean_grammer_method=clean_grammer):
  
  # if value is list or tuple
  if isinstance(value,(list,tuple)):
    # call clean itself
    value=[clean(sentc) for sentc in value]
    return value          # return cleaned value

  #remove links and @ tags
  for tag in tags2remove:
    value=re.sub(tag,'',value)        # remove and update string
  
  # replace special character tags
  for tag_name,tags in tag2convert.items():
    for tag in tags:
      value=value.replace(tag,tag_name)       # replace tag woth positive or negative

  value=re.sub(cleaner,'',value)              # remove all punctuations

  value=value.casefold()                      # casefold() - converts to lower case forcefully

  value=word_tokenize(value)                  # break sentence to list of words (string -> list)

  value=[word for word in value if word not in stopwords]             # remove stopwords

  value=[word for word in value if not len(word)<minimum_word_length]            # remove words less than given minimum word length

  if call_clean_grammer: return clean_grammer_method(value)
  else: return value              #return clean string without cleaning grammer


#load module
import pickle

#load model
with open('nltk.nb.model',mode='rb') as model_file:
        model=pickle.load(model_file)

#get string from user
tweet=input('Enter tweet to check:')
#check tweet
prediction=model.classify(dict([(word,True) for word in clean(tweet)]))
# see output
print(f'The given tweet -> \n\t"{tweet}"\n\t is "{prediction}".')
