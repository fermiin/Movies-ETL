#!/usr/bin/env python
# coding: utf-8

# In[39]:


import json
import pandas as pd
import numpy as np
import re


# In[2]:


file_dir = '/Users/saira/Desktop/class/'


# In[3]:


with open(f'{file_dir}/wikipedia-movies.json', mode='r') as file:
    wiki_movies_raw = json.load(file)


# In[4]:


len(wiki_movies_raw)


# In[5]:


kaggle_metadata = pd.read_csv(f'{file_dir}movies_metadata.csv', low_memory=False)
ratings = pd.read_csv(f'{file_dir}ratings.csv')


# In[6]:


wiki_movies_df = pd.DataFrame(wiki_movies_raw) 


# In[7]:


#see list of all columns 
wiki_movies_df.columns.tolist()


# In[25]:


#list comprehension to filter data to movies with a director, imdb link, tv shows, 
wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]
len(wiki_movies)


# In[26]:


#new function to clean movie data
def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    return movie


# In[27]:


#movies in arabic
wiki_movies_df[wiki_movies_df['Arabic'].notnull()]


# In[28]:


#Make an empty dict to hold all of the alternative titles
def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
#Loop through a list of all alternative title keys
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune–Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
#Check if the current key exists in the movie object.
        if key in movie:
    #If so, remove the key-value pair and add to the alternative titles dictionary.
            alt_titles[key] = movie[key]
            movie.pop(key)
    #add the alternative titles dict to the movie object.
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles

# merge column names
    def change_column_name(old_name, new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)
    change_column_name('Adaptation by', 'Written by')
    change_column_name('Country of origin', 'Country')
    change_column_name('Directed by', 'Director')
    change_column_name('Distributed by', 'Distributor')
    change_column_name('Edited by', 'Editor(s)')
    change_column_name('Length', 'Running time')
    change_column_name('Original release', 'Release date')
    change_column_name('Music by', 'Composer(s)')
    change_column_name('Produced by', 'Producer(s)')
    change_column_name('Producer', 'Producer(s)')
    change_column_name('Productioncompanies ', 'Production company(s)')
    change_column_name('Productioncompany ', 'Production company(s)')
    change_column_name('Released', 'Release date')
    change_column_name('Screen story by', 'Written by')
    change_column_name('Screenplay by', 'Written by')
    change_column_name('Story by', 'Written by')
    change_column_name('Theme music composer', 'Composer(s)')
    change_column_name('Written by', 'Writer(s)')

    return movie


# In[29]:


#make a list of cleaned movies with a list comprehension
clean_movies = [clean_movie(movie) for movie in wiki_movies]


# In[23]:


#list of the columns
wiki_movies_df = pd.DataFrame(clean_movies)
sorted(wiki_movies_df.columns.tolist())


# In[30]:


#extracting imbd id 
wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
print(len(wiki_movies_df))
#dropping duplicates
wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
print(len(wiki_movies_df))
wiki_movies_df.head()


# In[31]:


#get the count of null values for each column using a list comprehension
[[column,wiki_movies_df[column].isnull().sum()] for column in wiki_movies_df.columns]


# In[32]:


#columns that have less than 90% null values, will be kept
[column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]


# In[33]:


#columns to keep on Pandas DataFrame
wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]


# In[34]:


#data types of columns
wiki_movies_df.dtypes


# In[42]:


#data series that drops missing values in box office column
box_office = wiki_movies_df['Box office'].dropna()
#new function to see which values are not strings
def is_not_a_string(x):
    return type(x) != str
box_office[box_office.map(is_not_a_string)]


# In[43]:


#using lambda instead of 'is_not_a_string'
box_office[box_office.map(lambda x: type(x) != str)]


# In[44]:


#join() string method that concatenates list items into one string
box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[45]:


#first form for box office 
form_one = r'\$\d+\.?\d*\s*[mb]illion'
box_office.str.contains(form_one, flags=re.IGNORECASE, na=False).sum()


# In[46]:


#second form
form_two = r'\$\d{1,3}(?:,\d{3})+'
box_office.str.contains(form_two, flags=re.IGNORECASE, na=False).sum()


# In[47]:


#two boolean series 
matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE, na=False)
matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE, na=False)


# In[48]:


#what didnt fit the formats checked
box_office[~matches_form_one & ~matches_form_two]


# In[64]:


#new, to add more formats
form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'

box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)


# In[65]:


#extract parts of the strings that match 
box_office.str.extract(f'({form_one}|{form_two})')


# In[66]:


#function to turn the extracted values into a numeric value 
def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.# million
    if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " billion"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a billion
        value = float(s) * 10**9

        # return value
        return value

    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

        # remove dollar sign and commas
        s = re.sub('\$|,','', s)

        # convert to float
        value = float(s)

        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan


# In[67]:


wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)


# In[70]:


#dropping box office column
wiki_movies_df.drop('box_office', axis=1, inplace=True)


# In[73]:


#budget variable
budget = wiki_movies_df['Budget'].dropna()
#convert any lists to strings
budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)
#removing any values between $ and - (for budgets given in ranges)
budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)


# In[74]:


matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE, na=False)
matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE, na=False)
budget[~matches_form_one & ~matches_form_two]


# In[75]:


#removing the citation references []
budget = budget.str.replace(r'\[\d+\]\s*', '')
budget[~matches_form_one & ~matches_form_two]


# In[76]:


#parse the budget values
wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)


# In[77]:


#drop budget column
wiki_movies_df.drop('Budget', axis=1, inplace=True)


# In[78]:


#parsing release date
#converting lists to strings
release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[79]:


date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]?\d,\s\d{4}'
date_form_two = r'\d{4}.[01]\d.[0123]\d'
date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
date_form_four = r'\d{4}'


# In[80]:


#extracting the dates
release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)


# In[81]:


#parse the dates
wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)


# In[82]:


#parse running time
#convert list to strings
running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[83]:


#how many entries look like "blank minutes"
running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE, na=False).sum()


# In[84]:


#what other entries look like
running_time[running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE, na=False) != True]


# In[85]:


#adding abbreviations of minutes
running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE, na=False).sum()


# In[86]:


#remaing entries 
running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE, na=False) != True]


# In[87]:


#parsing remaining entries
running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')


# In[88]:


#turns any empty strings into NaN, which will be 0 
running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)


# In[89]:


#function to convert to minutes and save to wiki_movies_df
wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)


# In[90]:


#drop running time from dataset
wiki_movies_df.drop('Running time', axis=1, inplace=True)


# In[91]:


#cleaning kaggle data


# In[92]:


kaggle_metadata.dtypes


# In[93]:


#check if all the values are either true or false
kaggle_metadata['adult'].value_counts()


# In[94]:


#movies with corrupted data
kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]


# In[95]:


#keep rows where adult columns false and drop the adult column
kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')


# In[96]:


#video column values, check if true or false
kaggle_metadata['video'].value_counts()


# In[97]:


#convert data type 
kaggle_metadata['video'] == 'True'


# In[98]:


#assigning back to video
kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'


# In[99]:


#if any errors, "raise"
kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')


# In[100]:


#converting release_Date to datetime
kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])


# In[101]:


#checking the ratings data
ratings.info(null_counts=True)


# In[103]:


#unix for time
pd.to_datetime(ratings['timestamp'], unit='s')


# In[104]:


#assign it to timestamp column
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')


# In[105]:


#checking stats of the ratings with a histogram
pd.options.display.float_format = '{:20,.2f}'.format
ratings['rating'].plot(kind='hist')
ratings['rating'].describe()


# In[108]:


#merge kaggle and wiki data
movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])
movies_df.columns.tolist()


# In[109]:


# Competing data:
# Wiki                     Movielens                Resolution
#--------------------------------------------------------------------------
# title_wiki               title_kaggle             Drop wiki 
# running_time             runtime                  Keep Kaggle; fill in zeros with Wikipedia data
# budget_wiki              budget_kaggle            Keep Kaggle; fill in zeros with Wikipedia data
# box_office               revenue                  Keep Kaggle; fill in zeros with Wikipedia data
# release_date_wiki        release_date_kaggle      Drop wiki    
# Language                 original_language        Drop wiki 
# Production company(s)    production_companies     Drop wiki


# In[110]:


#rows were titles dont match
movies_df[movies_df['title_wiki'] != movies_df['title_kaggle']][['title_wiki','title_kaggle']]


# In[111]:


# Show any rows where title_kaggle is empty
movies_df[(movies_df['title_kaggle'] == '') | (movies_df['title_kaggle'].isnull())]


# In[112]:


#scatter plot of running time
movies_df.fillna(0).plot(x='running_time', y='runtime', kind='scatter')


# In[113]:


#plot line for movie dates
movies_df[['release_date_wiki','release_date_kaggle']].plot(x='release_date_wiki', y='release_date_kaggle', style='.')


# In[114]:


#look at outlier 
movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')]


# In[115]:


#drop row
movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)


# In[116]:


#check for null values
movies_df[movies_df['release_date_wiki'].isnull()]


# In[117]:


#convert lists in language to tuples
movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)


# In[118]:


#kaggle data value counts
movies_df['original_language'].value_counts(dropna=False)


# In[119]:


#drop the title_wiki, release_date_wiki, Language, and Production company(s) columns
movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)


# In[126]:


#function that fills in missing data and drops redundant column
def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)


# In[132]:


#fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
#fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
#fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
movies_df


# In[ ]:




