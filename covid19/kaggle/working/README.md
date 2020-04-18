![Selected places in the extracted NLP answers for Task 3 - "HELP US UNDERSTAND HOW GEOGRAPHY AFFECTS VIRALITY?" (bioRxiv-medRxiv)](https://github.com/gisblog/nih-covid19/raw/master/covid19/kaggle/working/answers.task.3.biorxiv_medrxiv.json.jpg)
**Selected places in the extracted NLP answers for Task 3 - "HELP US UNDERSTAND HOW GEOGRAPHY AFFECTS VIRALITY?"**

# What

See the Kaggle notebook at https://www.kaggle.com/gisblog/cord-19-vectorizer-all-tasks/.

Walk given path for papers. If conditions are met, then write paper paths to a main papers file.
This creates - papers.biorxiv_medrxiv.json, papers.comm_use_subset.json, papers.noncomm_use_subset.json, papers.pmc_custom_license.json.

Walk given path for answer files. If conditions are met, then merge all answers on a given path by task # and source type into a main answer file. The merged JSON is structured like the original papers, and contains pointers to the original papers for reference.
This creates - answers.task.0.biorxiv_medrxiv.json, answers.task.0.comm_use_subset.json, answers.task.0.noncomm_use_subset.json, answers.task.0.pmc_custom_license.json.

For example, potential answers for Task 1 - "WHAT IS KNOWN ABOUT TRANSMISSION, INCUBATION, AND ENVIRONMENTAL STABILITY?" can be found here: (broken down by source type)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.0.biorxiv_medrxiv.json">bioRxiv-medRxiv</a> (640 kb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.0.comm_use_subset.json">Commmercial Use</a> (9.2 mb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.0.noncomm_use_subset.json">Non-commercial Use</a> (1.7 mb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.0.pmc_custom_license.json">PubMed Central (PMC)</a> (1.2 mb)

Potential answers for Task 3 - "HELP US UNDERSTAND HOW GEOGRAPHY AFFECTS VIRALITY?" can be found here:
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.3.biorxiv_medrxiv.json">bioRxiv-medRxiv</a> (660 kb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.3.comm_use_subset.json">Commmercial Use</a> (9.1 mb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.3.noncomm_use_subset.json">Non-commercial Use</a> (1.7 mb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.3.pmc_custom_license.json">PubMed Central (PMC)</a> (1.2 mb)

We can also geoparse and geocode the extracted NLP answers for visualization.

Geoparsed answers for Task 3 - "HELP US UNDERSTAND HOW GEOGRAPHY AFFECTS VIRALITY?":

Cities -

* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/cities.answers.task.3.biorxiv_medrxiv.json">bioRxiv-medRxiv</a>
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/cities.answers.task.3.comm_use_subset.json">Commmercial Use</a>
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/cities.answers.task.3.noncomm_use_subset.json">Non-commercial Use</a>
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/cities.answers.task.3.pmc_custom_license.json">PubMed Central (PMC)</a>

Country Codes and Counts -

* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/country_mentions.answers.task.3.biorxiv_medrxiv.json">bioRxiv-medRxiv</a>
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/country_mentions.answers.task.3.comm_use_subset.json">Commmercial Use</a>
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/country_mentions.answers.task.3.noncomm_use_subset.json">Non-commercial Use</a>
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/country_mentions.answers.task.3.pmc_custom_license.json">PubMed Central (PMC)</a>

# Why

Pros and cons of using CountVectorizer over HashingVectorizer (or TfidfVectorizer) -
* CountVectorizer uses in-memory vocabulary.
* HashingVectorizer doesn't have a way to compute the inverse transform (from feature indices to string feature names). This can be a problem when trying to introspect which features are most important to a model. Also, no IDF (Inverse Document Frequency) weighting - IDF measures how important a word is to a doc in a collection of docs.
  * TF-IDF increases with the # of times a word appears in a doc.
  * TF-IDF decreases with the # of docs in the collection that contain the word.

Given the large unlabeled corpora, WORD2VEC - a group of models that represents each word in a large text as a [vector] in a space of N-dimensions (or features) making similar words closer to each other - was found to be more granular for the COVID19 use-case. It was used over DOC2VEC because the concepts or individual IDs of the docs/papers themselves weren't the most important factors towards the closest answers i.e. What mattered more than their authors, sponsors or tags was if the papers had the rightly-worded answers to the questions posed. DOC2VEC adds a doc/para [vector], and is generally more helpful when the papers have tags. E.g. to find duplicate papers, or papers by similar authors.

The Euclidean distance (or cosine similarity) between 2 word [vector]s provided an effective method for measuring the linguistic/semantic similarity of the 2 words. Nearest neighbor reveals relevant similarities outside an average vocabulary. Similarity metrics used in nearest neighbor evaluations produce 1 (scalar) that quantifies the relatedness of its 2 words. This simplicity can be an issue since 2 words may exhibit other relationships. In order to capture that in a quantitative way, it was necessary to associate more than 1 number to a word pair. NGRAM_RANGE was used to determine context by weighing nearby words more heavily than distant words. BOW was found to be less accurate as it ignored word ordering.

Populating GloVe required 1 pass through the entire COVID19 dataset. For the large COVID19 dataset, this pass was computationally expensive. Subsequent training iterations would have been faster. Also, pre-trained word [vector] datasets downloaded (e.g. Wikipedia 2014 + Gigaword 5) didn't match the semantics for COVID19.

And a search engine-type approach was found to be more deterministic than what was understood to be expected or even accurate. Therefore, the approach taken was to massively trim the papers the scientists would need to digest. Future enhancements could include clustering for visualization.

Related:
* https://arxiv.org/abs/1301.3781
* http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
* https://radimrehurek.com/gensim/
* https://nlp.stanford.edu/projects/glove/
* https://github.com/stanleyfok/sentence2vec
