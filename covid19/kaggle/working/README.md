See the notebook for kaggle.py at https://www.kaggle.com/gisblog/cord-19-vectorizer-all-tasks/.

Walk given path for papers. If conditions are met, then write paper paths to a main papers file.
This creates - papers.biorxiv_medrxiv.json, papers.comm_use_subset.json, papers.noncomm_use_subset.json, papers.pmc_custom_license.json.

Walk given path for answer files. If conditions are met, then merge all answers on a given path by task # and source type into a main answer file. The merged JSON is structured like the original papers, and contains pointers to the original papers for reference.
This creates - answers.task.0.biorxiv_medrxiv.json, answers.task.0.comm_use_subset.json, answers.task.0.noncomm_use_subset.json, answers.task.0.pmc_custom_license.json.

For example, potential answers for Task 1 - "WHAT IS KNOWN ABOUT TRANSMISSION, INCUBATION, AND ENVIRONMENTAL STABILITY?" can be found here: (broken down by source type)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.0.biorxiv_medrxiv.json">bioRxiv-medRxiv</a> (640 kb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.0.comm_use_subset.json">Commmercial Use</a> (9.2 mb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.0.noncomm_use_subset.json ">Non-commercial Use</a> (1.7 mb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.0.pmc_custom_license.json">PubMed Central (PMC)</a> (1.2 mb)

Potential answers for Task 3 - "HELP US UNDERSTAND HOW GEOGRAPHY AFFECTS VIRALITY?" can be found here:
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.3.biorxiv_medrxiv.json">bioRxiv-medRxiv</a> (660 kb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.3.comm_use_subset.json">Commmercial Use</a> (9.1 mb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.3.noncomm_use_subset.json ">Non-commercial Use</a> (1.7 mb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/answers.task.3.pmc_custom_license.json">PubMed Central (PMC)</a> (1.2 mb)

We can also geoparse and geocode the extracted NLP answers, and map them so that the scientific community can visualize its research - See kaggle-geoparse.py and answers.task.3.biorxiv_medrxiv.json.

Geoparsed answers for Task 3 - "HELP US UNDERSTAND HOW GEOGRAPHY AFFECTS VIRALITY?" can be found here:
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/cities.answers.task.3.biorxiv_medrxiv.json">bioRxiv-medRxiv</a> (660 kb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/cities.answers.task.3.comm_use_subset.json">Commmercial Use</a> (9.1 mb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/cities.answers.task.3.noncomm_use_subset.json ">Non-commercial Use</a> (1.7 mb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/cities.answers.task.3.pmc_custom_license.json">PubMed Central (PMC)</a> (1.2 mb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/country_mentions.answers.task.3.biorxiv_medrxiv.json">bioRxiv-medRxiv</a> (660 kb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/country_mentions.answers.task.3.comm_use_subset.json">Commmercial Use</a> (9.1 mb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/working/country_mentions.answers.task.3.noncomm_use_subset.json ">Non-commercial Use</a> (1.7 mb)
* <a href="//raw.githubusercontent.com/gisblog/nih-covid19/master/covid19/kaggle/country_mentions.working/answers.task.3.pmc_custom_license.json">PubMed Central (PMC)</a> (1.2 mb)
