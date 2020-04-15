```sh
$ pip3 install kaggle
# account > api token key > kaggle.json
$ chmod 600 ~/.kaggle/kaggle.json
$ kaggle --version
	Kaggle API 1.5.6
$ kaggle competitions list
	...
# accept rules, validate user
$ kaggle competitions submissions -c nlp-getting-started
	403 - Forbidden
# competition != task
$ kaggle competitions submit -c nlp-getting-started --file /mnt/g/Users/pie/Downloads/nih/covid19/kaggle/kernel36728ad265.ipynb --message gisblog
# fork..
```sh
