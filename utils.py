from sklearn.pipeline import Pipeline

pipe = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=5,
                                               norm='l2', encoding='latin-1',
                                               ngram_range=(1, 1),
                                               stop_words=['english'])),
                     ('svd', TruncatedSVD(n_components=n_components,
                                          n_iter=7,
                                          random_state=random_state)),
                     ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1),
                                                   n_jobs=-1))
                 ])
