"""Reproduce curățarea, comparația pe validare, testul final și modelul livrat."""
from pathlib import Path
import argparse, json, pickle, hashlib, platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.base import clone

ROOT = Path(__file__).resolve().parent

def title_stats(texts):
    """Caracteristici disponibile și la predicție: lungime, cuvinte, cifre, simboluri."""
    return np.array([[len(t), len(t.split()), sum(c.isdigit() for c in t),
                      sum(not c.isalnum() and not c.isspace() for c in t)] for t in texts], dtype=float)

def load_data(path):
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()
    required = ['Product Title', 'Category Label']
    if not set(required).issubset(raw): raise ValueError('Lipsesc coloanele pentru titlu/categorie.')
    d = raw.dropna(subset=required).copy()
    for col in required: d[col] = d[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    d = d.loc[d[required].ne('').all(axis=1)].copy()
    d['Category Label'] = d['Category Label'].replace({'CPU':'CPUs', 'Mobile Phone':'Mobile Phones', 'fridge':'Fridges'})
    d['key'] = d['Product Title'].str.lower()
    conflicts = d.groupby('key')['Category Label'].nunique()
    conflict_keys = conflicts[conflicts > 1].index
    conflict_rows = int(d['key'].isin(conflict_keys).sum())
    d = d.loc[~d['key'].isin(conflict_keys)].copy()
    duplicates = int(d.duplicated('key').sum())
    d = d.drop_duplicates('key').reset_index(drop=True)
    quality = {'raw_rows':len(raw), 'missing_by_column':raw.isna().sum().to_dict(),
               'missing_or_empty_required':len(raw)-len(raw.dropna(subset=required).loc[lambda z: z[required].astype(str).apply(lambda s:s.str.strip()).ne('').all(axis=1)]),
               'conflicting_titles':len(conflict_keys), 'conflicting_rows_removed':conflict_rows,
               'duplicates_removed':duplicates, 'clean_rows':len(d)}
    return raw, d, quality

def candidates():
    word = lambda: TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True, max_features=50000)
    return {
        'word_nb': Pipeline([('features',word()),('classifier',MultinomialNB(alpha=0.5))]),
        'word_lr': Pipeline([('features',word()),('classifier',LogisticRegression(max_iter=1000,random_state=42))]),
        'word_svc': Pipeline([('features',word()),('classifier',LinearSVC(random_state=42,max_iter=5000))]),
        'char_svc': Pipeline([('features',TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=60000,sublinear_tf=True)),('classifier',LinearSVC(random_state=42,max_iter=5000))]),
        'word_stats_svc': Pipeline([('features',FeatureUnion([
            ('text',word()),('stats',Pipeline([('extract',FunctionTransformer(title_stats,validate=False)),('scale',MaxAbsScaler())]))],transformer_weights={'text':1.0,'stats':0.2})),('classifier',LinearSVC(random_state=42,max_iter=5000))])
    }

def train(data_path=ROOT/'products.csv', output=ROOT):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    raw,d,quality=load_data(data_path)
    dev,test=train_test_split(d,test_size=0.2,random_state=42,stratify=d['Category Label'])
    train,valid=train_test_split(dev,test_size=0.25,random_state=42,stratify=dev['Category Label'])
    assert set(train.key).isdisjoint(valid.key) and set(dev.key).isdisjoint(test.key)
    scores=[]; options=candidates()
    for name, model in options.items():
        model.fit(train['Product Title'],train['Category Label'])
        pred=model.predict(valid['Product Title'])
        scores.append({'model':name,'accuracy':accuracy_score(valid['Category Label'],pred),
                       'macro_f1':f1_score(valid['Category Label'],pred,average='macro')})
        print(scores[-1],flush=True)
    comparison=pd.DataFrame(scores).sort_values(['macro_f1','accuracy'],ascending=False)
    winner=comparison.iloc[0]['model']
    selected=clone(options[winner]).fit(dev['Product Title'],dev['Category Label'])
    pred=selected.predict(test['Product Title'])
    labels=sorted(d['Category Label'].unique())
    report=classification_report(test['Category Label'],pred,labels=labels,output_dict=True,zero_division=0)
    matrix=confusion_matrix(test['Category Label'],pred,labels=labels)
    comparison.to_csv(output/'validation_comparison.csv',index=False)
    pd.DataFrame(report).T.to_csv(output/'classification_report.csv')
    pd.DataFrame(matrix,index=labels,columns=labels).to_csv(output/'confusion_matrix.csv')
    errors=test[['Product Title','Category Label']].copy();errors['prediction']=pred
    errors.loc[errors['Category Label']!=errors.prediction].to_csv(output/'test_errors.csv',index=False)
    final=clone(options[winner]).fit(d['Product Title'],d['Category Label'])
    with (output/'product_model.pkl').open('wb') as f:pickle.dump(final,f,protocol=4)
    with (output/'product_model.pkl').open('rb') as f:loaded=pickle.load(f)
    assert np.array_equal(final.predict(test['Product Title'].head(20)),loaded.predict(test['Product Title'].head(20)))
    meta={'quality':quality,'split':{'train':len(train),'validation':len(valid),'test':len(test)},
          'winner':winner,'test_accuracy':report['accuracy'],'test_macro_f1':report['macro avg']['f1-score'],
          'majority_baseline':float((test['Category Label']==dev['Category Label'].mode()[0]).mean()),
          'source_sha256':hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
          'python':platform.python_version(),'sklearn':sklearn.__version__,
          'note':'Test scores belong to model fit on development 80%; delivered model refit on all clean data.'}
    (output/'metrics.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2),flush=True)
    return raw,d,comparison,meta,loaded

if __name__=='__main__':
    # Import via module name so custom transformers remain portable in pickle.
    from train_model import train as run
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=ROOT/'products.csv');ap.add_argument('--output',type=Path,default=ROOT)
    args=ap.parse_args();run(args.data,args.output)
