"""Încarcă pipeline-ul salvat și prezice categorii pentru titluri introduse."""
from pathlib import Path
import argparse, pickle

def predict(model, title):
    title=' '.join(title.split())
    if not title: raise ValueError('Introdu un titlu nevid.')
    return str(model.predict([title])[0])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,default=Path(__file__).resolve().parent/'product_model.pkl')
    parser.add_argument('--title',help='Predicție unică; dacă lipsește, pornește modul interactiv.')
    args=parser.parse_args()
    if not args.model.exists():parser.error('Modelul lipsește. Rulează train_model.py.')
    # Încarcă numai fișiere pickle provenite din surse de încredere.
    with args.model.open('rb') as f:model=pickle.load(f)
    if args.title is not None:
        try:print(predict(model,args.title))
        except ValueError as e:parser.error(str(e))
        return
    print('Introdu titlul produsului. Scrie exit pentru închidere.')
    while True:
        try:title=input('Produs: ')
        except (EOFError,KeyboardInterrupt):break
        if title.strip().lower() in {'exit','quit','iesire'}:break
        try:print('Categoria sugerată:',predict(model,title))
        except ValueError as e:print(e)

if __name__=='__main__':main()
