from tkinter import *
from tkinter import filedialog
from tkinter import Text, Tk

import pandas as pd
import numpy as np
import re

data = pd.read_csv('cleaned_data.csv')
df = pd.DataFrame(data)
df = df.dropna(axis=0)
df['Article'] = df['Article'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))
with open('tagalog_stop_words.txt', 'r') as f:
    stop_tagalog = [line.strip() for line in f]


def text_process(mess):
    """
    Returns a list of important words only
    """

    return [word for word in mess.split() if word.lower() not in stop_tagalog]


from sklearn.feature_extraction.text import CountVectorizer

bow_transformer = CountVectorizer(analyzer=text_process).fit(df['Article'])
article_bow = bow_transformer.transform(df['Article'])
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(article_bow)
article_tfidf = tfidf_transformer.transform(article_bow)
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

text_clf_svm = Pipeline([('vect', CountVectorizer(analyzer=text_process)),('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha = 1e-3, n_iter = 5, random_state = 42))])
_ = text_clf_svm.fit(df['Article'], df['Category'])





#GUI - FrontEND
if __name__ == '__main__':
    root = Tk()
root.geometry("600x700")
root.configure(background ='white')
root.title("Filipino News Classifier")
root.iconbitmap(r'icon.ico')
photo = PhotoImage(file='header.png', width=600)
labelPhoto = Label(root, image=photo)
labelPhoto.pack()
label2 = Label(text="A system that utilizes SVM's algorithm", width = 70, font='helvlight 14',bg = 'white', anchor='n')
label2.pack()
label3 = Label(text="to classify Filipino News based on its category;", width=70, font='helvlight 14',bg='white', anchor='n')
label3.pack()
label4 = Label(text="Categories namely: Bansa, Metro, Probinsya,", width=70, font="helvlight 14", bg='white', anchor='n')
label4.pack()
label5 = Label(text="Palaro, Showbizz and Opinyon", width =70, font="helvlight 14", bg='white', anchor='n')
label5.pack()
L1 = Label(root, text="Copy & paste the News/Article here: ", font='helvlight 13', bg='white', anchor='n')
L1.pack()
S1 = Scrollbar(root)
S1.pack(side=RIGHT, fill=Y)
T1 = Text(root, wrap=WORD, yscrollcommand=S1.set, height=15, width=65)
T1.pack()
S1.config(command=T1.yview)

L2 = Label(root, text="Type of News/Article: ", font='helvlight 13', bg='white', anchor='n')
L2.pack()
T2 = Text(root, height=2, width=30)
T2.pack()
T2.config(state=DISABLED)

def clear():
    T2.config(state=NORMAL)
    T1.delete(1.0,END)
    T2.delete(1.0,END)
    T2.config(state=DISABLED)


def predict(value):
    T2.config(state=NORMAL)
    T2.delete(1.0,END)
    T2.insert(END,text_clf_svm.predict([value])[0])
    T2.config(state=DISABLED)




def openfile():
    filename=filedialog.askopenfilename(initialdir="/", title="Select .txt file")
    return filename

def batch_classify():
    f=open(openfile(),'r')
    write=open('newfile.txt','w')
    temp=""
    for line in f:
        if line=="\n":
            write.write("Category: "+ text_clf_svm.predict([temp])[0]+"\n\n")
            temp=""

        else:
            temp=temp+line
            write.write(line)

    write.write("\n"+"Category: "+ text_clf_svm.predict([temp])[0])
    T2.config(state=NORMAL)
    T2.delete(1.0,END)
    T2.insert(END, "File now available at directory")
    T2.config(state=DISABLED)

class Parent():
    b1 = Button(root, text='CLEAR', command=lambda:clear() ,font='helvlight 13', bg='#5dbcd2', fg='#050505')
    b1.pack(side=LEFT, expand=YES)


class Child(Parent):
    b2 = Button(root, text='BATCH CLASSIFY', command=lambda:batch_classify() ,font='helvlight 13', bg='#5dbcd2', fg='#050505')
    b2.pack(side=LEFT, expand=YES)
    b3 = Button(root, text='CLASSIFY', command= lambda: predict(T1.get(1.0,END)),font='helvlight 13', bg='#5dbcd2', fg='#050505')
    b3.pack(side=LEFT, expand=YES)
    b4 = Button(root, text='RETRAIN', font='helvlight 13', bg='#5dbcd2', fg='#050505')
    b4.pack(side=LEFT, expand=YES)





root.mainloop()

