import nltk
import PyPDF2
from gensim.summarization import summarize
from nltk.util import ngrams
from collections import Counter

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

pdf_file = input("Enter the pdf file name :")
summary_file = input("Enter the summary file name :")
start_page = int(input("Enter the start page number :"))
end_page = int(input("Enter the end page number :"))

# Open the PDF file
with open(pdf_file, "rb") as file:
    pdf = PyPDF2.PdfFileReader(file)
    
    # Extract the text from the PDF
    text = ""
    for page in range(start_page-1,end_page):
        text += pdf.getPage(page).extractText()
        
    # Summarize the text
    summary = summarize(text, ratio=0.2, split=True)
    
    # Save the summary to a text file
    with open(summary_file, "w") as sum_file:
        for point in summary:
            sum_file.write("- " + point + "\n")
    
    print(f"Summary saved to {summary_file}")
   
    # tokenize the text
    tokens = nltk.word_tokenize(text)
    # extract bigrams
    bigrams = ngrams(tokens,2)
    # count the bigrams
    bigram_freq = Counter(bigrams)
    # print the 10 most common bigrams
    print("\n10 most common bigrams :")
    print(bigram_freq.most_common(10))
