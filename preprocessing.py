import spacy
import re

nlp = spacy.load('en_core_web_sm')

def preprocess_email(email):
    email = email.lower()
    email = re.sub(r'^subject:\s*(.*)', r'\1', email, flags=re.IGNORECASE)
    tokens = [
        token.lemma_
        for token in nlp(email)
        if not token.is_stop and not token.is_punct
    ]
    return ' '.join(tokens)

def preprocess_emails(emails):
    return [preprocess_email(str(email)) for email in emails]