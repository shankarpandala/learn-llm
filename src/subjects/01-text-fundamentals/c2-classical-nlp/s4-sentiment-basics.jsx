import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

export default function SentimentBasics() {
  return (
    <div className="mx-auto max-w-4xl space-y-8 px-4 py-8">
      <h1 className="text-3xl font-bold">Sentiment Analysis Basics</h1>
      <p className="text-lg text-gray-700 dark:text-gray-300">
        Sentiment analysis determines the emotional tone or opinion expressed in text. It ranges
        from simple positive/negative classification to fine-grained analysis of aspect-level
        sentiment and emotion detection. It is one of the most commercially important NLP tasks.
      </p>

      <DefinitionBlock
        title="Sentiment Analysis"
        definition="Sentiment analysis (or opinion mining) is the task of identifying and extracting subjective information from text. At its simplest, it classifies text as positive, negative, or neutral. More advanced forms detect the target of sentiment, the holder of the opinion, and the intensity of emotion."
        id="def-sentiment"
      />

      <h2 className="text-2xl font-semibold">Lexicon-Based Methods</h2>
      <p className="text-gray-700 dark:text-gray-300">
        The simplest approach uses a sentiment lexicon: a dictionary that maps words to sentiment
        scores. The overall sentiment of a text is computed by aggregating individual word scores.
      </p>

      <PythonCode
        title="lexicon_sentiment.py"
        code={`# Simple lexicon-based sentiment analysis
sentiment_lexicon = {
    "good": 1.0, "great": 1.5, "excellent": 2.0, "amazing": 2.0,
    "love": 1.5, "wonderful": 1.5, "fantastic": 2.0, "best": 1.5,
    "bad": -1.0, "terrible": -2.0, "awful": -2.0, "worst": -2.0,
    "hate": -1.5, "horrible": -2.0, "poor": -1.0, "boring": -1.0,
    "not": None,  # negation modifier
}

def lexicon_sentiment(text):
    """Simple lexicon sentiment with basic negation handling."""
    words = text.lower().split()
    score = 0.0
    negate = False
    for word in words:
        if word == "not" or word == "n't":
            negate = True
            continue
        if word in sentiment_lexicon and sentiment_lexicon[word] is not None:
            word_score = sentiment_lexicon[word]
            if negate:
                word_score *= -0.5  # Flip and dampen
                negate = False
            score += word_score
        else:
            negate = False  # Reset negation after non-sentiment word
    return score

reviews = [
    "This movie was great and the acting was excellent",
    "Terrible film with awful dialogue and bad pacing",
    "The food was not bad but not great either",
    "I love this amazing wonderful product",
]

for review in reviews:
    score = lexicon_sentiment(review)
    label = "POSITIVE" if score > 0 else "NEGATIVE" if score < 0 else "NEUTRAL"
    print(f"[{score:+.1f}] {label}: '{review}'")`}
        id="code-lexicon"
      />

      <h2 className="text-2xl font-semibold">Machine Learning Approaches</h2>
      <p className="text-gray-700 dark:text-gray-300">
        ML-based sentiment analysis treats the problem as text classification. A document is
        represented as a feature vector (e.g., TF-IDF), and a classifier (Naive Bayes, SVM,
        or logistic regression) predicts the sentiment label.
      </p>

      <p className="text-gray-700 dark:text-gray-300">
        For Naive Bayes, the predicted class is:
      </p>
      <div className="my-4">
        <BlockMath math="\hat{c} = \arg\max_{c} P(c) \prod_{i=1}^{n} P(w_i \mid c)" />
      </div>

      <PythonCode
        title="ml_sentiment.py"
        code={`from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# Sample training data
texts = [
    "I love this movie, it was fantastic",
    "Great film with wonderful performances",
    "The best movie I have seen this year",
    "Excellent acting and a compelling story",
    "This movie was terrible and boring",
    "Awful waste of time, worst film ever",
    "Bad acting, poor script, horrible movie",
    "I hated every minute of this film",
]
labels = [1, 1, 1, 1, 0, 0, 0, 0]  # 1=positive, 0=negative

# TF-IDF features
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
X = vectorizer.fit_transform(texts)

# Train classifiers
nb = MultinomialNB()
nb.fit(X, labels)

lr = LogisticRegression()
lr.fit(X, labels)

# Predict on new reviews
new_reviews = [
    "A wonderful and great experience",
    "Terrible and boring, do not watch",
    "The movie was okay, nothing special",
]

X_new = vectorizer.transform(new_reviews)
for review, nb_pred, lr_pred in zip(new_reviews, nb.predict(X_new), lr.predict(X_new)):
    print(f"NB={['NEG','POS'][nb_pred]}, LR={['NEG','POS'][lr_pred]}: '{review}'")`}
        id="code-ml-sentiment"
      />

      <ExampleBlock
        title="VADER: Rule-Based Sentiment for Social Media"
        problem="VADER (Valence Aware Dictionary and sEntiment Reasoner) handles emoticons, slang, and capitalization. How does it score: 'This movie is GREAT!!! :)' ?"
        steps={[
          { formula: '"GREAT" -> boosted score (all caps)', explanation: 'VADER boosts sentiment for capitalized words.' },
          { formula: '"!!!" -> intensified', explanation: 'Exclamation marks increase the intensity.' },
          { formula: '":)" -> positive emoticon', explanation: 'Emoticons have predefined sentiment values.' },
          { formula: 'Compound score: ~0.82 (strongly positive)', explanation: 'VADER combines and normalizes all signals into a compound score in [-1, 1].' },
        ]}
        id="example-vader"
      />

      <WarningBlock
        title="Sarcasm and Context"
        content="Lexicon and simple ML methods fail on sarcasm ('Oh great, another meeting'), domain-specific language ('This drug killed the infection' is positive in medicine), and implicit sentiment ('The battery lasted 20 minutes' is negative without using negative words). These require deeper contextual understanding."
        id="warning-sarcasm"
      />

      <NoteBlock
        type="note"
        title="From Classical to Neural Sentiment"
        content="Modern sentiment analysis uses fine-tuned Transformer models (BERT, RoBERTa) that achieve state-of-the-art accuracy by leveraging contextual embeddings. LLMs can also perform sentiment analysis via zero-shot prompting, often matching or exceeding supervised baselines without any training data."
        id="note-neural-sentiment"
      />
    </div>
  )
}
