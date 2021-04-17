
# first, we import the relevant modules from the NLTK library
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# next, we initialize VADER so we can use it within our Python script
sid = SentimentIntensityAnalyzer()

# the variable 'message_text' now contains the text we will analyze.
message_text = '''9-year-old Daniel and 13-year-old Nafissa receive new backpacks on their first day back to school in Cameroon. They're excited to be learning with their classmates and teachers again after months of COVID-19 closures.
“I was bored when the school was closed, and I missed my friends. I realize how much I like school." Daniel dreams of becoming a police officer.
"I’m so happy I can come back to school. I wasn’t happy when the school was closed. I like to learn," says Nafissa who hopes to one day become a nurse.
Every child deserves to have their dreams fulfilled. Yet, millions continue to be denied the chance to go to school and learn. Governments must spare no effort to get them back into classrooms. Their futures depend on it.
'''

print(message_text)

# Calling the polarity_scores method on sid and passing in the message_text outputs a dictionary with negative, neutral, positive, and compound scores for the input text
scores = sid.polarity_scores(message_text)

# Here we loop through the keys contained in scores (pos, neu, neg, and compound scores) and print the key-value pairs on the screen

for key in sorted(scores):
        print('{0}: {1}, '.format(key, scores[key]), end='')
#Save your Python file. Now we’re ready to execute the code. Using your preferred method (either your Integrated Development Environment, or the command line), run your Python file, sentiment.py.


