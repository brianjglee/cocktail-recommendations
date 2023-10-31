import json, urllib
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import subprocess
from spacy.pipeline import EntityRuler

path = "./annotations.json"
file = open(path)
data = json.load(file)

# load the annotations into a list of lists
dataset = data["annotations"]

# split data itno training and development data
TRAIN_DATA = dataset[:111]
DEV_DATA = dataset[111:]

# check the length of the annotations 
print(len(dataset))

# creating a function that converts our annotations into spacy files. 
def convert(path, dataset):
    # loading a blank model with English language as base language.
    nlp = spacy.blank("en")
    db = DocBin()
    for text, annot in tqdm(dataset): 
            doc = nlp.make_doc(text) 
            # list to store entities
            ents = []
            for start, end, label in annot["entities"]:
                # char_span is a function that extracts each entity separately from our list as an object that spacy can interpret
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is None:
                    print("Skipping entity")
                else:
                    # append span object to list of entities
                    ents.append(span)
            doc.ents = ents 
            db.add(doc)
    db.to_disk(path)
    
# convert trainig and development data into spacy files
convert("train.spacy", TRAIN_DATA)
convert("dev.spacy", DEV_DATA)

# running the terminal scripts that will configure the files and train our model
subprocess.run('python -m spacy init config config.cfg --lang pt --pipeline ner --optimize efficiency --force', shell=True)
subprocess.run('python -m spacy train config.cfg --output ./ --paths.train ./train.spacy --paths.dev ./dev.spacy', shell=True)

# predetermined list of cocktail names for entity ruler to support our model.
cocktail_names = [
    "Martini",
    "Mojito",
    "Cosmopolitan",
    "Margarita",
    "Daiquiri",
    "Old Fashioned",
    "Manhattan",
    "Negroni",
    "Pina Colada",
    "Bloody Mary",
    "Mai Tai",
    "Whiskey Sour",
    "Gin and Tonic",
    "Moscow Mule",
    "Screwdriver",
    "Tom Collins",
    "Sidecar",
    "French 75",
    "Mint Julep",
    "Sazerac",
    "White Russian",
    "Bellini",
    "Singapore Sling",
    "Long Island Iced Tea",
    "Mimosa",
    "Piña Colada",
    "Tequila Sunrise",
    "Sex on the Beach",
    "Black Russian",
    "Grasshopper",
    "Boulevardier",
    "Planter's Punch",
    "Blue Lagoon",
    "Hurricane",
    "Dark n Stormy",
    "Irish Coffee",
    "Vodka Martini",
    "Rusty Nail",
    "Corpse Reviver",
    "Bee's Knees",
    "Aviation",
    "Brandy Alexander",
    "Caipirinha",
    "Gimlet",
    "Harvey Wallbanger",
    "Mai Tai",
    "Pisco Sour",
    "Rob Roy",
    "Rum Swizzle",
    "Vesper",
    "Zombie",
    "Angel Face",
    "Monkey Gland",
    "Ward Eight",
    "Mary Pickford",
    "Bramble",
    "Derby",
    "French Connection",
    "Golden Dream",
    "Godfather",
    "Hemingway Special",
    "Jungle Bird",
    "Last Word",
    "Paradise",
    "Pegu Club",
    "Pisco Sour",
    "Suffering Bastard",
    "Vieux Carre"
]

# Load the best model 
nlp = spacy.load("model-best")

# Create the EntityRuler and add patterns for the specific entity words
ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"}, after='ner')

# create a list of patterns from cocktail_names
patterns = []
for cocktail in cocktail_names:
    patterns.append({"label":"COCKTAIL", "pattern":cocktail.lower()})
ruler.add_patterns(patterns)