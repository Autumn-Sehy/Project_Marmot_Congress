import pickle
import numpy as np
import re

#this whole file is mostly for saving the huge lists I use over and over again
#they were cumbersome, so I wanted to store  them in one place

def get_word_embeddings():
    with open("results_and_embeddings/word_to_index.pkl", "rb") as f:
        word_to_index = pickle.load(f)
        return word_to_index

def get_state_names():
    state_names = [
        'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut',
        'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
        'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan',
        'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
        'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
        'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
        'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia',
        'Wisconsin', 'Wyoming', 'District of Columbia'
    ]
    return state_names

def get_congress_118_lastnames():
    congress_118_lastnames = [
        # Democrats
        "Adams", "Aguilar", "Allred", "Amo", "Auchincloss", "Balint", "Barragán", "Beatty", "Beyer",
        "Budzinski", "Bush", "Bishop","Boyle",  "Blumenauer","Bowman",  "Bonamici", "Bluntrochester", "Brownley", "Brown", "Bera","Caraveo", "Carbajal", "Cárdenas", "Carson", "Carter", "Cartwright",
        "Casar", "Case", "Casten", "Castor", "Castro", "Cherfilus-McCormick", "Chu", "Cicilline", "casar", "case", "casten", "castro", "cherfilus mccormick", "chu", "cicilline",
        "Clark", "Clarke", "Cleaver", "Clyburn", "Cohen", "David Scott","Connolly", "Correa", "Costa", "Courtney",
        "Craig", "Crockett", "Crow", "Cuellar", "Davids", "Davis", "Dean", "DeGette",
        "DeLauro", "DelBene", "Deluzio", "DeSaulnier", "Dingell", "Doggett", "Escobar", "Eshoo",
        "Espaillat", "Evans", "Fletcher", "Foster", "Foushee", "Frankel", "Frost", "Gallego",
        "Garamendi", "Garcia", "Golden", "Goldman", "Gomez", "Gonzalez", "Gottheimer", "Green", "Gluesenkamp Perez",
        "Grijalva", "Harder", "Hayes", "Himes", "Horsford", "Houlahan", "Hoyer", "Hoyle",
        "Huffman", "Ivey", "Jackson", "Jackson Lee", "Jacobs", "Jayapal", "Jeffries", "Johnson", "Kamlager-Dove",
        "Kaptur", "Keating", "Kelly", "Kennedy", "Khanna", "Kildee", "Kilmer", "Kim",
        "Krishnamoorthi", "Kuster", "Landsman", "Larsen", "Larson", "Lee", "Leger Fernandez",
        "Levin", "Lieu", "Lofgren", "Lynch", "Magaziner", "Manning", "Matsui", "McBath",
        "McClellan", "McCollum", "McGarvey","McGovern", "McIver", "Meeks", "Menendez", "Meng", "Mfume", "Moore",
        "Morelle", "Moskowitz", "Moulton", "Mrvan", "Mullin", "Nadler", "Napolitano", "Neal",
        "Neguse", "Nickel", "Norcross", "Norton", "Ocasio-Cortez", "Omar", "Pallone", "Panetta",
        "Pappas", "Pascrell", "Payne", "Pelosi", "Peltola", "Perez", "Peters", "Pettersen",
        "Phillips", "Pingree", "Plaskett", "Pocan", "Porter", "Pressley", "Quigley", "Ramirez",
        "Raskin", "Ross", "Ruiz", "Ruppersberger", "Ryan", "Sablan", "Salinas", "Sánchez",
        "Sarbanes", "Scanlon", "Schakowsky", "Schiff", "Schneider", "Scholten", "Schrier", "Scott",
        "Sewell", "Sherman", "Sherrill", "Slotkin", "Smith", "Sorensen", "Soto", "Spanberger",
        "Stansbury", "Stanton", "Stevens", "Strickland", "Suozzi", "Swalwell", "Sykes", "Takano",
        "Thanedar", "Thompson", "Titus", "Tlaib", "Tokuda", "Tonko", "Torres", "Trahan", "Trone",
        "Underwood", "Vargas", "Vasquez", "Veasey", "Velázquez", "Wasserman Schultz", "Waters",
        "Watson Coleman", "Wexton", "Wild", "Williams", "Wilson",

        # Republicans
        "Aderholt", "Alford", "Allen", "Amodei", "Armstrong", "Arrington", "Austin Scott", "Babin",
        "Bacon", "Baird", "Balderson", "Banks", "Barr", "Bean", "Bentz", "Bergman", "Bice",
        "Biggs", "Bilirakis", "Bishop", "Boebert", "Bost", "Brecheen", "Buchanan", "Buck",
        "Bucshon", "Burchett", "Burgess", "Burlison", "Calvert", "Cammack", "Carey", "Carl",
        "Carter", "Chavez-DeRemer", "Ciscomani", "Cline", "Cloud", "Clyde", "Cole", "Collins",
        "Comer", "Crane", "Crawford", "Crenshaw", "Curtis", "Scott Franklin", "D'Esposito",
        "Davidson", "De La Cruz", "DesJarlais", "Diaz-Balart", "Donalds", "Duarte", "Duncan",
        "Dunn", "Edwards", "Ellzey", "Emmer", "Estes", "Ezell", "Fallon", "Feenstra", "Ferguson",
        "Finstad", "Fischbach", "Fitzgerald", "Fitzpatrick", "Fleischmann", "Flood", "Fong",
        "Foxx", "Franklin", "Fry", "Fulcher", "Gaetz", "Gallagher", "Garbarino", "Garcia",
        "Gimenez", "Gonzales", "González-Colón", "Good", "Gooden", "Gosar", "Granger", "Graves",
        "Green", "Greene", "Griffith", "Grothman", "Guest", "Guthrie", "Hageman", "Hudson",
        "Harris", "Harshbarger", "Hern", "Higgins", "Hill", "Hinson", "Houchin", "Huizenga",
        "Hunt", "Issa", "Jackson", "James", "Johnson", "Jordan", "Joyce", "Kean", "Kelly",
        "Kiggans", "Kiley", "Kim", "Kustoff", "LaHood", "LaLota", "LaMalfa", "Lamborn",
        "Langworthy", "Latta", "LaTurner", "Lawler", "Lee", "Lesko", "Letlow", "Lopez",
        "Loudermilk", "Lucas", "Luetkemeyer", "Luna", "Luttrell", "Mace", "Malliotakis",
        "Maloy", "Mann", "Massie", "Mast", "McCarthy", "McCaul", "McClain", "McClintock",
        "McCormick", "McHenry", "McMorris Rodgers", "Meuser", "Miller-Meeks", "Miller", "Mills", "Molinaro",
        "Moolenaar", "Mooney", "Moore", "Moran", "Moylan", "Murphy", "Nehls", "Newhouse",
        "Norman", "Nunn", "Obernolte", "Ogles", "Owens", "Palmer", "Pence", "Perry", "Pfluger",
        "Posey", "Radewagen", "Reschenthaler", "Rogers", "Rose", "Rosendale", "Rouzer", "Roy", "Rulli",
        "Rutherford", "Salazar", "Santos", "Scalise", "Schweikert", "Scott", "Self", "Sessions",
        "Simpson", "Smith", "Smucker", "Spartz", "Stauber", "Steel", "Stefanik", "Steil",
        "Steube", "Stewart", "Strong", "Tenney", "Thompson", "Tiffany", "Timmons", "Turner",
        "Valadao", "Van Drew", "Van Duyne", "Van Orden", "Wagner", "Walberg", "Waltz", "Weber",
        "Webster", "Wenstrup", "Westerman", "Wied", "Williams", "Wilson", "Wittman", "Womack",
        "Yakym", "Zinke"
    ]
    return congress_118_lastnames

def get_data_directory():
    return "Data/First Session"

def get_dems():
    dems = [
        "adams", "aguilar", "allred", "amo", "auchincloss", "balint", "barragan", "beatty", "beyer",
        "blumenauer","bowman",  "bonamici","bluntrochester", "boyle", "brownley", "brown", "bera", "bishopGA",
        "budzinski", "bush", "caraveo", "carbajal", "cardenas", "carson", "carterLA", "cartwright",
        "casar", "case", "casten", "castor", "castro", "cherfilusmccormick", "chu", "cicilline",
        "clarkMA", "clarkeNY", "cleaver", "clyburn", "cohen", "connolly", "correa", "costa", "courtney",
        "craig", "crockett", "crow", "cuellar", "davids","davidscottGA","davisil", "davisnc", "deanpa", "degette",
        "delauro", "delbene", "deluzio", "desaulnier", "dingell", "doggett", "escobar", "eshoo",
        "espaillat", "evans", "fletcher", "foster", "foushee", "loisfrankel", "frost", "gallego",
        "garamendi", "garciail", "garciaca", "garciatx", "golden", "goldman", "gomez", "gonzalezTX", "gluesenkampperez",
        "gottheimer", "greenLA", "grijalva", "harder", "hayes", "himes", "higginsNY", "horsford", "houlahan",
        "hoyer", "hoyle", "huffman", "ivey", "jacksonlee", "jacksonnc", "jacksonil", "jacobs",
        "jayapal", "jeffries", "johnsonga", "kamlagerdove", "kaptur", "keating", "kelly",
        "kennedy", "khanna", "kildee", "kilmer", "kimNJ", "KimCA","krishnamoorthi", "kuster", "landsman",
        "larsenWA", "larsonCN", "leeca", "leepa", "leenv", "legerfernandez", "levin", "lieu", "lofgren",
        "lynch", "magaziner", "manning", "matsui", "mcbath", "mcclellan", "mccollum", "mcgovern", "mcgarvey",
        "meeks", "menendez", "meng", "mfume", "mooreWI", "morelle", "moskowitz", "moulton",
        "mrvan", "mullin", "nadler", "napolitano", "neal", "neguse", "nickel", "norcross",
        "norton", "ocasiocortez", "omar", "pallone", "panetta", "pappas", "pascrell",
        "payne", "pelosi", "peltola", "perez", "peters", "pettersen", "phillips", "pingree",
        "plaskett", "pocan", "porter", "pressley", "quigley", "ramirez", "raskin", "ross",
        "ruiz", "ruppersberger", "ryan", "sablan", "salinas", "sanchez", "sarbanes",
        "scanlon", "schakowsky", "schiff", "schneider", "scholten", "schrier", "scottga",
        "scottva", "sewell", "sherman", "sherrill", "slotkin", "smithwa", "sorensen", "soto",
        "spanberger", "stansbury", "stanton", "stevens", "strickland", "suozzi", "swalwell",
        "sykes", "takano", "thanedar", "thompsonms", "thompsonca", "titus", "tlaib",
        "tokuda", "tonko", "torresca", "torresny", "trahan", "trone", "underwood", "vargas",
        "vasquez", "veasey", "velazquez", "wassermanschultz", "waters", "watsoncoleman",
        "wexton", "wild", "williamsga", "wilsonfl"
    ]
    return dems

def get_repubs():
    reps = [
        "aderholt", "alford", "allen", "amodei", "armstrong", "arrington", "austinscottga","babin", "bacon",
        "baird", "balderson", "banks", "barr", "beanFL", "bentz", "bergman", "bice", "biggs",
        "bilirakis", "bishopNC", "boebert", "bost", "brecheen", "buchanan", "buck", "bucshon",
        "burchett", "burgess", "burlison", "calvert", "cammack", "carey", "carl", "carterga", "castrotx",
        "cartertx", "chavezderemer", "ciscomani", "cline", "cloud", "clyde", "cole", "collins",
        "comer", "crane", "crawford", "crenshaw", "curtis","cscottfranklin", "desposito", "davidson",
        "delacruz", "desjarlais", "diazbalart", "donalds", "duarte", "duncan", "dunn",
        "edwards", "ellzey", "emmer", "estes", "ezell", "fallon", "feenstra", "ferguson",
        "finstad", "fischbach", "fitzgerald", "fitzpatrick", "fleischmann", "flood", "fong",
        "foxx", "franklin", "fry", "fulcher", "gaetz", "gallagher", "garbarino", "mikegarciaCA",
        "gimenez", "tonygonzalesTX", "gonzalezcolon", "good", "gooden", "gosar", "granger", "gravesla",
        "gravesmo", "greenTN", "greenTX","greene", "griffith", "grothman", "guest", "guthrie", "hageman", "hudson",
        "harris", "harshbarger", "hern", "higginsla", "hill", "hinson", "houchin", "huizenga",
        "hunt", "issa", "jacksontx", "james", "johnsonoh", "johnsonsd", "johnsonla", "jordan",
        "joyceoh", "joycepa", "kean", "kellypa", "kellyms", "kiggans", "kiley", "kim",
        "kustoff", "lahood", "lalota", "lamalfa", "lamborn", "langworthy", "latta", "laturner",
        "lawler", "leefl", "lesko", "letlow", "lopez", "loudermilk", "lucas", "luetkemeyer", "luna",
        "luttrell", "mace", "malliotakis", "maloy", "mann", "massie", "mast", "mccarthy", "mccaul",
        "mcclain", "mcclintock", "mccormick", "mchenry", "mcmorrisrodgers", "meuser", "millerwv",
        "milleril", "milleroh", "millermeeks", "mills", "molinaro", "moolenaar", "mooney", "mooreal",
        "mooreut", "moran", "moylan", "murphy", "nehls", "newhouse", "norman", "nunn", "obernolte",
        "ogles", "owens", "palmer", "pence", "perry", "pfluger", "posey", "radewagen", "reschenthaler", "rogersky",
        "rogersal", "rodgerswa", "rose", "rosendale", "rouzer", "roy", "rulli", "rutherford", "salazar", "santos",
        "scalise", "schweikert", "scott", "scottfranklinfl","self", "sessions", "simpson", "smithne", "smithnj",
        "smithmo", "smucker", "spartz", "stauber", "steel", "stefanik", "steil", "steube",
        "stewart", "strong", "tenney", "thompsonpa", "tiffany", "timmons", "turner", "valadao",
        "vandrew", "vanduyne", "vanorden", "wagner", "walberg", "waltz", "weber", "webster",
        "wenstrup", "westerman", "wied", "williamsny", "williamstx", "wilsonsc", "wittman", "womack",
        "yakym", "zinke"
    ]
    return reps

def get_freedom_caucus():
    freedom_caucus = [
        "bishop","boebert", "breechan", "crane", "desjarlais","duncan","fulcher","gaetz",
        "good","gosar","greenTN","harris","higginsLA","jordan","lesko","luna","millerIL",
        "mooreAL","ogles","perry","rosendale","roy","weber","biggs","burlison","cline",
        "cloud","clyde","donalds","griffith","harshbarger","norman","tiffany","self",
        "buck","greene","davidson","schweikert"
    ]
    return freedom_caucus

def get_stop_phrases():
    extra_stop_phrases = ["Mr. Speaker", "madam chair", "mister chair", "I reserve my time"]
    return extra_stop_phrases

def get_mcs_states():
    mcs_states = {'peltola': 'AK', 'carl': 'AL', 'mooreal': 'AL', 'rogersal': 'AL', 'aderholt': 'AL', 'strong': 'AL', 'palmer': 'AL',
     'sewell': 'AL', 'crawford': 'AR', 'hill': 'AR', 'womack': 'AR', 'westerman': 'AR', 'schweikert': 'AZ',
     'crane': 'AZ', 'gallego': 'AZ', 'stanton': 'AZ', 'biggs': 'AZ', 'ciscomani': 'AZ', 'grijalva': 'AZ', 'lesko': 'AZ',
     'gosar': 'AZ', 'lamalfa': 'CA', 'huffman': 'CA', 'kiley': 'CA', 'thompsonca': 'CA', 'mcclintock': 'CA',
     'bera': 'CA', 'matsui': 'CA', 'garamendi': 'CA', 'harder': 'CA', 'desaulnier': 'CA', 'pelosi': 'CA', 'leeca': 'CA',
     'duarte': 'CA', 'swalwell': 'CA', 'mullin': 'CA', 'eshoo': 'CA', 'khanna': 'CA', 'lofgren': 'CA', 'panetta': 'CA',
     'mccarthy': 'CA', 'fong': 'CA', 'costa': 'CA', 'valadao': 'CA', 'obernolte': 'CA', 'carbajal': 'CA', 'ruiz': 'CA',
     'brownley': 'CA', 'mikegarciaca': 'CA', 'chu': 'CA', 'cardenas': 'CA', 'schiff': 'CA', 'napolitano': 'CA',
     'sherman': 'CA', 'aguilar': 'CA', 'gomez': 'CA', 'torresca': 'CA', 'lieu': 'CA', 'kamlagerdove': 'CA',
     'sanchez': 'CA', 'takano': 'CA', 'KimCA': 'CA', 'calvert': 'CA', 'waters': 'CA', 'barragan': 'CA', 'steel': 'CA',
     'correa': 'CA', 'porter': 'CA', 'issa': 'CA', 'levin': 'CA', 'peters': 'CA', 'jacobs': 'CA', 'vargas': 'CA',
     'degette': 'CO', 'neguse': 'CO', 'boebert': 'CO', 'buck': 'CO', 'lopez': 'CO', 'lamborn': 'CO', 'crow': 'CO',
     'pettersen': 'CO', 'caraveo': 'CO', 'Larson': 'CT', 'courtney': 'CT', 'delauro': 'CT', 'himes': 'CT',
     'hayes': 'CT', 'bluntrochester': 'DE', 'gaetz': 'FL', 'dunn': 'FL', 'cammack': 'FL', 'beanFL': 'FL',
     'rutherford': 'FL', 'waltz': 'FL', 'mills': 'FL', 'posey': 'FL', 'soto': 'FL', 'frost': 'FL', 'webster': 'FL',
     'bilirakis': 'FL', 'luna': 'FL', 'castorfl': 'FL', 'leefl': 'FL', 'buchanan': 'FL', 'steube': 'FL',
     'franklin': 'FL', 'donalds': 'FL', 'cherfilusmccormick': 'FL', 'mast': 'FL', 'Frankel': 'FL', 'moskowitz': 'FL',
     'wilsonfl': 'FL', 'wassermanschultz': 'FL', 'diazbalart': 'FL', 'salazar': 'FL', 'gimenez': 'FL', 'carterga': 'GA',
     'bishopGA': 'GA', 'ferguson': 'GA', 'johnsonga': 'GA', 'williamsga': 'GA', 'mccormick': 'GA', 'mcbath': 'GA',
     'scottga': 'GA', 'clyde': 'GA', 'collins': 'GA', 'loudermilk': 'GA', 'allen': 'GA', 'greene': 'GA', 'case': 'HI',
     'tokuda': 'HI', 'millermeeks': 'IA', 'hinson': 'IA', 'nunnIA': 'IA', 'feenstra': 'IA', 'fulcher': 'ID',
     'simpson': 'ID', 'jacksonil': 'IL', 'kelly': 'IL', 'ramirez': 'IL', 'garciail': 'IL', 'quigley': 'IL',
     'casten': 'IL', 'davisil': 'IL', 'krishnamoorthi': 'IL', 'schakowsky': 'IL', 'schneider': 'IL', 'foster': 'IL',
     'bost': 'IL', 'budzinski': 'IL', 'underwood': 'IL', 'milleril': 'IL', 'lahood': 'IL', 'sorensen': 'IL',
     'mrvan': 'IN', 'yakym': 'IN', 'banks': 'IN', 'baird': 'IN', 'spartz': 'IN', 'pence': 'IN', 'carson': 'IN',
     'bucshon': 'IN', 'houchin': 'IN', 'mann': 'KS', 'laturner': 'KS', 'davids': 'KS', 'estes': 'KS', 'comer': 'KY',
     'guthrie': 'KY', 'mcgarvey': 'KY', 'massie': 'KY', 'rogersky': 'KY', 'barr': 'KY', 'scalise': 'LA',
     'carterLA': 'LA', 'higginsla': 'LA', 'johnsonla': 'LA', 'letlow': 'LA', 'gravesla': 'LA', 'neal': 'MA',
     'mcgovern': 'MA', 'trahan': 'MA', 'auchincloss': 'MA', 'clarkMA': 'MA', 'moulton': 'MA', 'pressley': 'MA',
     'lynch': 'MA', 'keating': 'MA', 'harris': 'MD', 'ruppersberger': 'MD', 'sarbanes': 'MD', 'ivey': 'MD',
     'hoyer': 'MD', 'trone': 'MD', 'mfume': 'MD', 'raskin': 'MD', 'pingree': 'ME', 'golden': 'ME', 'bergman': 'MI',
     'moolenaar': 'MI', 'scholten': 'MI', 'huizenga': 'MI', 'walberg': 'MI', 'dingell': 'MI', 'slotkin': 'MI',
     'kildee': 'MI', 'mcclain': 'MI', 'james': 'MI', 'stevens': 'MI', 'tlaib': 'MI', 'thanedar': 'MI', 'finstad': 'MN',
     'craig': 'MN', 'phillips': 'MN', 'mccollum': 'MN', 'omar': 'MN', 'emmer': 'MN', 'fischbach': 'MN', 'stauber': 'MN',
     'bush': 'MO', 'wagner': 'MO', 'luetkemeyer': 'MO', 'alford': 'MO', 'cleaver': 'MO', 'gravesmo': 'MO',
     'burlison': 'MO', 'smithmo': 'MO', 'kellyms': 'MS', 'thompsonms': 'MS', 'guest': 'MS', 'ezell': 'MS',
     'zinke': 'MT', 'rosendale': 'MT', 'davisnc': 'NC', 'ross': 'NC', 'murphy': 'NC', 'foushee': 'NC', 'foxx': 'NC',
     'manning': 'NC', 'rouzer': 'NC', 'bishopNC': 'NC', 'hudson': 'NC', 'mchenry': 'NC', 'edwards': 'NC', 'adams': 'NC',
     'nickel': 'NC', 'jacksonnc': 'NC', 'armstrong': 'ND', 'flood': 'NE', 'bacon': 'NE', 'smithne': 'NE',
     'pappas': 'NH', 'kuster': 'NH', 'norcross': 'NJ', 'vandrew': 'NJ', 'kimNJ': 'NJ', 'smithnj': 'NJ',
     'gottheimer': 'NJ', 'pallone': 'NJ', 'kean': 'NJ', 'menendez': 'NJ', 'pascrell': 'NJ', 'payne': 'NJ',
     'McIver': 'NJ', 'sherrill': 'NJ', 'watsoncoleman': 'NJ', 'stansbury': 'NM', 'vasquez': 'NM',
     'legerfernandez': 'NM', 'titus': 'NV', 'amodei': 'NV', 'leenv': 'NV', 'horsford': 'NV', 'lalota': 'NY',
     'garbarino': 'NY', 'santos': 'NY', 'suozzi': 'NY', "D'Esposito": 'NY', 'meeks': 'NY', 'meng': 'NY',
     'velazquez': 'NY', 'jeffries': 'NY', 'clarkeNY': 'NY', 'goldman': 'NY', 'malliotakis': 'NY', 'nadler': 'NY',
     'espaillat': 'NY', 'ocasiocortez': 'NY', 'torresny': 'NY', 'bowman': 'NY', 'lawler': 'NY', 'ryan': 'NY',
     'molinaro': 'NY', 'tonko': 'NY', 'stefanik': 'NY', 'williamsny': 'NY', 'langworthy': 'NY', 'tenney': 'NY',
     'morelle': 'NY', 'higginsNY': 'NY', 'kennedy': 'NY', 'landsman': 'OH', 'wenstrup': 'OH', 'beatty': 'OH',
     'jordan': 'OH', 'latta': 'OH', 'johnsonoh': 'OH', 'rulli': 'OH', 'milleroh': 'OH', 'davidson': 'OH',
     'kaptur': 'OH', 'turner': 'OH', 'brown': 'OH', 'balderson': 'OH', 'sykes': 'OH', 'joyceoh': 'OH', 'carey': 'OH',
     'hern': 'OK', 'brecheen': 'OK', 'lucas': 'OK', 'cole': 'OK', 'bice': 'OK', 'bonamici': 'OR', 'bentz': 'OR',
     'blumenauer': 'OR', 'hoyle': 'OR', 'chavezderemer': 'OR', 'salinas': 'OR', 'fitzpatrick': 'PA', 'boyle': 'PA',
     'evans': 'PA', 'deanpa': 'PA', 'scanlon': 'PA', 'houlahan': 'PA', 'wild': 'PA', 'cartwright': 'PA', 'meuser': 'PA',
     'perry': 'PA', 'smucker': 'PA', 'leepa': 'PA', 'joycepa': 'PA', 'reschenthaler': 'PA', 'thompsonpa': 'PA',
     'kellypa': 'PA', 'deluzio': 'PA', 'cicilline': 'RI', 'amo': 'RI', 'magaziner': 'RI', 'mace': 'SC',
     'wilsonsc': 'SC', 'duncan': 'SC', 'timmons': 'SC', 'norman': 'SC', 'clyburn': 'SC', 'fry': 'SC', 'johnsonsd': 'SD',
     'harshbarger': 'TN', 'burchett': 'TN', 'fleischmann': 'TN', 'desjarlais': 'TN', 'ogles': 'TN', 'rose': 'TN',
     'greenTN': 'TN', 'kustoff': 'TN', 'cohen': 'TN', 'moran': 'TX', 'crenshaw': 'TX', 'self': 'TX', 'fallon': 'TX',
     'gooden': 'TX', 'ellzey': 'TX', 'fletcher': 'TX', 'luttrell': 'TX', 'greenTX': 'TX', 'mccaul': 'TX',
     'pfluger': 'TX', 'granger': 'TX', 'jacksontx': 'TX', 'weberTX': 'TX', 'delacruz': 'TX', 'escobar': 'TX',
     'sessions': 'TX', 'jacksonlee': 'TX', 'cartertx': 'TX', 'arrington': 'TX', 'castrotx': 'TX', 'roy': 'TX',
     'nehls': 'TX', 'Gonzales': 'TX', 'vanduyne': 'TX', 'williamstx': 'TX', 'burgess': 'TX', 'cloud': 'TX',
     'cuellar': 'TX', 'garciatx': 'TX', 'crockett': 'TX', 'allred': 'TX', 'veasey': 'TX', 'gonzalezTX': 'TX',
     'casar': 'TX', 'babin': 'TX', 'doggett': 'TX', 'hunt': 'TX', 'mooreut': 'UT', 'stewart': 'UT', 'maloy': 'UT',
     'curtis': 'UT', 'owens': 'UT', 'wittman': 'VA', 'kiggans': 'VA', 'scottva': 'VA', 'mcclellan': 'VA',
     'good': 'VA', 'cline': 'VA', 'spanberger': 'VA', 'beyer': 'VA', 'griffith': 'VA', 'wexton': 'VA', 'connolly': 'VA',
     'balint': 'VT', 'delbene': 'WA', 'larsenWA': 'WA', 'gluesenkampperez': 'WA', 'newhouse': 'WA',
     'mcmorrisrodgers': 'WA', 'kilmer': 'WA', 'jayapal': 'WA', 'schrier': 'WA', 'smithwa': 'WA', 'strickland': 'WA',
     'steil': 'WI', 'pocan': 'WI', 'vanorden': 'WI', 'mooreWI': 'WI', 'fitzgerald': 'WI', 'grothman': 'WI',
     'tiffany': 'WI', 'gallagher': 'WI', 'wied': 'WI', 'millerwv': 'WV', 'mooney': 'WV', 'hageman': 'WY'}

    return mcs_states

def state_regions():
    us_state_postal_codes = [
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
        "DC", "PR", "GU", "AS", "MP", "VI"
    ]

    north = ["CT", "DE", "ME", "MD", "MA","NH", "NJ", "NY",  "PA", "RI", "VT", "VI", "DC"]
    south = ["AL", "AR", "FL","GA","KY", "LA", "MS", "NC", "OK", "SC", "TN", "TX", "WV"]
    midwest = ["IL", "IN", "MI", "MN", "MO", "OH", "WI"]
    prairie = ["IA", "KS", "NE", "ND", "SD"]
    west = ["ID","MT", "WY", "CO"]
    southwest = ["AZ", "NV", "NM", "UT"]
    west_coast = ["CA", "OR", "WA"]
    pacific = ["AK","HI", "GU", "AS", "MP"]
    atlantic = ["PR", "VI"]

    return north, south, midwest, prairie, west, southwest, west_coast, pacific, atlantic

def states_to_regions():
    mcs_regions = get_mcs_states()
    north, south, midwest, prairie, west, southwest, west_coast, pacific, atlantic = state_regions()

    #change this to condense any regions :)
    region_map = {}
    for state in north:
        region_map[state] = "north"
    for state in south:
        region_map[state] = "south"
    for state in midwest:
        region_map[state] = "midwest"
    for state in prairie:
        region_map[state] = "prairie"
    for state in west:
        region_map[state] = "west"
    for state in southwest:
        region_map[state] = "southwest"
    for state in west_coast:
        region_map[state] = "west_coast"
    for state in pacific:
        region_map[state] = "pacific"
    for state in atlantic:
        region_map[state] = "atlantic"

    for mc, state_code in mcs_regions.items():
        mcs_regions[mc] = region_map.get(state_code, "unknown")

    return mcs_regions


def get_mcs_regions():
    #some of these are odd due to naming differences in files
    #that's honestly the most difficult part of this project
    regions_dict = {
        # A
        'adams': 'south', 'aderholt': 'south', 'aguilar': 'west_coast', 'allred': 'south',
        'alford': 'midwest', 'allen': 'south', 'amodei': 'southwest', 'armstrong': 'prairie',
        'arrington': 'south', 'auchincloss': 'north', 'amo': 'north', 'austinscottga': 'south',

        # B
        'babin': 'south', 'bacon': 'prairie', 'baird': 'midwest', 'balint': 'north',
        'balderson': 'midwest', 'banks': 'midwest', 'barr': 'south', 'barragan': 'west_coast',
        'beanfl': 'south', 'beatty': 'midwest', 'bera': 'west_coast', 'bergman': 'midwest',
        'beyer': 'south', 'bice': 'south', 'biggs': 'southwest', 'bilirakis': 'south',
        'bishopga': 'south', 'bishopnc': 'south', 'blumenauer': 'west_coast',
        'bluntrochester': 'north', 'boebert': 'west', 'bonamici': 'west_coast',
        'bost': 'midwest', 'bowman': 'north', 'boyle': 'north', 'brecheen': 'south',
        'brown': 'midwest', 'brownley': 'west_coast', 'budzinski': 'midwest',
        'buck': 'west', 'bucshon': 'midwest', 'burchett': 'south', 'burgess': 'south',
        'burlison': 'midwest', 'bush': 'midwest',

        # C
        'cammack': 'south', 'caraveo': 'west', 'carbajal': 'west_coast', 'cardenas': 'west_coast',
        'carey': 'midwest', 'carl': 'south', 'carson': 'midwest', 'carter': 'south',
        'carterga': 'south', 'carterla': 'south', 'cartertx': 'south', 'cartwright': 'north',
        'casar': 'south', 'case': 'pacific', 'casten': 'midwest', 'castor': 'south',
        'castorfl': 'south', 'castro': 'south', 'castrotx': 'south', 'chavezderemer': 'west_coast',
        'cherfilusmccormick': 'south', 'chu': 'west_coast', 'cicilline': 'north',
        'ciscomani': 'southwest', 'clarkma': 'north', 'clarkeny': 'north',
        'clarke': 'north', 'cleaver': 'midwest', 'cline': 'south', 'cloud': 'south',
        'clyburn': 'south', 'clyde': 'south', 'cohen': 'south', 'cole': 'south',
        'collins': 'south', 'comer': 'south', 'connolly': 'south', 'correa': 'west_coast',
        'costa': 'west_coast', 'courtney': 'north', 'craig': 'midwest',
        'crane': 'southwest', 'crawford': 'south', 'crenshaw': 'south', 'crockett': 'south',
        'crow': 'west', 'cscottfranklin': 'south', 'cuellar': 'south', 'curtis': 'southwest',

        # D
        'davidson': 'midwest', 'davids': 'prairie', 'davisil': 'midwest', 'davisnc': 'south',
        'davidscottga': 'south', 'deanpa': 'north', 'degette': 'west', 'delacruz': 'south',
        'delauro': 'north', 'delbene': 'west_coast', 'deluzio': 'north', 'desaulnier': 'west_coast',
        'desjarlais': 'south', 'desposito': 'north', 'diazbalart': 'south', 'dingell': 'midwest',
        'doggett': 'south', 'donalds': 'south', 'duarte': 'west_coast', 'duncan': 'south',
        'dunn': 'south',

        # E-F
        'edwards': 'south', 'ellzey': 'south', 'emmer': 'midwest', 'escobar': 'south',
        'eshoo': 'west_coast', 'espaillat': 'north', 'estes': 'prairie', 'evans': 'north',
        'ezell': 'south', 'fallon': 'south', 'feenstra': 'prairie', 'ferguson': 'south',
        'finstad': 'midwest', 'fischbach': 'midwest', 'fitzgerald': 'midwest',
        'fitzpatrick': 'north', 'fleischmann': 'south', 'fletcher': 'south', 'flood': 'prairie',
        'fong': 'west_coast', 'foster': 'midwest', 'foushee': 'south', 'foxx': 'south',
        'franklin': 'south', 'frankel': 'south', 'loisfrankel': 'south', 'frost': 'south',
        'fry': 'south', 'fulcher': 'west',

        # G
        'gaetz': 'south', 'gallagher': 'midwest', 'gallego': 'southwest', 'garamendi': 'west_coast',
        'garbarino': 'north', 'garciail': 'midwest', 'garciaca': 'west_coast', 'garciatx': 'south',
        'gimenez': 'south', 'gluesenkampperez': 'west_coast', 'golden': 'north',
        'goldman': 'north', 'gomez': 'west_coast', 'good': 'south', 'gooden': 'south',
        'gonzales': 'south', 'gonzalezcolon': 'atlantic', 'gonzaleztx': 'south',
        'tonygonzaleztx': 'south', 'gosar': 'southwest', 'gottheimer': 'north',
        'granger': 'south', 'gravesla': 'south', 'gravesmo': 'midwest',
        'greenla': 'south', 'greentn': 'south', 'greentx': 'south', 'greene': 'south',
        'griffith': 'south', 'grijalva': 'southwest', 'grothman': 'midwest', 'guest': 'south',
        'guthrie': 'south',

        # H
        'hageman': 'west', 'harder': 'west_coast', 'harshbarger': 'south', 'harris': 'north',
        'hayes': 'north', 'hern': 'south', 'higgins': 'south', 'higginsla': 'south',
        'higginsny': 'north', 'hill': 'south', 'himes': 'north',
        'hinson': 'prairie', 'horsford': 'southwest', 'houlahan': 'north', 'houchin': 'midwest',
        'hoyer': 'north', 'hoyle': 'west_coast', 'hudson': 'south', 'huffman': 'west_coast',
        'huizenga': 'midwest', 'hunt': 'south',

        # I-J
        'issa': 'west_coast', 'ivey': 'north', 'jacksonnc': 'south', 'jacksonlee': 'south',
        'jacksonil': 'midwest', 'jacksontx': 'south', 'jacobs': 'west_coast', 'james': 'midwest',
        'jayapal': 'west_coast', 'jeffries': 'north', 'johnsonga': 'south', 'johnsonla': 'south',
        'johnsonoh': 'midwest', 'johnsonsd': 'prairie', 'jordan': 'midwest', 'joyceoh': 'midwest',
        'joycepa': 'north',

        # K
        'kamlagerdove': 'west_coast', 'kaptur': 'midwest', 'kean': 'north', 'keating': 'north',
        'kelly': 'midwest', 'kellyms': 'south', 'kellypa': 'north', 'kennedy': 'north',
        'khanna': 'west_coast', 'kiggans': 'south', 'kildee': 'midwest', 'kiley': 'west_coast',
        'kilmer': 'west_coast', 'kim': 'west_coast', 'kimca': 'west_coast', 'kimnj': 'north',
        'krishnamoorthi': 'midwest', 'kuster': 'north', 'kustoff': 'south',

        # L
        'lahood': 'midwest', 'lalota': 'north', 'lamalfa': 'west_coast', 'lamborn': 'west',
        'landsman': 'midwest', 'langworthy': 'north', 'larsen': 'west_coast', 'larsenwa': 'west_coast',
        'larson': 'north', 'larsoncn': 'north', 'latta': 'midwest', 'laturner': 'prairie',
        'lawler': 'north', 'leeca': 'west_coast', 'leefl': 'south', 'leenv': 'southwest',
        'leepa': 'north', 'legerfernandez': 'southwest', 'lesko': 'southwest', 'letlow': 'south',
        'levin': 'west_coast', 'lieu': 'west_coast', 'lofgren': 'west_coast', 'lopez': 'west',
        'loudermilk': 'south', 'lucas': 'south', 'luetkemeyer': 'midwest', 'luna': 'south',
        'luttrell': 'south', 'lynch': 'north',

        # M
        'mace': 'south', 'magaziner': 'north', 'malliotakis': 'north', 'maloy': 'southwest',
        'mann': 'prairie', 'manning': 'south', 'massie': 'south', 'mast': 'south',
        'matsui': 'west_coast', 'mcbath': 'south', 'mccaul': 'south', 'mcclellan': 'south',
        'mcclain': 'midwest', 'mcclintock': 'west_coast', 'mccollum': 'midwest', 'mccormick': 'south',
        'mcgarvey': 'south', 'mcgovern': 'north', 'mchenry': 'south', 'mciver': 'north',
        'mcmorrisrodgers': 'west_coast', 'meeks': 'north', 'menendez': 'north', 'meng': 'north',
        'meuser': 'north', 'mfume': 'north', 'milleril': 'midwest', 'millermeeks': 'prairie',
        'milleroh': 'midwest', 'millerwv': 'south', 'mills': 'south', 'mikegarciaca': 'west_coast',
        'molinaro': 'north', 'moolenaar': 'midwest', 'mooney': 'south', 'mooreal': 'south',
        'mooreut': 'southwest', 'moorewi': 'midwest', 'moran': 'south', 'morelle': 'north',
        'moskowitz': 'south', 'moulton': 'north', 'moylan': 'midwest', 'mrvan': 'midwest',
        'mullin': 'west_coast', 'murphy': 'south',

        # N
        'nadler': 'north', 'napolitano': 'west_coast', 'neal': 'north', 'neguse': 'west',
        'nehls': 'south', 'newhouse': 'west_coast', 'nickel': 'south', 'norcross': 'north',
        'norman': 'south', 'norton': 'north', 'nunn': 'prairie', 'nunnia': 'prairie',

        # O
        'obernolte': 'west_coast', 'ocasiocortez': 'north', 'ogles': 'south', 'omar': 'midwest',
        'owens': 'southwest',

        # P
        'pallone': 'north', 'palmer': 'south', 'panetta': 'west_coast', 'pappas': 'north',
        'pascrell': 'north', 'payne': 'north', 'pelosi': 'west_coast', 'peltola': 'pacific',
        'pence': 'midwest', 'perez': 'west_coast', 'perry': 'north', 'peters': 'west_coast',
        'pettersen': 'west', 'pfluger': 'south', 'phillips': 'midwest', 'pingree': 'north',
        'plaskett': 'atlantic', 'pocan': 'midwest', 'porter': 'west_coast', 'posey': 'south',
        'pressley': 'north',

        # Q-R
        'quigley': 'midwest', 'radewagen': 'pacific', 'ramirez': 'midwest', 'raskin': 'north',
        'reschenthaler': 'north', 'rodgerswa': 'west_coast', 'rogersal': 'south',
        'rogersky': 'south', 'rose': 'south', 'rosendale': 'west', 'ross': 'south',
        'rouzer': 'south', 'roy': 'south', 'ruiz': 'west_coast', 'rulli': 'midwest',
        'ruppersberger': 'north', 'rutherford': 'south', 'ryan': 'north',

        # S
        'sablan': 'pacific', 'salazar': 'south', 'salinas': 'west_coast', 'sanchez': 'west_coast',
        'santos': 'north', 'sarbanes': 'north', 'scalise': 'south', 'scanlon': 'north',
        'schakowsky': 'midwest', 'schiff': 'west_coast', 'schneider': 'midwest', 'scholten': 'midwest',
        'schrier': 'west_coast', 'schweikert': 'southwest', 'scott': 'south', 'scottfranklinfl': 'south',
        'scottga': 'south', 'scottva': 'south', 'self': 'south', 'sessions': 'south',
        'sewell': 'south', 'sherrill': 'north', 'sherman': 'west_coast', 'simpson': 'west',
        'slotkin': 'midwest', 'smithmo': 'midwest', 'smithne': 'prairie', 'smithnj': 'north',
        'smithwa': 'west_coast', 'smucker': 'north', 'sorensen': 'midwest', 'soto': 'south',
        'spanberger': 'south', 'spartz': 'midwest', 'stansbury': 'southwest', 'stanton': 'southwest',
        'stauber': 'midwest', 'steel': 'west_coast', 'stefanik': 'north', 'steil': 'midwest',
        'steube': 'south', 'stevens': 'midwest', 'stewart': 'southwest', 'strickland': 'west_coast',
        'strong': 'south', 'suozzi': 'north', 'swalwell': 'west_coast', 'sykes': 'midwest',

        # T
        'takano': 'west_coast', 'tenney': 'north', 'thanedar': 'midwest', 'thompsonca': 'west_coast',
        'thompsonms': 'south', 'thompsonpa': 'north', 'tiffany': 'midwest', 'timmons': 'south',
        'titus': 'southwest', 'tlaib': 'midwest', 'tokuda': 'pacific', 'tonko': 'north',
        'torres': 'west_coast', 'torresca': 'west_coast', 'torresny': 'north', 'trahan': 'north',
        'trone': 'north', 'turner': 'midwest',

        # U-V
        'underwood': 'midwest', 'valadao': 'west_coast', 'vandrew': 'north', 'vanduyne': 'south',
        'vanorden': 'midwest', 'vargas': 'west_coast', 'vasquez': 'southwest', 'veasey': 'south',
        'velazquez': 'north',

        # W
        'wagner': 'midwest', 'walberg': 'midwest', 'waltz': 'south', 'wassermanschultz': 'south',
        'waters': 'west_coast', 'watsoncoleman': 'north', 'webertx': 'south', 'weber': 'south',
        'webster': 'south', 'wenstrup': 'midwest', 'westerman': 'south', 'wexton': 'south',
        'wied': 'midwest', 'wild': 'north', 'williamsga': 'south', 'williamsny': 'north',
        'williamstx': 'south', 'wilsonfl': 'south', 'wilsonsc': 'south', 'wittman': 'south',
        'womack': 'south',

        # X-Y-Z
        'yakym': 'midwest', 'zinke': 'west'
    }

    return {k.lower(): v for k, v in regions_dict.items()}


def get_female_mcs():
    female_mcs = ['peltola', 'sewell', 'lesko', 'matsui', 'pelosi', 'leeca', 'eshoo', 'lofgren', 'brownley', 'chu',
               'napolitano', 'torresca', 'kamlagerdove', 'sanchez', 'KimCA', 'waters', 'barragan', 'steel', 'porter',
               'jacobs', 'degette', 'boebert', 'pettersen', 'caraveo', 'delauro', 'hayes', 'bluntrochester', 'cammack',
               'luna', 'castor', 'leefl', 'cherfilusmccormick', 'Frankel', 'wilsonfl', 'wassermanschultz', 'salazar',
               'williamsga', 'mcbath', 'greene', 'tokuda', 'millermeeks', 'hinson', 'kelly', 'ramirez', 'schakowsky',
               'budzinski', 'underwood', 'milleril', 'spartz', 'houchin', 'davids', 'letlow', 'trahan', 'clarkMA',
               'pressley', 'pingree', 'scholten', 'dingell', 'slotkin', 'mcclain', 'stevens', 'tlaib', 'craig',
               'mccollum', 'omar', 'fischbach', 'bush', 'wagner', 'ross', 'foushee', 'foxx', 'manning', 'adams',
               'kuster', 'McIver', 'sherrill', 'watsoncoleman', 'stansbury', 'legerfernandez', 'titus', 'leenv', 'meng',
               'velazquez', 'clarkeNY', 'malliotakis', 'ocasiocortez', 'stefanik', 'tenney', 'beatty', 'kaptur',
               'brown', 'sykes', 'bice', 'bonamici', 'hoyle', 'chavezderemer', 'salinas', 'deanpa', 'scanlon',
               'houlahan', 'wild', 'leepa', 'mace', 'harshbarger', 'fletcher', 'granger', 'delacruz', 'escobar',
               'jacksonlee', 'cartertx', 'vanduyne', 'garciatx', 'crockett', 'maloy', 'kiggans', 'mcclellan',
               'spanberger', 'wexton', 'balint', 'delbene', 'gluesenkampperez', 'perez', 'mcmorrisrodgers', 'jayapal', 'schrier',
               'strickland', 'mooreWI', 'millerwv', 'hageman', 'gonzalezcolon', 'norton', 'loisfrankel', 'plaskett', 'castorfl',
                  'radewagen', ]
    return female_mcs

def get_male_mcs():
    male_mcs = ['carl', 'mooreal', 'rogersal', 'aderholt', 'strong', 'palmer', 'crawford', 'hill', 'womack', 'westerman',
               'schweikert', 'crane', 'gallego', 'stanton', 'biggs', 'ciscomani', 'grijalva', 'gosar', 'lamalfa',
               'huffman', 'kiley', 'thompsonca', 'mcclintock', 'bera', 'garamendi', 'harder', 'desaulnier', 'duarte',
               'swalwell', 'mullin', 'khanna', 'panetta', 'mccarthy', 'fong', 'costa', 'valadao', 'obernolte',
               'carbajal', 'ruiz', 'mikegarciaca', 'cardenas', 'schiff', 'sherman', 'aguilar', 'gomez', 'lieu', 'takano',
               'calvert', 'garciaca', 'correa', 'issa', 'levin', 'peters', 'vargas', 'neguse', 'buck', 'lopez',
               'lamborn', 'crow', 'Larson', 'courtney', 'himes', 'gaetz', 'dunn', 'beanFL', 'rutherford', 'waltz',
               'mills', 'posey', 'soto', 'frost', 'webster', 'bilirakis', 'buchanan', 'steube', 'franklin', 'donalds',
               'mast', 'moskowitz', 'diazbalart', 'gimenez', 'carterga', 'bishopGA', 'ferguson', 'johnsonga',
               'mccormick', 'scottga', 'clyde', 'collins', 'loudermilk', 'allen', 'scottga', 'case', 'nunnIA',
               'feenstra', 'fulcher', 'simpson', 'jacksonil', 'garciail', 'quigley', 'casten', 'davisil',
               'krishnamoorthi', 'schneider', 'foster', 'bost', 'lahood', 'sorensen', 'mrvan', 'yakym', 'banks',
               'baird', 'pence', 'carson', 'bucshon', 'mann', 'laturner', 'estes', 'comer', 'guthrie', 'mcgarvey',
               'massie', 'rogersky', 'barr', 'scalise', 'carterLA', 'higginsla', 'johnsonla', 'gravesla', 'neal',
               'mcgovern', 'auchincloss', 'moulton', 'lynch', 'keating', 'harris', 'ruppersberger', 'sarbanes', 'ivey',
               'hoyer', 'trone', 'mfume', 'raskin', 'golden', 'bergman', 'moolenaar', 'huizenga', 'walberg', 'kildee',
               'james', 'thanedar', 'finstad', 'phillips', 'emmer', 'stauber', 'luetkemeyer', 'alford', 'cleaver',
               'gravesmo', 'burlison', 'smithmo', 'kellyms', 'thompsonms', 'guest', 'ezell', 'zinke', 'rosendale',
               'davisnc', 'murphy', 'rouzer', 'bishopNC', 'hudson', 'mchenry', 'edwards', 'nickel', 'jacksonnc',
               'armstrong', 'flood', 'bacon', 'smithne', 'pappas', 'norcross', 'vandrew', 'kimNJ', 'smithnj',
               'gottheimer', 'pallone', 'kean', 'menendez', 'pascrell', 'payne', 'vasquez', 'amodei', 'horsford',
               'lalota', 'garbarino', 'santos', 'suozzi', "D'Esposito", 'meeks', 'jeffries', 'goldman', 'nadler',
               'espaillat', 'torresny', 'bowman', 'lawler', 'ryan', 'molinaro', 'tonko', 'williamsny', 'langworthy',
               'morelle', 'higginsNY', 'kennedy', 'landsman', 'wenstrup', 'jordan', 'latta', 'johnsonoh', 'rulli',
               'milleroh', 'davidson', 'turner', 'balderson', 'joyceoh', 'carey', 'hern', 'brecheen', 'lucas', 'cole',
               'bentz', 'blumenauer', 'fitzpatrick', 'boyle', 'evans', 'cartwright', 'meuser', 'perry', 'smucker',
               'joycepa', 'reschenthaler', 'thompsonpa', 'kellypa', 'deluzio', 'cicilline', 'amo', 'magaziner',
               'wilsonsc', 'duncan', 'timmons', 'norman', 'clyburn', 'fry', 'johnsonsd', 'burchett', 'fleischmann',
               'desjarlais', 'ogles', 'rose', 'greenTN', 'kustoff', 'cohen', 'moran', 'crenshaw', 'self', 'fallon',
               'gooden', 'ellzey', 'luttrell', 'greenTX', 'mccaul', 'pfluger', 'jacksontx', 'weberTX', 'sessions',
               'arrington', 'castrotx', 'roy', 'nehls', 'Gonzales', 'williamstx', 'burgess', 'cloud', 'cuellar',
               'cartertx', 'allred', 'veasey', 'gonzalezTX', 'casar', 'babin', 'doggett', 'hunt', 'mooreut', 'stewart',
               'curtis', 'owens', 'wittman', 'scottva', 'good', 'cline', 'beyer', 'griffith', 'connolly', 'larsenWA',
               'newhouse', 'kilmer', 'smithwa', 'steil', 'pocan', 'vanorden', 'fitzgerald', 'grothman', 'tiffany',
               'gallagher', 'wied', 'mooney', 'austinscottga', 'moylan', 'desposito', 'larsonCN', 'sablan', 'tonygonzaleztx',
                'castro', 'davidscottga', 'cscottfranklin']
    return male_mcs


def precompile_regex_patterns(congress_118_lastnames, state_names, extra_stop_phrases):
    congress_last_names = set(name.lower() for name in congress_118_lastnames)
    state_regex = re.compile(r'\b(?:' + '|'.join(map(re.escape, state_names)) + r')\b', flags=re.IGNORECASE)
    extra_phrases_regex = re.compile(r'\b(?:' + '|'.join(map(re.escape, extra_stop_phrases)) + r')\b',
                                     flags=re.IGNORECASE)
    return congress_last_names, state_regex, extra_phrases_regex
